"""NP-NLI-T — Noise Fingerprint Transformer.

Pure transformer architecture for noise-based tamper detection.
Replaces the CNN UNet (v1/v2) with PVT-v2 encoder + SparseViT-style
selective attention + bidirectional UN decoder.

Architecture:
  Forensic preprocessing (BayarConv + SRM + optional Noiseprint++):
    → 48-49ch noise features (NOT learned — fixed preprocessing)

  Triple-stream patch embedding:
    Stream A: RGB context (3ch) → tokens
    Stream B: Noise features (45ch from multi-scale BayarConv + SRM) → tokens
    Stream C: Noiseprint++ fingerprint (1ch, optional, frozen) → tokens

  Encoder: PVT-v2 backbone with SparseViT-style attention
    Stage 1:  64ch @ H/4   — standard SRA attention
    Stage 2: 128ch @ H/8   — sparse attention (oversample flat/texture regions)
    Stage 3: 320ch @ H/16  — sparse attention
    Stage 4: 512ch @ H/32  — standard attention

  Decoder: Bidirectional UN decoder (top-down + bottom-up, from MUN AAAI 2025)

  Output heads:
    ① Pixel tamper map (H×W)
    ② Confidence map (H×W)
    ③ Boundary map (H×W)
    ④ Trust scalar
    ⑤ Fingerprint anomaly (if Noiseprint++ active)
    ⑥ Deep supervision (3 aux heads)

Source: SparseViT (AAAI 2025), MUN (AAAI 2025), TruFor (CVPR 2023)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_level_inconsistency.pvt_v2 import PVTv2, build_pvt_v2_b2, build_pvt_v2_b1, build_pvt_v2_b0
from noise_level_inconsistency.learned_model_v2 import MultiScaleBayarConv, SRMFilterBank

# ---------------------------------------------------------------------------
# Noise Feature Stem (BayarConv + SRM → tokens)
# ---------------------------------------------------------------------------


class NoiseFeatureStem(nn.Module):
    """Extract forensic noise features and project to token space.

    bayar_target='rgb': MultiScaleBayarConv(3→30ch) + SRM(3→15ch) = 45ch from RGB.
    bayar_target='y':   MultiScaleBayarConv(1→30ch) + SRM(1→15ch) = 45ch from Y channel.
    Then project to embed_dim via overlapping conv stem.
    """

    def __init__(self, embed_dim: int = 64, bayar_target: str = "rgb") -> None:
        super().__init__()
        self.bayar_target = bayar_target
        noise_in_ch = 1 if bayar_target == "y" else 3
        self.bayar = MultiScaleBayarConv(in_channels=noise_in_ch,
                                         out_channels_per_scale=10)  # 30ch
        self.srm = SRMFilterBank(in_channels=noise_in_ch,
                                 out_channels=15)  # 15ch
        # 45ch → embed_dim via progressive conv
        self.proj = nn.Sequential(
            nn.Conv2d(45, embed_dim, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1,
                      bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1,
                      bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )  # output: embed_dim @ H/4

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args: rgb (B, 3, H, W) — always RGB; Y is computed internally if needed.
        Returns: noise_tokens (B, embed_dim, H/4, W/4)
        """
        if self.bayar_target == "y":
            # ITU-R BT.601 luminance
            noise_input = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] +
                           0.114 * rgb[:, 2:3])
        else:
            noise_input = rgb
        bayar_out = self.bayar(noise_input)  # (B, 30, H, W)
        srm_out = self.srm(noise_input)  # (B, 15, H, W)
        noise = torch.cat([bayar_out, srm_out], dim=1)  # (B, 45, H, W)
        return self.proj(noise)


# ---------------------------------------------------------------------------
# Block-level noise feature stem (PDF §7.1)
# ---------------------------------------------------------------------------


def _make_gaussian_blur_conv(sigma: float = 2.0) -> nn.Conv2d:
    """Fixed frozen Gaussian blur Conv2d (1-in, 1-out). Used by BlockNoiseFeatureStem."""
    k = int(2 * round(2 * sigma) + 1) | 1
    x = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    conv = nn.Conv2d(1, 1, k, padding=k // 2, bias=False)
    conv.weight.data = kernel
    for p in conv.parameters():
        p.requires_grad = False
    return conv


class BlockNoiseFeatureStem(nn.Module):
    """Compute block-level noise features (N_hp, N_sigma) from RGB; project to H/4.

    PDF §7.1: inject per-block noise statistics as additional model input alongside
    the BayarConv/SRM stream. Fixed frozen operators for feature computation;
    learned projection to embed_dim.
    """

    def __init__(self, embed_dim: int = 64, block_size: int = 16) -> None:
        super().__init__()
        self.block_size = block_size
        self.blur = _make_gaussian_blur_conv(sigma=2.0)
        self.proj = nn.Sequential(
            nn.Conv2d(2, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1,
                      bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )  # 2ch → embed_dim @ H/4

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # Luminance Y (BT.601)
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        blurred = self.blur(y)
        n_hp = (y - blurred).abs()  # (B, 1, H, W)
        # N_sigma proxy: per-block RMS of n_hp (sqrt of avg n_hp^2 per block)
        bs = self.block_size
        n_sq = F.avg_pool2d(n_hp**2, kernel_size=bs, stride=bs, padding=0)
        n_sigma = F.interpolate(n_sq.sqrt(),
                                size=n_hp.shape[2:],
                                mode="nearest")
        noise_2ch = torch.cat([n_hp, n_sigma], dim=1)
        return self.proj(noise_2ch)  # (B, embed_dim, H/4, W/4)


# ---------------------------------------------------------------------------
# Stream Fusion (RGB + noise tokens → fused tokens)
# ---------------------------------------------------------------------------


class StreamFusion(nn.Module):
    """Fuse RGB encoder tokens with noise stem tokens at H/4.

    Concat + 1×1 projection + LayerNorm.
    """

    def __init__(self, rgb_dim: int, noise_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(rgb_dim + noise_dim, out_dim, 1, bias=False),
            nn.GroupNorm(min(8, out_dim), out_dim),
            nn.GELU(),
        )

    def forward(self, rgb_feat: torch.Tensor,
                noise_feat: torch.Tensor) -> torch.Tensor:
        """Both inputs: (B, C, H/4, W/4). Returns: (B, out_dim, H/4, W/4)."""
        if noise_feat.shape[2:] != rgb_feat.shape[2:]:
            noise_feat = F.interpolate(noise_feat,
                                       size=rgb_feat.shape[2:],
                                       mode="bilinear",
                                       align_corners=False)
        return self.proj(torch.cat([rgb_feat, noise_feat], dim=1))


# ---------------------------------------------------------------------------
# Bidirectional UN Decoder (from MUN, AAAI 2025)
# ---------------------------------------------------------------------------


class BidirectionalDecoderBlock(nn.Module):
    """Single decoder level with bidirectional fusion."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        g = min(8, out_ch)
        while out_ch % g != 0:
            g -= 1
        # Top-down path
        self.td_conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
        # Bottom-up refinement gate
        self.bu_gate = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self,
                x: torch.Tensor,
                skip: torch.Tensor,
                bu_hint: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: from deeper level (upsampled)
        skip: encoder skip connection at this level
        bu_hint: bottom-up feature hint (optional, from previous pass)
        """
        x = F.interpolate(x,
                          size=skip.shape[2:],
                          mode="bilinear",
                          align_corners=False)
        td = self.td_conv(torch.cat([x, skip], dim=1))
        if bu_hint is not None:
            bu_hint = F.interpolate(bu_hint,
                                    size=td.shape[2:],
                                    mode="bilinear",
                                    align_corners=False)
            gate = self.bu_gate(bu_hint)
            td = td * (1 + gate)  # modulate top-down with bottom-up signal
        return td


class BidirectionalDecoder(nn.Module):
    """UN-style bidirectional decoder (MUN, AAAI 2025).

    Pass 1 (bottom-up): coarse features from fine encoder features
    Pass 2 (top-down): progressive upsample guided by bottom-up hints
    """

    def __init__(self, encoder_dims: list[int], out_ch: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_dims

        # Top-down decoder blocks
        self.up3 = BidirectionalDecoderBlock(c4, c3, c3)
        self.up2 = BidirectionalDecoderBlock(c3, c2, c2)
        self.up1 = BidirectionalDecoderBlock(c2, c1, c1)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

        # Bottom-up path (lightweight — just pools + projects)
        self.bu_proj1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.bu_proj2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False)

        # Deep supervision aux heads
        self.aux_h16 = nn.Conv2d(c3, 1, 1)
        self.aux_h8 = nn.Conv2d(c2, 1, 1)
        self.aux_h4 = nn.Conv2d(c1, 1, 1)

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        f1, f2, f3, f4 = features  # H/4, H/8, H/16, H/32

        # Bottom-up hints: f1 → pool → f2 scale, f2 → pool → f3 scale
        bu_h8 = self.bu_proj1(f1)  # approximate bottom-up at H/8
        bu_h16 = self.bu_proj2(f2)  # approximate bottom-up at H/16

        # Top-down with bottom-up modulation
        d3 = self.up3(f4, f3, bu_hint=bu_h16)  # H/16
        d2 = self.up2(d3, f2, bu_hint=bu_h8)  # H/8
        d1 = self.up1(d2, f1, bu_hint=None)  # H/4
        d0 = self.up0(d1)  # H

        aux = [self.aux_h16(d3), self.aux_h8(d2), self.aux_h4(d1)]
        return d0, aux


# ---------------------------------------------------------------------------
# NP-NLI-T: Full Transformer Model
# ---------------------------------------------------------------------------


class NPNLITransformer(nn.Module):
    """Noise Fingerprint Transformer for tamper detection.

    Parameters
    ----------
    backbone_size : str
        PVT-v2 size: 'b0', 'b1', 'b2'. Default 'b2'.
    use_noiseprint : bool
        Enable Noiseprint++ fingerprint branch (frozen). Default False.
    decoder_out_ch : int
        Decoder output channels before heads. Default 64.
    bayar_target : str
        'rgb' — Bayar/SRM on RGB (3ch); 'y' — Bayar/SRM on Y luminance (1ch).
    """

    def __init__(
        self,
        backbone_size: str = "b2",
        use_noiseprint: bool = False,
        decoder_out_ch: int = 64,
        bayar_target: str = "rgb",
        use_block_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_noiseprint = use_noiseprint
        self.bayar_target = bayar_target
        self.use_block_features = use_block_features

        # Build PVT-v2 backbone for RGB stream (always 3ch)
        builders = {
            "b0": build_pvt_v2_b0,
            "b1": build_pvt_v2_b1,
            "b2": build_pvt_v2_b2,
        }
        self.rgb_encoder = builders[backbone_size](in_channels=3)
        embed_dims = self.rgb_encoder.embed_dims  # [64, 128, 320, 512] for B2

        # Noise feature stem (BayarConv + SRM → H/4 tokens)
        self.noise_stem = NoiseFeatureStem(embed_dim=embed_dims[0],
                                           bayar_target=bayar_target)

        # Stream fusion at H/4 (RGB stage 1 + noise stem)
        self.stream_fusion = StreamFusion(embed_dims[0], embed_dims[0],
                                          embed_dims[0])

        # Block-level noise features (PDF §7.1) — gated residual, gate init=0
        if use_block_features:
            self.block_feat_stem = BlockNoiseFeatureStem(
                embed_dim=embed_dims[0])
            self.block_feat_gate = nn.Parameter(torch.zeros(1))

        # Noiseprint++ branch (optional, frozen)
        if use_noiseprint:
            from noise_level_inconsistency.noiseprint import NoiseprintEncoder
            self.noiseprint_encoder = NoiseprintEncoder(in_channels=3,
                                                        mid_channels=64,
                                                        num_layers=12,
                                                        out_channels=1)
            for p in self.noiseprint_encoder.parameters():
                p.requires_grad = False
            self.fp_proj = nn.Sequential(
                nn.Conv2d(1, embed_dims[0], 3, stride=4, padding=1,
                          bias=False),
                nn.GroupNorm(8, embed_dims[0]),
                nn.GELU(),
            )

        # Bidirectional decoder
        self.decoder = BidirectionalDecoder(embed_dims, out_ch=decoder_out_ch)

        # Primary output heads
        self.pixel_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.confidence_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.boundary_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dims[3], 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # Auxiliary noise-profile heads (intermediate supervision)
        # noise_residual_head predicts N_hp (high-pass residual), range [0,1]
        self.noise_residual_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch,
                      decoder_out_ch // 2,
                      3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(min(8, decoder_out_ch // 2), decoder_out_ch // 2),
            nn.GELU(),
            nn.Conv2d(decoder_out_ch // 2, 1, 1),
            nn.Sigmoid(),
        )
        # sigma_head predicts N_sigma (per-block noise level), range [0,∞) → clipped by target [0,1]
        self.sigma_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch,
                      decoder_out_ch // 2,
                      3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(min(8, decoder_out_ch // 2), decoder_out_ch // 2),
            nn.GELU(),
            nn.Conv2d(decoder_out_ch // 2, 1, 1),
            nn.ReLU(),
        )
        # jpeg_head predicts E_jpeg (JPEG artifact map), range [0,1]
        self.jpeg_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch,
                      decoder_out_ch // 2,
                      3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(min(8, decoder_out_ch // 2), decoder_out_ch // 2),
            nn.GELU(),
            nn.Conv2d(decoder_out_ch // 2, 1, 1),
            nn.Sigmoid(),
        )

        self._init_heads()

    def _init_heads(self) -> None:
        for head in [
                self.pixel_head, self.confidence_head, self.boundary_head
        ]:
            nn.init.zeros_(head.bias)
            nn.init.kaiming_normal_(head.weight)
        for m in self.trust_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head_seq in [
                self.noise_residual_head, self.sigma_head, self.jpeg_head
        ]:
            for m in head_seq.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def load_noiseprint_weights(self, checkpoint_path: str) -> None:
        if not self.use_noiseprint:
            raise RuntimeError("Noiseprint++ not enabled")
        state = torch.load(checkpoint_path,
                           map_location="cpu",
                           weights_only=True)
        if "model_state" in state:
            state = state["model_state"]
        self.noiseprint_encoder.load_state_dict(state)
        for p in self.noiseprint_encoder.parameters():
            p.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, H, W) RGB image in [0, 1]

        Returns:
            dict with keys: pixel, confidence, boundary, trust, aux,
                            n_hp_pred, n_sigma_pred, e_jpeg_pred,
                            fingerprint (if noiseprint active)
        """
        # ── RGB encoder (PVT-v2) ──────────────────────────────────────
        rgb_features = self.rgb_encoder(
            rgb)  # [f1@H/4, f2@H/8, f3@H/16, f4@H/32]

        # ── Noise stem (BayarConv + SRM on rgb or Y) ─────────────────
        noise_tokens = self.noise_stem(rgb)  # (B, C1, H/4, W/4)

        # ── Stream fusion at H/4 ─────────────────────────────────────
        fused_f1 = self.stream_fusion(rgb_features[0], noise_tokens)

        # ── Block-level noise features (PDF §7.1) ────────────────────
        if self.use_block_features:
            block_tokens = self.block_feat_stem(rgb)
            fused_f1 = fused_f1 + torch.tanh(
                self.block_feat_gate) * block_tokens

        # ── Noiseprint++ fingerprint (optional) ───────────────────────
        fingerprint = None
        if self.use_noiseprint:
            with torch.no_grad():
                fingerprint = self.noiseprint_encoder(rgb)  # (B, 1, H, W)
            fp_tokens = self.fp_proj(fingerprint)  # (B, C1, H/4, W/4)
            fused_f1 = fused_f1 + fp_tokens

        features = [
            fused_f1, rgb_features[1], rgb_features[2], rgb_features[3]
        ]

        # ── Trust head (from bottleneck) ──────────────────────────────
        trust = self.trust_head(features[3])

        # ── Bidirectional decoder ─────────────────────────────────────
        decoded, aux = self.decoder(features)

        # ── Output heads ──────────────────────────────────────────────
        result = {
            "pixel": self.pixel_head(decoded),
            "confidence": self.confidence_head(decoded),
            "boundary": self.boundary_head(decoded),
            "trust": trust,
            "aux": aux if self.training else [],
            "n_hp_pred": self.noise_residual_head(decoded),  # (B,1,H,W) [0,1]
            "n_sigma_pred": self.sigma_head(decoded),  # (B,1,H,W) >=0
            "e_jpeg_pred": self.jpeg_head(decoded),  # (B,1,H,W) [0,1]
        }
        if fingerprint is not None:
            result["fingerprint"] = fingerprint

        return result

    def extract_encoder_features(self,
                                 rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale features for SG-DFT fusion."""
        rgb_features = self.rgb_encoder(rgb)
        noise_tokens = self.noise_stem(rgb)
        fused_f1 = self.stream_fusion(rgb_features[0], noise_tokens)

        if self.use_noiseprint:
            with torch.no_grad():
                fp = self.noiseprint_encoder(rgb)
            fused_f1 = fused_f1 + self.fp_proj(fp)

        if self.use_block_features:
            block_tokens = self.block_feat_stem(rgb)
            fused_f1 = fused_f1 + torch.tanh(
                self.block_feat_gate) * block_tokens

        return {
            "h4": fused_f1,
            "h8": rgb_features[1],
            "h16": rgb_features[2],
            "h32": rgb_features[3],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_nli_transformer(
    backbone_size: str = "b2",
    use_noiseprint: bool = False,
    decoder_out_ch: int = 64,
    bayar_target: str = "rgb",
    use_block_features: bool = False,
) -> NPNLITransformer:
    """Build the NP-NLI Transformer model.

    bayar_target: 'rgb' uses RGB for Bayar/SRM (default V0 behaviour);
                  'y' routes Y-channel luminance to Bayar/SRM (V0.1 ablation).
    use_block_features: inject block-level N_hp+N_sigma features (PDF §7.1).
    """
    return NPNLITransformer(
        backbone_size=backbone_size,
        use_noiseprint=use_noiseprint,
        decoder_out_ch=decoder_out_ch,
        bayar_target=bayar_target,
        use_block_features=use_block_features,
    )
