"""V0.4 — NPNLITransformer student with teacher-aux 4th input channel.

Subclass of NPNLITransformer that accepts (B, 4, H, W) input where:
    channel 0..2 = RGB
    channel 3    = teacher's n_hp_pred (precomputed offline)

The 4th channel feeds ONLY the PVT-v2 RGB encoder (via a widened patch_embed1.proj).
The internal noise stem (BayarConv/SRM), block-feature stem, and Noiseprint++
branch keep their 3-channel RGB inputs — only the encoder gets the auxiliary signal.

Warm-start helper `load_175116_into_v04` loads V0.1's checkpoint into the V04 model:
all matching keys load directly; patch_embed1.proj's RGB weights copy into
channels 0..2, channel 3 initializes as `mean(RGB_weights)`.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from noise_level_inconsistency.learned_model_transformer import NPNLITransformer


class NPNLITransformerV04(NPNLITransformer):
    """V0.4 student: NPNLITransformer with 4-channel input (RGB + teacher n_hp)."""

    def __init__(
        self,
        backbone_size: str = "b2",
        use_noiseprint: bool = False,
        decoder_out_ch: int = 64,
        bayar_target: str = "rgb",
        use_block_features: bool = False,
    ) -> None:
        super().__init__(
            backbone_size=backbone_size,
            use_noiseprint=use_noiseprint,
            decoder_out_ch=decoder_out_ch,
            bayar_target=bayar_target,
            use_block_features=use_block_features,
        )
        self._widen_patch_embed1_to_4ch()

    def _widen_patch_embed1_to_4ch(self) -> None:
        old_proj: nn.Conv2d = self.rgb_encoder.patch_embed1.proj
        assert old_proj.in_channels == 3, (
            f"expected 3-channel input conv, got {old_proj.in_channels}")

        new_proj = nn.Conv2d(
            in_channels=4,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight[:, :3].copy_(old_proj.weight)
            new_proj.weight[:, 3:4].copy_(
                old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.rgb_encoder.patch_embed1.proj = new_proj

    def forward(self, rgb: torch.Tensor) -> dict:
        """Args:
            rgb: (B, 4, H, W) — channels 0..2 RGB, channel 3 teacher n_hp.
        """
        assert rgb.shape[1] == 4, (
            f"V04 expects 4-channel input (RGB+teacher), got {rgb.shape[1]}")
        rgb_3ch = rgb[:, :3]

        rgb_features = self.rgb_encoder(rgb)
        noise_tokens = self.noise_stem(rgb_3ch)
        fused_f1 = self.stream_fusion(rgb_features[0], noise_tokens)

        if self.use_block_features:
            block_tokens = self.block_feat_stem(rgb_3ch)
            fused_f1 = fused_f1 + torch.tanh(
                self.block_feat_gate) * block_tokens

        fingerprint = None
        if self.use_noiseprint:
            with torch.no_grad():
                fingerprint = self.noiseprint_encoder(rgb_3ch)
            fp_tokens = self.fp_proj(fingerprint)
            fused_f1 = fused_f1 + fp_tokens

        features = [
            fused_f1, rgb_features[1], rgb_features[2], rgb_features[3]
        ]
        trust = self.trust_head(features[3])
        decoded, aux = self.decoder(features)

        result = {
            "pixel": self.pixel_head(decoded),
            "confidence": self.confidence_head(decoded),
            "boundary": self.boundary_head(decoded),
            "trust": trust,
            "aux": aux if self.training else [],
            "n_hp_pred": self.noise_residual_head(decoded),
            "n_sigma_pred": self.sigma_head(decoded),
            "e_jpeg_pred": self.jpeg_head(decoded),
        }
        if fingerprint is not None:
            result["fingerprint"] = fingerprint
        return result

    def extract_encoder_features(self, rgb: torch.Tensor) -> dict:
        rgb_3ch = rgb[:, :3]
        rgb_features = self.rgb_encoder(rgb)
        noise_tokens = self.noise_stem(rgb_3ch)
        fused_f1 = self.stream_fusion(rgb_features[0], noise_tokens)

        if self.use_noiseprint:
            with torch.no_grad():
                fp = self.noiseprint_encoder(rgb_3ch)
            fused_f1 = fused_f1 + self.fp_proj(fp)

        if self.use_block_features:
            block_tokens = self.block_feat_stem(rgb_3ch)
            fused_f1 = fused_f1 + torch.tanh(
                self.block_feat_gate) * block_tokens

        return {
            "h4": fused_f1,
            "h8": rgb_features[1],
            "h16": rgb_features[2],
            "h32": rgb_features[3],
        }


def build_nli_transformer_v04(
    backbone_size: str = "b2",
    use_noiseprint: bool = False,
    decoder_out_ch: int = 64,
    bayar_target: str = "rgb",
    use_block_features: bool = False,
) -> NPNLITransformerV04:
    return NPNLITransformerV04(
        backbone_size=backbone_size,
        use_noiseprint=use_noiseprint,
        decoder_out_ch=decoder_out_ch,
        bayar_target=bayar_target,
        use_block_features=use_block_features,
    )


def load_175116_into_v04(
    model: NPNLITransformerV04,
    ckpt_path: str | Path,
) -> tuple[int, int]:
    """Warm-start V04 from a 3-channel V1 checkpoint (e.g. nli_transformer_20260423_175116).

    Behavior:
        - All matching-shape keys load via filtered load_state_dict.
        - patch_embed1.proj.weight is shape-mismatched (3ch vs 4ch) — handled
          manually: ckpt RGB weights → channels 0..2; channel 3 ← mean(RGB).
        - patch_embed1.proj.bias copies as-is (output dim unchanged).

    Returns: (n_loaded, n_skipped) including the 1 manually-handled key.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))

    ms = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items() if k in ms and v.shape == ms[k].shape
    }
    skipped = [
        k for k in state.keys()
        if k not in ms or (k in ms and state[k].shape != ms[k].shape)
    ]
    model.load_state_dict(filtered, strict=False)

    # Manually handle the 3→4 channel patch_embed1.proj
    pe_key = "rgb_encoder.patch_embed1.proj.weight"
    if pe_key in state:
        old_w = state[pe_key]
        new_proj: nn.Conv2d = model.rgb_encoder.patch_embed1.proj
        if old_w.shape[1] == 3 and new_proj.weight.shape[1] == 4:
            with torch.no_grad():
                new_proj.weight[:, :3].copy_(old_w)
                new_proj.weight[:, 3:4].copy_(old_w.mean(dim=1, keepdim=True))
                bias_key = "rgb_encoder.patch_embed1.proj.bias"
                if bias_key in state and new_proj.bias is not None:
                    new_proj.bias.copy_(state[bias_key])
            # Mark the patch_embed1.proj keys as loaded for the count
            n_loaded = len(filtered) + 1
            if "rgb_encoder.patch_embed1.proj.bias" in state and new_proj.bias is not None:
                n_loaded += 1
            return n_loaded, len(skipped) - n_loaded + len(filtered)

    return len(filtered), len(skipped)
