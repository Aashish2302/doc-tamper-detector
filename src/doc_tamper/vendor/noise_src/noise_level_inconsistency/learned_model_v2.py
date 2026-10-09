"""NP-NLI v2 — Noiseprint-Enhanced Noise Level Inconsistency detector.

Upgrades over v1 (learned_model.py):
  1. Multi-Scale BayarConv: 3×3 + 5×5 + 7×7 constrained high-pass filters
     instead of a single 5×5.  Captures noise at multiple spatial scales.
  2. Optional Noiseprint++ branch (Branch C): DnCNN-style self-supervised
     fingerprint.  Pre-trained on authentic images, frozen during NLI training.
     Enabled via config — test independently before combining with Branch B.
  3. Fingerprint anomaly map: additional output head when Noiseprint++ is active.
  4. Deeper encoder: 5-level instead of 4 (adds 1024ch level before bottleneck).
  5. Boundary auxiliary head: like ELA/SEAM, predict tamper boundaries too.
  6. Enhanced trust head: acquisition-type-aware (lower trust on rendered docs).

Architecture
------------
Branch A (RGB context):
    RGB (3ch) → 2× DoubleConv (3→48→96) with GroupNorm(8)

Branch B (Enhanced forensic noise):
    RGB → MultiScaleBayarConv(3→30ch, kernels 3+5+7)
    RGB → SRMFilterBank(3→15ch, 30 fixed kernels)
    Concat: 30+15 = 45ch → 2× DoubleConv (45→48→96)

Branch C (Noiseprint++ fingerprint, OPTIONAL):
    RGB → NoiseprintEncoder (frozen, pre-trained) → 1ch fingerprint
    → expand via DoubleConv (1→48→96)

Fusion: concat branches → skip fusion via 1×1 conv at 2 levels

Encoder: 192→384→512→768→1024  (5 levels)
Decoder: 1024→768→512→384→192  (with fused skip connections)

Heads:
    - Pixel tamper map:      192→1 (sigmoid)
    - Pixel confidence:      192→1 (sigmoid)
    - Boundary map:          192→1 (sigmoid)  [NEW in v2]
    - Fingerprint anomaly:   1ch (from Noiseprint++ branch, if enabled)
    - Image-level trust:     GAP(1024) → 512 → 256 → 1 (sigmoid)

Build order (from FORENSIC_METHODS.md):
    Start with Option B (Multi-Scale BayarConv + SRM — proven, works on all docs).
    Test Option A (Noiseprint++) on scanned documents separately.
    Combine only if both contribute.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-Scale BayarConv2d — constrained high-pass at 3 kernel sizes
# ---------------------------------------------------------------------------

class BayarConv2d(nn.Module):
    """Constrained convolution: centre weight = -sum(rest) → guaranteed high-pass.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
        Must be odd (3, 5, or 7).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.kernel_size = kernel_size
        self.center = kernel_size // 2
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.clone()
        w[:, :, self.center, self.center] = 0
        rest_sum = w.sum(dim=(2, 3))
        w[:, :, self.center, self.center] = -rest_sum
        return F.conv2d(x, w, self.bias, padding=self.padding)


class MultiScaleBayarConv(nn.Module):
    """Multi-scale constrained high-pass: 3×3 + 5×5 + 7×7 BayarConv in parallel.

    Captures noise at different spatial scales:
      - 3×3: fine-grained, pixel-level noise
      - 5×5: medium-range noise correlations
      - 7×7: broader noise structure

    Output: concatenation of all three → out_channels_per_scale × 3 channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels_per_scale: int = 10,
    ) -> None:
        super().__init__()
        self.bayar_3 = BayarConv2d(in_channels, out_channels_per_scale, kernel_size=3)
        self.bayar_5 = BayarConv2d(in_channels, out_channels_per_scale, kernel_size=5)
        self.bayar_7 = BayarConv2d(in_channels, out_channels_per_scale, kernel_size=7)
        self.out_channels = out_channels_per_scale * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.bayar_3(x),
            self.bayar_5(x),
            self.bayar_7(x),
        ], dim=1)


# ---------------------------------------------------------------------------
# SRM Filter Bank (same as v1 — fixed, non-learnable)
# ---------------------------------------------------------------------------

def _build_srm_kernels_30() -> np.ndarray:
    """Build 30 SRM kernels (5×5) from steganalysis literature.

    All kernels sum to zero (high-pass property).  Not learnable.
    """
    kernels: list[np.ndarray] = []

    # Group 1: 1st-order differences
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        k = np.zeros((5, 5), dtype=np.float32)
        k[2, 2] = -1.0
        k[2 + dy, 2 + dx] = 1.0
        kernels.append(k)

    # Group 2: 2nd-order differences
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        k = np.zeros((5, 5), dtype=np.float32)
        k[2 - dy, 2 - dx] = 1.0
        k[2, 2] = -2.0
        k[2 + dy, 2 + dx] = 1.0
        kernels.append(k)

    # Group 3: 3rd-order horizontal/vertical
    for axis in range(2):
        for direction in [1, -1]:
            k = np.zeros((5, 5), dtype=np.float32)
            if axis == 0:  # horizontal
                k[2, 2] = -3.0 * direction
                k[2, 2 + direction] = 3.0 * direction
                k[2, 2 + 2 * direction] = -1.0 * direction
                k[2, 2 - direction] = 1.0 * direction
            else:  # vertical
                k[2, 2] = -3.0 * direction
                k[2 + direction, 2] = 3.0 * direction
                k[2 + 2 * direction, 2] = -1.0 * direction
                k[2 - direction, 2] = 1.0 * direction
            kernels.append(k)

    # Group 4: Laplacian variants
    k = np.zeros((5, 5), dtype=np.float32)
    k[1:4, 1:4] = -1; k[2, 2] = 8
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 2] = 1; k[2, 1] = 1; k[2, 2] = -4; k[2, 3] = 1; k[3, 2] = 1
    kernels.append(k)

    # Group 5: Prewitt/Sobel
    for weights in [([1, 1, 1], [0, 0, 0], [-1, -1, -1]),
                    ([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]),
                    ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]),
                    ([-1, -2, -1], [0, 0, 0], [1, 2, 1])]:
        k = np.zeros((5, 5), dtype=np.float32)
        for r, row in enumerate(weights):
            for c, v in enumerate(row):
                k[r + 1, c + 1] = v
        kernels.append(k)

    # Group 6: Long-range 2nd order
    for dy, dx in [(2, 0), (0, 2), (2, 2), (2, -2)]:
        k = np.zeros((5, 5), dtype=np.float32)
        k[2 - dy, 2 - dx] = 1; k[2, 2] = -2; k[2 + dy, 2 + dx] = 1
        kernels.append(k)

    # Group 7: SPAM-like
    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 0] = -0.25; k[2, 1] = 0.5; k[2, 3] = -0.5; k[2, 4] = 0.25
    kernels.append(k)
    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 2] = -0.25; k[1, 2] = 0.5; k[3, 2] = -0.5; k[4, 2] = 0.25
    kernels.append(k)

    # Pad to 30 with rotations
    while len(kernels) < 30:
        k_rot = np.rot90(kernels[len(kernels) % 14]).copy()
        kernels.append(k_rot.astype(np.float32))

    return np.stack(kernels[:30], axis=0)


class SRMFilterBank(nn.Module):
    """Fixed SRM filter bank: 30 kernels → 1×1 learnable projection."""

    def __init__(self, in_channels: int = 3, out_channels: int = 15) -> None:
        super().__init__()
        srm_np = _build_srm_kernels_30()
        weight = np.zeros((30, in_channels, 5, 5), dtype=np.float32)
        for c in range(in_channels):
            weight[:, c, :, :] = srm_np / in_channels
        self.register_buffer("srm_weight", torch.from_numpy(weight))
        self.register_buffer("srm_bias", torch.zeros(30))
        self.project = nn.Conv2d(30, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        srm_out = F.conv2d(x, self.srm_weight, self.srm_bias, padding=2)
        return self.project(srm_out)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two 3×3 convolutions each followed by GroupNorm and GELU."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        # Ensure groups divides out_ch
        g = min(groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """MaxPool2d(2) → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# NP-NLI v2 Model
# ---------------------------------------------------------------------------

class NPNLIUNet(nn.Module):
    """Noiseprint-Enhanced Noise Level Inconsistency detector (v2).

    Triple-branch architecture:
        Branch A: RGB context features
        Branch B: Multi-scale BayarConv + SRM forensic noise features
        Branch C: Noiseprint++ fingerprint features (OPTIONAL, frozen)

    Parameters
    ----------
    use_noiseprint : bool
        If True, Branch C (Noiseprint++) is active.  Requires a pre-trained
        NoiseprintEncoder to be loaded via ``load_noiseprint_weights()``.
    base_channels : int
        Width multiplier for encoder/decoder (default 48).
    bayar_channels_per_scale : int
        Output channels per BayarConv scale (default 10, total = 30).
    srm_channels : int
        SRM output channels after 1×1 projection (default 15).
    """

    def __init__(
        self,
        use_noiseprint: bool = False,
        base_channels: int = 48,
        bayar_channels_per_scale: int = 10,
        srm_channels: int = 15,
    ) -> None:
        super().__init__()
        self.use_noiseprint = use_noiseprint
        c = base_channels  # 48

        # ══════════════════════════════════════════════════════════════════
        # Branch A: RGB context
        # ══════════════════════════════════════════════════════════════════
        self.branch_a_block1 = DoubleConv(3, c)                    # 48ch @ H
        self.branch_a_pool1 = nn.MaxPool2d(2)
        self.branch_a_block2 = DoubleConv(c, c * 2)               # 96ch @ H/2

        # ══════════════════════════════════════════════════════════════════
        # Branch B: Enhanced forensic noise (multi-scale Bayar + SRM)
        # ══════════════════════════════════════════════════════════════════
        bayar_total = bayar_channels_per_scale * 3  # 30
        noise_concat = bayar_total + srm_channels    # 45

        self.multi_bayar = MultiScaleBayarConv(3, bayar_channels_per_scale)
        self.srm = SRMFilterBank(3, srm_channels)
        self.branch_b_block1 = DoubleConv(noise_concat, c)        # 48ch @ H
        self.branch_b_pool1 = nn.MaxPool2d(2)
        self.branch_b_block2 = DoubleConv(c, c * 2)               # 96ch @ H/2

        # ══════════════════════════════════════════════════════════════════
        # Branch C: Noiseprint++ fingerprint (optional, frozen)
        # ══════════════════════════════════════════════════════════════════
        if use_noiseprint:
            from noise_level_inconsistency.noiseprint import NoiseprintEncoder
            self.noiseprint_encoder = NoiseprintEncoder(
                in_channels=3, mid_channels=64, num_layers=12, out_channels=1,
            )
            # Freeze by default — will be loaded with pre-trained weights
            for p in self.noiseprint_encoder.parameters():
                p.requires_grad = False
            self.branch_c_block1 = DoubleConv(1, c)               # 48ch @ H
            self.branch_c_pool1 = nn.MaxPool2d(2)
            self.branch_c_block2 = DoubleConv(c, c * 2)           # 96ch @ H/2

        # ══════════════════════════════════════════════════════════════════
        # Skip fusion: 1×1 convs to compress concatenated branch features
        # ══════════════════════════════════════════════════════════════════
        n_branches = 3 if use_noiseprint else 2
        skip0_in = c * n_branches       # 48*2=96 or 48*3=144
        skip1_in = c * 2 * n_branches   # 96*2=192 or 96*3=288

        self.skip_fuse_0 = nn.Sequential(
            nn.Conv2d(skip0_in, c * 2, kernel_size=1, bias=False),      # → 96ch @ H
            nn.GroupNorm(8, c * 2), nn.GELU(),
        )
        self.skip_fuse_1 = nn.Sequential(
            nn.Conv2d(skip1_in, c * 4, kernel_size=1, bias=False),      # → 192ch @ H/2
            nn.GroupNorm(8, c * 4), nn.GELU(),
        )

        # ══════════════════════════════════════════════════════════════════
        # Encoder (5-level, on fused features)
        # ══════════════════════════════════════════════════════════════════
        self.enc1 = DownBlock(c * 4, c * 8)       # 384ch @ H/4
        self.enc2 = DownBlock(c * 8, c * 10)      # 480ch @ H/8    (was 512 in v1, adjusted for GroupNorm)
        self.enc3 = DownBlock(c * 10, c * 16)     # 768ch @ H/16
        self.enc4 = DownBlock(c * 16, c * 21)     # 1008ch @ H/32  (bottleneck, divisible by 8)

        # ══════════════════════════════════════════════════════════════════
        # Decoder (5-level, with skip connections)
        # ══════════════════════════════════════════════════════════════════
        self.dec4 = UpBlock(c * 21, c * 16, c * 16)     # 768ch @ H/16
        self.dec3 = UpBlock(c * 16, c * 10, c * 10)     # 480ch @ H/8
        self.dec2 = UpBlock(c * 10, c * 8, c * 8)       # 384ch @ H/4
        self.dec1 = UpBlock(c * 8, c * 4, c * 4)        # 192ch @ H/2
        self.dec0 = UpBlock(c * 4, c * 2, c * 4)        # 192ch @ H

        # ══════════════════════════════════════════════════════════════════
        # Output heads
        # ══════════════════════════════════════════════════════════════════
        head_in = c * 4  # 192

        # Pixel tamper probability
        self.pixel_head = nn.Conv2d(head_in, 1, kernel_size=1)

        # Pixel confidence (self-assessed reliability)
        self.confidence_head = nn.Conv2d(head_in, 1, kernel_size=1)

        # Boundary auxiliary head (tamper region boundaries)
        self.boundary_head = nn.Conv2d(head_in, 1, kernel_size=1)

        # Image-level trust classifier (deeper than v1 for better calibration)
        bottleneck_ch = c * 21  # 1008
        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        # Deep supervision auxiliary heads (training only)
        self.aux_head_h16 = nn.Conv2d(c * 16, 1, kernel_size=1)  # @ H/16
        self.aux_head_h8 = nn.Conv2d(c * 10, 1, kernel_size=1)   # @ H/8
        self.aux_head_h4 = nn.Conv2d(c * 8, 1, kernel_size=1)    # @ H/4

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_noiseprint_weights(self, checkpoint_path: str) -> None:
        """Load pre-trained Noiseprint++ encoder weights."""
        if not self.use_noiseprint:
            raise RuntimeError("Noiseprint++ is not enabled in this model")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state" in state:
            state = state["model_state"]
        self.noiseprint_encoder.load_state_dict(state)
        # Re-freeze
        for p in self.noiseprint_encoder.parameters():
            p.requires_grad = False
        print(f"[NP-NLI] Loaded Noiseprint++ weights from {checkpoint_path}")

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, 3, H, W) RGB input in [0, 1]

        Returns
        -------
        dict with keys:
            pixel:       (B, 1, H, W)  — tamper probability logits
            confidence:  (B, 1, H, W)  — pixel confidence logits
            boundary:    (B, 1, H, W)  — boundary logits
            trust:       (B, 1)        — image-level trust logit
            fingerprint: (B, 1, H, W)  — raw fingerprint (only if use_noiseprint)
            aux:         list of (B, 1, H_i, W_i) — deep supervision (training only)
        """
        # ── Branch A: RGB context ─────────────────────────────────────────
        a0 = self.branch_a_block1(x)                          # (B, 48, H, W)
        a1 = self.branch_a_block2(self.branch_a_pool1(a0))    # (B, 96, H/2, W/2)

        # ── Branch B: Enhanced forensic noise ─────────────────────────────
        bayar_out = self.multi_bayar(x)                        # (B, 30, H, W)
        srm_out = self.srm(x)                                  # (B, 15, H, W)
        noise_feat = torch.cat([bayar_out, srm_out], dim=1)    # (B, 45, H, W)
        b0 = self.branch_b_block1(noise_feat)                  # (B, 48, H, W)
        b1 = self.branch_b_block2(self.branch_b_pool1(b0))    # (B, 96, H/2, W/2)

        # ── Branch C: Noiseprint++ fingerprint (optional) ─────────────────
        fingerprint = None
        if self.use_noiseprint:
            with torch.no_grad():
                fingerprint = self.noiseprint_encoder(x)       # (B, 1, H, W)
            c0 = self.branch_c_block1(fingerprint)             # (B, 48, H, W)
            c1 = self.branch_c_block2(self.branch_c_pool1(c0))  # (B, 96, H/2, W/2)

        # ── Fuse branches ─────────────────────────────────────────────────
        if self.use_noiseprint:
            skip0 = self.skip_fuse_0(torch.cat([a0, b0, c0], dim=1))
            fused = self.skip_fuse_1(torch.cat([a1, b1, c1], dim=1))
        else:
            skip0 = self.skip_fuse_0(torch.cat([a0, b0], dim=1))
            fused = self.skip_fuse_1(torch.cat([a1, b1], dim=1))

        # ── Encoder ───────────────────────────────────────────────────────
        e1 = self.enc1(fused)    # (B, 384, H/4)
        e2 = self.enc2(e1)      # (B, 480, H/8)
        e3 = self.enc3(e2)      # (B, 768, H/16)
        e4 = self.enc4(e3)      # (B, 1008, H/32) — bottleneck

        # ── Image-level trust ─────────────────────────────────────────────
        trust_logit = self.trust_head(e4)  # (B, 1)

        # ── Decoder ───────────────────────────────────────────────────────
        d4 = self.dec4(e4, e3)   # (B, 768, H/16)
        d3 = self.dec3(d4, e2)   # (B, 480, H/8)
        d2 = self.dec2(d3, e1)   # (B, 384, H/4)
        d1 = self.dec1(d2, fused)  # (B, 192, H/2)
        d0 = self.dec0(d1, skip0)  # (B, 192, H)

        # ── Pixel heads ───────────────────────────────────────────────────
        pixel_logits = self.pixel_head(d0)
        confidence_logits = self.confidence_head(d0)
        boundary_logits = self.boundary_head(d0)

        # ── Deep supervision (training only) ──────────────────────────────
        aux = []
        if self.training:
            aux.append(self.aux_head_h16(d4))  # @ H/16
            aux.append(self.aux_head_h8(d3))   # @ H/8
            aux.append(self.aux_head_h4(d2))   # @ H/4

        result = {
            "pixel": pixel_logits,
            "confidence": confidence_logits,
            "boundary": boundary_logits,
            "trust": trust_logit,
            "aux": aux,
        }
        if fingerprint is not None:
            result["fingerprint"] = fingerprint

        return result

    # ── Convenience: extract features for SG-DFT fusion ──────────────────

    def extract_encoder_features(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run forward and return intermediate encoder features for fusion.

        Used by the SG-DFT specialist backbone to extract features at
        H/8 and H/16 for cross-attention injection.
        """
        # Run branches
        a0 = self.branch_a_block1(x)
        a1 = self.branch_a_block2(self.branch_a_pool1(a0))
        bayar_out = self.multi_bayar(x)
        srm_out = self.srm(x)
        noise_feat = torch.cat([bayar_out, srm_out], dim=1)
        b0 = self.branch_b_block1(noise_feat)
        b1 = self.branch_b_block2(self.branch_b_pool1(b0))

        if self.use_noiseprint:
            with torch.no_grad():
                fp = self.noiseprint_encoder(x)
            c0 = self.branch_c_block1(fp)
            c1 = self.branch_c_block2(self.branch_c_pool1(c0))
            skip0 = self.skip_fuse_0(torch.cat([a0, b0, c0], dim=1))
            fused = self.skip_fuse_1(torch.cat([a1, b1, c1], dim=1))
        else:
            skip0 = self.skip_fuse_0(torch.cat([a0, b0], dim=1))
            fused = self.skip_fuse_1(torch.cat([a1, b1], dim=1))

        e1 = self.enc1(fused)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        return {
            "h4": e1,   # 384ch @ H/4
            "h8": e2,   # 480ch @ H/8   ← injected into SG-DFT at H/8
            "h16": e3,  # 768ch @ H/16  ← injected into SG-DFT at H/16
            "h32": e4,  # 1008ch @ H/32 (bottleneck)
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_nli_model_v2(
    use_noiseprint: bool = False,
    base_channels: int = 48,
    bayar_channels_per_scale: int = 10,
    srm_channels: int = 15,
) -> NPNLIUNet:
    """Build and return the NP-NLI v2 model."""
    return NPNLIUNet(
        use_noiseprint=use_noiseprint,
        base_channels=base_channels,
        bayar_channels_per_scale=bayar_channels_per_scale,
        srm_channels=srm_channels,
    )
