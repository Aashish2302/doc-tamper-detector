"""Learned dual-branch NLI tamper localizer.

Architecture
------------
Branch A (RGB context):
    RGB (3ch) -> 2x DoubleConv (3->32->64) with GroupNorm(groups=8)

Branch B (Forensic noise, dual front-end):
    RGB (3ch) -> BayarConv2d (3->15ch, constrained high-pass)
                 SRM filter bank (3->15ch, fixed 30 SRM kernels -> 1x1 conv)
              -> Concat 15+15=30ch
              -> 2x DoubleConv (30->32->64) with GroupNorm(groups=8)

Fusion: concat A+B = 128ch, per-level skip fusion via 1x1 conv

Encoder: 128->256->512->768  (4 levels)

Decoder: 768->512->256->128  (with fused skip connections)

Heads:
    - Pixel tamper map:  128->1 (sigmoid)
    - Pixel confidence:  128->1 (sigmoid)
    - Image-level trust: GAP(bottleneck 768) -> 256 -> 1 (sigmoid)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BayarConv2d -- learned constrained high-pass convolution
# ---------------------------------------------------------------------------

class BayarConv2d(nn.Module):
    """Constrained convolutional layer whose kernels act as high-pass filters.

    The centre weight is forced to ``-sum(rest)`` so each kernel integrates
    to zero, guaranteeing high-pass behaviour while allowing the non-centre
    weights to be learned freely.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 15,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.kernel_size = kernel_size
        self.center = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clone to avoid in-place on the parameter
        w = self.weight.clone()
        # Zero out centre, sum the rest, then set centre = -rest_sum
        w[:, :, self.center, self.center] = 0
        rest_sum = w.sum(dim=(2, 3))  # (out_channels, in_channels)
        w[:, :, self.center, self.center] = -rest_sum
        return F.conv2d(x, w, self.bias, padding=self.center)


# ---------------------------------------------------------------------------
# SRM filter bank -- fixed (non-learnable) spatial rich model kernels
# ---------------------------------------------------------------------------

def _build_srm_kernels_30() -> np.ndarray:
    """Build 30 SRM kernels (5x5) as used in steganalysis literature.

    Returns shape (30, 5, 5) float32.

    We construct 30 kernels from 6 base patterns x 5 rotations/reflections.
    All kernels sum to zero (high-pass property).
    """
    kernels: list[np.ndarray] = []

    # ---- Group 1: 1st-order horizontal/vertical differences ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 1] = -1.0; k[2, 3] = 1.0
    kernels.append(k)  # horizontal

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 2] = -1.0; k[3, 2] = 1.0
    kernels.append(k)  # vertical

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = -1.0; k[3, 3] = 1.0
    kernels.append(k)  # diagonal-1

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 3] = -1.0; k[3, 1] = 1.0
    kernels.append(k)  # diagonal-2

    # ---- Group 2: 2nd-order differences ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 1] = 1.0; k[2, 2] = -2.0; k[2, 3] = 1.0
    kernels.append(k)  # horizontal 2nd order

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 2] = 1.0; k[2, 2] = -2.0; k[3, 2] = 1.0
    kernels.append(k)  # vertical 2nd order

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = 1.0; k[2, 2] = -2.0; k[3, 3] = 1.0
    kernels.append(k)  # diagonal 2nd order

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 3] = 1.0; k[2, 2] = -2.0; k[3, 1] = 1.0
    kernels.append(k)  # anti-diagonal 2nd order

    # ---- Group 3: 3rd-order differences ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 0] = -1.0; k[2, 1] = 3.0; k[2, 2] = -3.0; k[2, 3] = 1.0
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 2] = -1.0; k[1, 2] = 3.0; k[2, 2] = -3.0; k[3, 2] = 1.0
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 1] = 1.0; k[2, 2] = -3.0; k[2, 3] = 3.0; k[2, 4] = -1.0
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 2] = 1.0; k[2, 2] = -3.0; k[3, 2] = 3.0; k[4, 2] = -1.0
    kernels.append(k)

    # ---- Group 4: SQUARE 3x3 and 5x5 Laplacian (isotropic) ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = -1; k[1, 2] = -1; k[1, 3] = -1
    k[2, 1] = -1; k[2, 2] = 8;  k[2, 3] = -1
    k[3, 1] = -1; k[3, 2] = -1; k[3, 3] = -1
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 2] = 1; k[2, 1] = 1; k[2, 2] = -4; k[2, 3] = 1; k[3, 2] = 1
    kernels.append(k)  # 4-connected Laplacian

    # ---- Group 5: EDGE (Prewitt/Sobel-like patterns) ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = -1; k[1, 2] = 0; k[1, 3] = 1
    k[2, 1] = -1; k[2, 2] = 0; k[2, 3] = 1
    k[3, 1] = -1; k[3, 2] = 0; k[3, 3] = 1
    kernels.append(k)  # Prewitt horizontal

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = -1; k[1, 2] = -1; k[1, 3] = -1
    k[2, 1] = 0;  k[2, 2] = 0;  k[2, 3] = 0
    k[3, 1] = 1;  k[3, 2] = 1;  k[3, 3] = 1
    kernels.append(k)  # Prewitt vertical

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = -1; k[1, 2] = 0; k[1, 3] = 1
    k[2, 1] = -2; k[2, 2] = 0; k[2, 3] = 2
    k[3, 1] = -1; k[3, 2] = 0; k[3, 3] = 1
    kernels.append(k)  # Sobel horizontal

    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = -1; k[1, 2] = -2; k[1, 3] = -1
    k[2, 1] = 0;  k[2, 2] = 0;  k[2, 3] = 0
    k[3, 1] = 1;  k[3, 2] = 2;  k[3, 3] = 1
    kernels.append(k)  # Sobel vertical

    # ---- Group 6: extended difference patterns (long-range) ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 2] = 1; k[2, 2] = -2; k[4, 2] = 1
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 0] = 1; k[2, 2] = -2; k[2, 4] = 1
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 0] = 1; k[2, 2] = -2; k[4, 4] = 1
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 4] = 1; k[2, 2] = -2; k[4, 0] = 1
    kernels.append(k)

    # ---- Group 7: SPAM-like (averaging predictor residuals) ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 0] = -0.25; k[2, 1] = 0.5; k[2, 2] = 0; k[2, 3] = -0.5; k[2, 4] = 0.25
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 2] = -0.25; k[1, 2] = 0.5; k[2, 2] = 0; k[3, 2] = -0.5; k[4, 2] = 0.25
    kernels.append(k)

    # ---- Group 8: cross patterns ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 2] = -1; k[2, 0] = -1; k[2, 2] = 4; k[2, 4] = -1; k[4, 2] = -1
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 0] = -1; k[0, 4] = -1; k[2, 2] = 4; k[4, 0] = -1; k[4, 4] = -1
    kernels.append(k)

    # ---- Group 9: 2nd order mixed partial ----
    k = np.zeros((5, 5), dtype=np.float32)
    k[1, 1] = 1; k[1, 3] = -1; k[3, 1] = -1; k[3, 3] = 1
    kernels.append(k)

    k = np.zeros((5, 5), dtype=np.float32)
    k[0, 2] = 0.5; k[2, 0] = 0.5; k[2, 2] = -2; k[2, 4] = 0.5; k[4, 2] = 0.5
    kernels.append(k)

    # Pad to exactly 30 kernels if needed
    while len(kernels) < 30:
        # Generate additional rotated/mirrored versions of existing kernels
        idx = len(kernels) % len(kernels)
        k_rot = np.rot90(kernels[idx % 14]).copy()
        kernels.append(k_rot.astype(np.float32))

    kernels = kernels[:30]
    return np.stack(kernels, axis=0)  # (30, 5, 5)


class SRMFilterBank(nn.Module):
    """Fixed SRM filter bank: 30 SRM kernels applied per channel, projected to 15ch.

    The SRM kernels are NOT learnable. Only the 1x1 projection conv is learned.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 15) -> None:
        super().__init__()
        srm_np = _build_srm_kernels_30()  # (30, 5, 5)

        # Expand to (30, in_channels, 5, 5): each of the 30 filters applied
        # identically across all input channels (averaged)
        weight = np.zeros((30, in_channels, 5, 5), dtype=np.float32)
        for c in range(in_channels):
            weight[:, c, :, :] = srm_np / in_channels

        self.register_buffer(
            "srm_weight", torch.from_numpy(weight)
        )
        self.register_buffer(
            "srm_bias", torch.zeros(30)
        )

        # Learnable 1x1 projection from 30 -> out_channels
        self.project = nn.Conv2d(30, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed SRM convolution (padding=2 for 5x5 kernels)
        srm_out = F.conv2d(x, self.srm_weight, self.srm_bias, padding=2)
        return self.project(srm_out)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two 3x3 convolutions each followed by GroupNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample + concat skip + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, groups=groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Dual-branch NLI UNet
# ---------------------------------------------------------------------------

class NLIUNet(nn.Module):
    """Dual-branch UNet for learned NLI tamper localisation.

    Branch A processes RGB context; Branch B extracts forensic noise features
    via BayarConv2d (learned) and SRM filter bank (fixed). The two branches
    are fused and fed through a 4-level UNet encoder/decoder.

    Three output heads:
        1. Pixel tamper probability map (H x W)
        2. Pixel confidence map (H x W)
        3. Image-level trust scalar
    """

    def __init__(self) -> None:
        super().__init__()

        # ==== Branch A: RGB context ====
        self.branch_a_block1 = DoubleConv(3, 32, groups=8)   # -> 32ch
        self.branch_a_pool1 = nn.MaxPool2d(2)
        self.branch_a_block2 = DoubleConv(32, 64, groups=8)  # -> 64ch

        # ==== Branch B: Forensic noise ====
        self.bayar = BayarConv2d(in_channels=3, out_channels=15, kernel_size=5)
        self.srm = SRMFilterBank(in_channels=3, out_channels=15)
        # After concat: 15+15 = 30ch
        self.branch_b_block1 = DoubleConv(30, 32, groups=8)  # -> 32ch
        self.branch_b_pool1 = nn.MaxPool2d(2)
        self.branch_b_block2 = DoubleConv(32, 64, groups=8)  # -> 64ch

        # ==== Skip fusion: 1x1 convs to compress concatenated skips ====
        # Level 0 (full res): skip_a(32) + skip_b(32) = 64 -> compress to 64
        self.skip_fuse_0 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # Level 1 (half res): skip_a(64) + skip_b(64) = 128 -> compress to 128
        self.skip_fuse_1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        # ==== Encoder (on fused 64+64=128ch features) ====
        # Input: 128ch at 1/2 resolution
        self.enc1 = DownBlock(128, 256, groups=8)    # 1/4 res
        self.enc2 = DownBlock(256, 512, groups=8)     # 1/8 res
        self.enc3 = DownBlock(512, 768, groups=8)     # 1/16 res (bottleneck)

        # ==== Decoder ====
        self.dec3 = UpBlock(768, 512, 512, groups=8)  # 1/8 -> gets enc2 skip
        self.dec2 = UpBlock(512, 256, 256, groups=8)   # 1/4 -> gets enc1 skip
        self.dec1 = UpBlock(256, 128, 128, groups=8)   # 1/2 -> gets fused branch skip

        # Final upsample to full resolution using the level-0 fused skip
        self.dec0 = UpBlock(128, 64, 128, groups=8)    # full res -> gets level-0 skip

        # ==== Heads ====
        # Pixel tamper probability
        self.pixel_head = nn.Conv2d(128, 1, kernel_size=1)
        # Pixel confidence (how reliable is this pixel's prediction)
        self.confidence_head = nn.Conv2d(128, 1, kernel_size=1)
        # Image-level trust classifier
        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for conv/linear layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, 3, H, W) RGB input in [0, 1].

        Returns
        -------
        pixel_logits : (B, 1, H, W) -- raw logits for tamper probability
        confidence_logits : (B, 1, H, W) -- raw logits for pixel confidence
        trust_logit : (B, 1) -- raw logit for image-level tamper classification
        """
        # ---- Branch A: RGB context ----
        a0 = self.branch_a_block1(x)           # (B, 32, H, W)
        a1 = self.branch_a_block2(self.branch_a_pool1(a0))  # (B, 64, H/2, W/2)

        # ---- Branch B: Forensic noise ----
        bayar_out = self.bayar(x)               # (B, 15, H, W)
        srm_out = self.srm(x)                   # (B, 15, H, W)
        noise_feat = torch.cat([bayar_out, srm_out], dim=1)  # (B, 30, H, W)
        b0 = self.branch_b_block1(noise_feat)   # (B, 32, H, W)
        b1 = self.branch_b_block2(self.branch_b_pool1(b0))  # (B, 64, H/2, W/2)

        # ---- Fuse branches ----
        # Level 0 skip (full res)
        skip0 = self.skip_fuse_0(torch.cat([a0, b0], dim=1))  # (B, 64, H, W)
        # Level 1 features (half res) -- main feature path
        fused = torch.cat([a1, b1], dim=1)  # (B, 128, H/2, W/2)
        # Level 1 skip
        skip1 = self.skip_fuse_1(fused)  # (B, 128, H/2, W/2)

        # ---- Encoder ----
        e1 = self.enc1(fused)   # (B, 256, H/4, W/4)
        e2 = self.enc2(e1)      # (B, 512, H/8, W/8)
        e3 = self.enc3(e2)      # (B, 768, H/16, W/16)  -- bottleneck

        # ---- Image-level trust from bottleneck ----
        trust_logit = self.trust_head(e3)  # (B, 1)

        # ---- Decoder ----
        d3 = self.dec3(e3, e2)   # (B, 512, H/8, W/8)
        d2 = self.dec2(d3, e1)   # (B, 256, H/4, W/4)
        d1 = self.dec1(d2, skip1)  # (B, 128, H/2, W/2)
        d0 = self.dec0(d1, skip0)  # (B, 128, H, W)

        # ---- Pixel heads ----
        pixel_logits = self.pixel_head(d0)          # (B, 1, H, W)
        confidence_logits = self.confidence_head(d0)  # (B, 1, H, W)

        return pixel_logits, confidence_logits, trust_logit


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_nli_model() -> NLIUNet:
    """Build and return the dual-branch NLI UNet model."""
    return NLIUNet()
