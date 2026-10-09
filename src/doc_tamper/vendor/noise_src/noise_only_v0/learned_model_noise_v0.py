"""Stage-1 model for noise_V0: pure denoiser predicting per-pixel noise.

Input:  noisy RGB image, shape (B, 3, H, W), values in [0, 1]
Output: predicted noise tensor, shape (B, 3, H, W), values in roughly [-1, 1]
        such that  noisy - pred ≈ clean.

Architecture: PVT-V2-B2 encoder → light FPN decoder → 3ch conv head.
No BayarConv/SRM/noise stems — keep the noise hypothesis clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the existing PVT builder from the NLI module without coupling to its
# heads.
NLI_SRC = (Path(__file__).resolve().parents[3] /
           "noise_level_inconsistency/src")
if str(NLI_SRC) not in sys.path:
    sys.path.insert(0, str(NLI_SRC))
from noise_level_inconsistency.pvt_v2 import (build_pvt_v2_b0, build_pvt_v2_b1,
                                              build_pvt_v2_b2)


class _ConvBNReLU(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FPNDecoder(nn.Module):
    """Lightweight FPN: 4 PVT stages → top-down lateral fusion → C×H/4×W/4."""

    def __init__(self, embed_dims: list[int], out_ch: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4 = embed_dims  # [64, 128, 320, 512] for B2
        self.lat4 = nn.Conv2d(c4, out_ch, 1)
        self.lat3 = nn.Conv2d(c3, out_ch, 1)
        self.lat2 = nn.Conv2d(c2, out_ch, 1)
        self.lat1 = nn.Conv2d(c1, out_ch, 1)
        self.smooth4 = _ConvBNReLU(out_ch, out_ch)
        self.smooth3 = _ConvBNReLU(out_ch, out_ch)
        self.smooth2 = _ConvBNReLU(out_ch, out_ch)
        self.smooth1 = _ConvBNReLU(out_ch, out_ch)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = feats
        p4 = self.smooth4(self.lat4(f4))
        p3 = self.smooth3(
            self.lat3(f3) + F.interpolate(
                p4, size=f3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(
            self.lat2(f2) + F.interpolate(
                p3, size=f2.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.smooth1(
            self.lat1(f1) + F.interpolate(
                p2, size=f1.shape[-2:], mode="bilinear", align_corners=False))
        return p1  # (B, out_ch, H/4, W/4)


class NoiseV0Stage1(nn.Module):
    """Pure noise predictor.

    forward(rgb) → dict with 'noise' (B, 3, H, W) — additive noise estimate.
    """

    def __init__(self,
                 backbone_size: str = "b2",
                 decoder_out_ch: int = 64,
                 noise_scale: float = 0.5) -> None:
        super().__init__()
        builders = {
            "b0": build_pvt_v2_b0,
            "b1": build_pvt_v2_b1,
            "b2": build_pvt_v2_b2,
        }
        self.encoder = builders[backbone_size](in_channels=3)
        self.embed_dims = self.encoder.embed_dims
        self.decoder = FPNDecoder(self.embed_dims, decoder_out_ch)
        # 3-channel noise head; tanh keeps output bounded so the denoised
        # image stays in [0, 1] after the stage-2 sees a sane signal.
        self.head = nn.Sequential(
            _ConvBNReLU(decoder_out_ch, decoder_out_ch),
            nn.Conv2d(decoder_out_ch, 3, 1),
        )
        # noise_scale controls tanh range: 0.5 → [-0.5,0.5] (default, backward
        # compat), 1.0 → [-1,1] (Run E relaxed variant).
        self.noise_scale = noise_scale

    def forward(self, rgb: torch.Tensor) -> dict:
        H, W = rgb.shape[-2:]
        feats = self.encoder(rgb)
        feat = self.decoder(feats)
        feat = F.interpolate(feat,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)
        noise = self.head(feat)
        # Bound output: noise magnitude rarely exceeds ~0.3 in [0,1] images,
        # noise_scale*tanh keeps gradients sane while allowing full range when needed.
        noise = self.noise_scale * torch.tanh(noise)
        return {"noise": noise}


def build_noise_v0_stage1(backbone_size: str = "b2",
                          decoder_out_ch: int = 64,
                          noise_scale: float = 0.5) -> NoiseV0Stage1:
    return NoiseV0Stage1(backbone_size=backbone_size,
                         decoder_out_ch=decoder_out_ch,
                         noise_scale=noise_scale)
