"""Stage-2 model for noise_V0: tamper detector operating on noise tensor.

Input:  predicted noise tensor from frozen Stage-1, shape (B, 3, H, W).
Output: dict with
  - pixel: (B, 1, H, W)  — tamper mask logits
  - bbox:  (B, 4)        — (cx, cy, w, h) in [0,1]
  - trust: (B, 1)        — image-level tampered logit

Architecture: PVT-V2-B0 encoder on the noise tensor → FPN decoder → 3 heads.
B0 (~3.4M params) is small because the noise tensor is sparser than RGB —
no need for B2 capacity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NLI_SRC = (Path(__file__).resolve().parents[3] /
           "noise_level_inconsistency/src")
if str(NLI_SRC) not in sys.path:
    sys.path.insert(0, str(NLI_SRC))
from noise_level_inconsistency.pvt_v2 import build_pvt_v2_b0

from noise_only_v0.learned_model_noise_v0 import FPNDecoder, _ConvBNReLU


class NoiseV0Stage2(nn.Module):
    """Tamper detector taking only the noise tensor as input."""

    def __init__(self, decoder_out_ch: int = 64) -> None:
        super().__init__()
        self.encoder = build_pvt_v2_b0(in_channels=3)
        self.embed_dims = self.encoder.embed_dims  # [32, 64, 160, 256] for B0
        self.decoder = FPNDecoder(self.embed_dims, decoder_out_ch)

        # Pixel mask head
        self.pixel_head = nn.Sequential(
            _ConvBNReLU(decoder_out_ch, decoder_out_ch),
            nn.Conv2d(decoder_out_ch, 1, 1),
        )

        # Bbox + trust pool from the deepest feature
        c4 = self.embed_dims[-1]
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # cx, cy, w, h ∈ [0,1]
        )
        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, noise_tensor: torch.Tensor) -> dict:
        H, W = noise_tensor.shape[-2:]
        feats = self.encoder(noise_tensor)
        feat = self.decoder(feats)
        feat = F.interpolate(feat,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)
        pixel_logits = self.pixel_head(feat)
        bbox = self.bbox_head(feats[-1])
        trust = self.trust_head(feats[-1])
        return {
            "pixel": pixel_logits,
            "bbox": bbox,
            "trust": trust,
        }


def build_noise_v0_stage2(decoder_out_ch: int = 64) -> NoiseV0Stage2:
    return NoiseV0Stage2(decoder_out_ch=decoder_out_ch)
