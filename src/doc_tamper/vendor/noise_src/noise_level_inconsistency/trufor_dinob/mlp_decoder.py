"""SegFormer-style all-MLP decoder head.

Re-implementation of the all-MLP decoder head from:
  SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
  (Xie et al., NeurIPS 2021).

Accepts 4 feature scales [C0..C3] at strides [/4, /8, /16, /32] and produces a
single output logit map at stride /4 (H/4 × W/4).  Backbone-agnostic — works with
any encoder that emits the expected 4-scale list.

Missing/unexpected keys from partial checkpoint loading are handled gracefully.
Pure PyTorch — no TruFor / non-commercial imports.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.fc(x)))


class DecoderHead(nn.Module):
    """All-MLP segmentation head.

    Args:
        in_channels: list of 4 channel counts from encoder [C0, C1, C2, C3]
                     corresponding to strides [/4, /8, /16, /32].
        embed_dim:   unified embedding dimension (all scales projected to this).
        num_classes: number of output classes (2 for binary tamper detection).
        dropout:     dropout rate on the fused representation.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(in_channels) == 4

        # Project each scale to embed_dim
        self.linears = nn.ModuleList([
            MlpBlock(c, embed_dim) for c in in_channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [F0, F1, F2, F3] tensors at strides [/4, /8, /16, /32].
                      Each F_i is (B, C_i, H_i, W_i).
        Returns:
            logits: (B, num_classes, H/4, W/4)
        """
        H, W = features[0].shape[2], features[0].shape[3]
        outs = []
        for i, (feat, lin) in enumerate(zip(features, self.linears)):
            B, C, h, w = feat.shape
            # (B, C, h, w) → (B, h*w, C) → MLP → (B, h*w, D) → (B, D, h, w)
            x = feat.flatten(2).transpose(1, 2)
            x = lin(x).transpose(1, 2).reshape(B, -1, h, w)
            if i > 0:
                x = F.interpolate(x, size=(H, W), mode="bilinear",
                                  align_corners=False)
            outs.append(x)

        fused = self.fuse(torch.cat(outs, dim=1))
        return self.head(fused)
