"""CMX fusion modules: FRM (Feature Rectify Module) + FFM (Feature Fusion Module).

Adapted from CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers
(Zhang et al., TITS 2023). Modality-agnostic — works for any two feature streams of
matching spatial resolution and channel count.

FeatureRectifyModule (FRM):
    Channel branch: squeeze-and-excitation style cross-modal channel recalibration.
    Spatial branch: cross-modal spatial attention map applied to the other modality.
    Output: rectified (F_a, F_b) at the same spatial/channel dims as input.

FeatureFusionModule (FFM):
    Cross-attention between the two rectified feature maps.
    Q from one modality, K/V from the other, then residual merge.
    Output: single fused feature map at the same spatial/channel dims as input.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_pair(x):
    return (x, x) if isinstance(x, int) else x


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, relu=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


# ── FRM ──────────────────────────────────────────────────────────────────────


class FeatureRectifyModule(nn.Module):
    """Rectify each modality's features using cross-modal attention from the other.

    For modalities A and B with C channels each:
      channel_att_a = sigmoid(fc(gap(F_b)))    # B guides A
      channel_att_b = sigmoid(fc(gap(F_a)))    # A guides B
      spatial_att_a = sigmoid(conv([avg(F_b), max(F_b)]))
      spatial_att_b = sigmoid(conv([avg(F_a), max(F_a)]))
      F_a' = F_a * channel_att_a * spatial_att_a
      F_b' = F_b * channel_att_b * spatial_att_b
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 16)

        # Channel recalibration (one modality → other)
        self.ch_fc_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        self.ch_fc_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

        # Spatial recalibration
        self.sp_conv_a = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.sp_conv_b = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, fa: torch.Tensor, fb: torch.Tensor):
        # Channel: B guides A, A guides B
        ch_a = self.ch_fc_a(fb).view(fb.size(0), -1, 1, 1)
        ch_b = self.ch_fc_b(fa).view(fa.size(0), -1, 1, 1)

        # Spatial: B guides A, A guides B
        sp_a = self.sp_conv_a(
            torch.cat([fb.mean(1, keepdim=True), fb.max(1, keepdim=True)[0]], dim=1))
        sp_b = self.sp_conv_b(
            torch.cat([fa.mean(1, keepdim=True), fa.max(1, keepdim=True)[0]], dim=1))

        fa_out = fa * ch_a * sp_a
        fb_out = fb * ch_b * sp_b
        return fa_out, fb_out


# ── FFM ──────────────────────────────────────────────────────────────────────


class FeatureFusionModule(nn.Module):
    """Cross-attention fusion of two rectified feature maps.

    Projects F_a → Q and F_b → K,V (and vice-versa), computes scaled dot-product
    attention, then merges with residual connection.  Output has the same spatial
    resolution and channel count as the inputs.
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(channels // num_heads, 1)
        self.scale = self.head_dim ** -0.5

        # A→Q, B→KV (for A attending to B)
        self.q_a = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_b = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_b = nn.Conv2d(channels, channels, 1, bias=False)

        # B→Q, A→KV (for B attending to A)
        self.q_b = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_a = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_a = nn.Conv2d(channels, channels, 1, bias=False)

        self.proj_a = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_b = nn.Conv2d(channels, channels, 1, bias=False)

        self.norm_a = nn.BatchNorm2d(channels)
        self.norm_b = nn.BatchNorm2d(channels)

        self.ffn = nn.Sequential(
            ConvBnRelu(channels * 2, channels, k=1, p=0),
            nn.Conv2d(channels, channels, 1),
        )
        self.norm_out = nn.BatchNorm2d(channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)

    def _cross_attn(self, fq, fkv_k, fkv_v, q_proj, k_proj, v_proj, out_proj, norm):
        B, C, H, W = fq.shape
        N = H * W

        q = q_proj(fq).flatten(2).transpose(1, 2)   # B,N,C
        k = k_proj(fkv_k).flatten(2).transpose(1, 2)
        v = v_proj(fkv_v).flatten(2).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return norm(fq + out_proj(out))

    def forward(self, fa: torch.Tensor, fb: torch.Tensor):
        fa_r = self._cross_attn(fa, fb, fb, self.q_a, self.k_b, self.v_b,
                                self.proj_a, self.norm_a)
        fb_r = self._cross_attn(fb, fa, fa, self.q_b, self.k_a, self.v_a,
                                self.proj_b, self.norm_b)
        fused = self.norm_out(fa_r + self.ffn(torch.cat([fa_r, fb_r], dim=1)))
        return fused
