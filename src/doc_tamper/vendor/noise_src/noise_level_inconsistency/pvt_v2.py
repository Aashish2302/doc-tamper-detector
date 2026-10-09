"""PVT-v2 (Pyramid Vision Transformer v2) backbone for forensic detection.

Hierarchical vision transformer producing features at 4 scales:
  Stage 1:  C1 channels @ H/4
  Stage 2:  C2 channels @ H/8
  Stage 3:  C3 channels @ H/16
  Stage 4:  C4 channels @ H/32

Uses Spatial Reduction Attention (SRA) for efficiency — reduces K/V spatial
resolution before attention, keeping compute manageable at H/4 and H/8.

This is the shared backbone for all transformer-based specialists.

Architecture based on:
  Wang et al., "PVT v2: Improved Baselines with Pyramid Vision Transformer," CVMJ 2022.
  Used by M2SFormer (ICCV 2025 Highlight) for manipulation detection.

Sizes:
  PVT-v2-B0: [32, 64, 160, 256]   — 3.7M params (tiny)
  PVT-v2-B1: [64, 128, 320, 512]  — 14M params
  PVT-v2-B2: [64, 128, 320, 512]  — 25M params (default — deeper)
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """Depthwise convolution for position encoding in FFN."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    """FFN with depthwise conv for position encoding (PVT-v2 style)."""
    def __init__(self, in_features: int, hidden: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.dwconv = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialReductionAttention(nn.Module):
    """Spatial Reduction Attention (SRA) from PVT-v2.

    Reduces K/V spatial resolution by sr_ratio before attention.
    This makes attention at H/4 affordable (reduces from O(N²) to O(N²/sr²)).
    """

    def __init__(
        self, dim: int, num_heads: int = 8, sr_ratio: int = 1,
        attn_drop: float = 0.0, proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Attention in float32 to prevent FP16 overflow
        with torch.amp.autocast("cuda", enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn.to(q.dtype))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PVTBlock(nn.Module):
    """Single PVT-v2 transformer block: SRA + FFN with residuals."""

    def __init__(
        self, dim: int, num_heads: int, sr_ratio: int = 1,
        mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding: Conv with stride for downsampling."""

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 7, stride: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# PVT-v2 Backbone
# ---------------------------------------------------------------------------

class PVTv2(nn.Module):
    """Pyramid Vision Transformer v2 backbone.

    Produces hierarchical features at 4 scales: H/4, H/8, H/16, H/32.

    Parameters
    ----------
    in_channels : int
        Input channels (3 for RGB, or more for forensic inputs).
    embed_dims : list[int]
        Channel dims per stage. Default [64, 128, 320, 512] (B2).
    num_heads : list[int]
        Attention heads per stage. Default [1, 2, 5, 8].
    depths : list[int]
        Number of transformer blocks per stage. Default [3, 4, 6, 3] (B2).
    sr_ratios : list[int]
        Spatial reduction ratios per stage. Default [8, 4, 2, 1].
    mlp_ratio : float
        FFN expansion ratio. Default 4.0.
    drop_rate : float
        Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: list[int] | None = None,
        num_heads: list[int] | None = None,
        depths: list[int] | None = None,
        sr_ratios: list[int] | None = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 320, 512]
        if num_heads is None:
            num_heads = [1, 2, 5, 8]
        if depths is None:
            depths = [3, 4, 6, 3]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.embed_dims = embed_dims
        self.depths = depths
        self.num_stages = 4

        # Patch embeddings for each stage
        self.patch_embed1 = OverlapPatchEmbed(in_channels, embed_dims[0], patch_size=7, stride=4)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2)

        # Transformer blocks for each stage
        self.blocks1 = nn.ModuleList([
            PVTBlock(embed_dims[0], num_heads[0], sr_ratios[0], mlp_ratio, drop_rate)
            for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.blocks2 = nn.ModuleList([
            PVTBlock(embed_dims[1], num_heads[1], sr_ratios[1], mlp_ratio, drop_rate)
            for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.blocks3 = nn.ModuleList([
            PVTBlock(embed_dims[2], num_heads[2], sr_ratios[2], mlp_ratio, drop_rate)
            for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.blocks4 = nn.ModuleList([
            PVTBlock(embed_dims[3], num_heads[3], sr_ratios[3], mlp_ratio, drop_rate)
            for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning features at 4 scales.

        Args:
            x: (B, C, H, W) input image

        Returns:
            list of 4 tensors: [(B, C1, H/4, W/4), (B, C2, H/8, W/8),
                                (B, C3, H/16, W/16), (B, C4, H/32, W/32)]
        """
        features = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, H, W)
        x = self.norm1(x)
        features.append(x.reshape(-1, H, W, self.embed_dims[0]).permute(0, 3, 1, 2))

        # Stage 2
        x, H, W = self.patch_embed2(features[-1])
        for blk in self.blocks2:
            x = blk(x, H, W)
        x = self.norm2(x)
        features.append(x.reshape(-1, H, W, self.embed_dims[1]).permute(0, 3, 1, 2))

        # Stage 3
        x, H, W = self.patch_embed3(features[-1])
        for blk in self.blocks3:
            x = blk(x, H, W)
        x = self.norm3(x)
        features.append(x.reshape(-1, H, W, self.embed_dims[2]).permute(0, 3, 1, 2))

        # Stage 4
        x, H, W = self.patch_embed4(features[-1])
        for blk in self.blocks4:
            x = blk(x, H, W)
        x = self.norm4(x)
        features.append(x.reshape(-1, H, W, self.embed_dims[3]).permute(0, 3, 1, 2))

        return features


def build_pvt_v2_b2(in_channels: int = 3) -> PVTv2:
    """Build PVT-v2-B2 (default, ~25M params)."""
    return PVTv2(
        in_channels=in_channels,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
    )


def build_pvt_v2_b1(in_channels: int = 3) -> PVTv2:
    """Build PVT-v2-B1 (smaller, ~14M params)."""
    return PVTv2(
        in_channels=in_channels,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
    )


def build_pvt_v2_b0(in_channels: int = 3) -> PVTv2:
    """Build PVT-v2-B0 (tiny, ~3.7M params)."""
    return PVTv2(
        in_channels=in_channels,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
    )
