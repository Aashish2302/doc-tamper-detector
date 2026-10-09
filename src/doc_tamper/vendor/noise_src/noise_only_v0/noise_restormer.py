"""Restormer-based noise predictor for Stage-1 pre-training.

Restormer reference: Zamir et al., "Restormer: Efficient Transformer for
High-Resolution Image Restoration", CVPR 2022.  Architecture re-implemented in
pure PyTorch (no einops dependency — rearrange operations are expressed with
native ``.reshape``/``.permute``).  The official reference implementation is
MIT-licensed.

Forward: rgb (B,3,H,W) → {"noise": pred_noise (B,3,H,W), "pred_clean": pred_clean (B,3,H,W)}

The "noise" key is mandatory: the Stage-1 trainer reads ``out["noise"]``.
``pred_clean = restormer(rgb)`` (clamped to [0,1]) and ``pred_noise = rgb - pred_clean``.
"""
from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# LayerNorm (operates on (B,C,H,W), normalises over the channel dimension)
# ---------------------------------------------------------------------------


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    # (B, C, H, W) → (B, H*W, C)
    b, c, h, w = x.shape
    return x.reshape(b, c, h * w).permute(0, 2, 1)


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # (B, H*W, C) → (B, C, H, W)
    b, n, c = x.shape
    return x.permute(0, 2, 1).reshape(b, c, h, w)


class BiasFree_LayerNorm(nn.Module):
    """LayerNorm without mean subtraction or bias (RMS-style normalisation)."""

    def __init__(self, normalized_shape) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """Standard LayerNorm with learnable scale and bias."""

    def __init__(self, normalized_shape) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Channel-wise LayerNorm dispatcher for (B,C,H,W) tensors."""

    def __init__(self, dim: int, LayerNorm_type: str) -> None:
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


# ---------------------------------------------------------------------------
# Gated-Dconv Feed-Forward Network (GDFN)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """GDFN: 1x1 expand → 3x3 depthwise → gated GELU → 1x1 project."""

    def __init__(self, dim: int, ffn_expansion_factor: float,
                 bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim,
                                    hidden_features * 2,
                                    kernel_size=1,
                                    bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features,
                                     dim,
                                     kernel_size=1,
                                     bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ---------------------------------------------------------------------------
# Multi-Dconv Head Transposed Attention (MDTA)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """MDTA: channel-wise (transposed) self-attention with depthwise qkv."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, heads, C/heads, H*W) — native equivalent of einops
        # 'b (head c) h w -> b head c (h w)'.
        head = self.num_heads
        q = q.reshape(b, head, c // head, h * w)
        k = k.reshape(b, head, c // head, h * w)
        v = v.reshape(b, head, c // head, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, heads, C/heads, H*W)

        # 'b head c (h w) -> b (head c) h w'
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Restormer Transformer block: MDTA + GDFN, each with pre-LayerNorm."""

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float,
                 bias: bool, LayerNorm_type: str) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embed / down / up sampling
# ---------------------------------------------------------------------------


class OverlapPatchEmbed(nn.Module):
    """3x3 conv that lifts the input image into the feature space."""

    def __init__(self, in_c: int = 3, embed_dim: int = 48,
                 bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_c,
                              embed_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Halve spatial resolution, halve channel count (via PixelUnshuffle)."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat,
                      n_feat // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Double spatial resolution, halve channel count (via PixelShuffle)."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat,
                      n_feat * 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Restormer main network
# ---------------------------------------------------------------------------


class Restormer(nn.Module):
    """4-level encoder-decoder Transformer with a refinement stage.

    forward(inp_img) → restored CLEAN image (B, out_channels, H, W).

    Input H, W are reflect-padded to the next multiple of 8 (three 2x
    PixelUnshuffle downsamples) and cropped back after the global residual add.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 36,
        num_blocks=None,
        num_refinement_blocks: int = 4,
        heads=None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        dual_pixel_task: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]
        # Gradient checkpointing on the TransformerBlock stages — trades compute
        # for a large activation-memory reduction (needed at 512x512 when the
        # noise branch is trainable). Does not change params/state_dict, so
        # Stage-1 checkpoints load unchanged.
        self.use_checkpoint = use_checkpoint

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)  # dim -> dim*2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2**1))  # dim*2 -> dim*4
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2**2))  # dim*4 -> dim*8
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(int(dim * 2**3))  # dim*8 -> dim*4
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3),
                                            int(dim * 2**2),
                                            kernel_size=1,
                                            bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2**2))  # dim*4 -> dim*2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2),
                                            int(dim * 2**1),
                                            kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2**1))  # dim*2 -> dim
        # No channel reduction at level 1: concat -> dim*2 kept.
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim,
                                       int(dim * 2**1),
                                       kernel_size=1,
                                       bias=bias)

        self.output = nn.Conv2d(int(dim * 2**1),
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=bias)

    # ------------------------------------------------------------------
    # Reflect-pad H, W up to the next multiple of 8 (three 2x downsamples).
    # ------------------------------------------------------------------
    @staticmethod
    def _pad(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        factor = 8
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, h, w

    def _run_stage(self, stage: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a TransformerBlock stage, optionally under gradient checkpointing.

        Checkpointing only kicks in during training (grad enabled); at eval the
        stage runs normally. use_reentrant=False handles inputs that don't
        themselves require grad (the params do).
        """
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_padded, H, W = self._pad(inp_img)

        inp_enc_level1 = self.patch_embed(inp_padded)
        out_enc_level1 = self._run_stage(self.encoder_level1, inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self._run_stage(self.encoder_level2, inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self._run_stage(self.encoder_level3, inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self._run_stage(self.latent, inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self._run_stage(self.decoder_level3, inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self._run_stage(self.decoder_level2, inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self._run_stage(self.decoder_level1, inp_dec_level1)

        out_dec_level1 = self._run_stage(self.refinement, out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out = self.output(out_dec_level1)
        else:
            out = self.output(out_dec_level1) + inp_padded

        # Crop back to original (un-padded) spatial size.
        return out[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Noise-predictor wrapper (noise_only_v0 API)
# ---------------------------------------------------------------------------


class RestormerNoisePredictor(nn.Module):
    """Wraps Restormer as a noise predictor compatible with noise_only_v0.

    forward(rgb) → {
        "noise":      pred_noise  (B, 3, H, W) — estimated additive noise,
        "pred_clean": pred_clean  (B, 3, H, W) — denoised reconstruction,
    }

    pred_clean = restormer(rgb).clamp(0,1)  and  pred_noise = rgb - pred_clean.
    The "noise" key is mandatory (the trainer reads out["noise"]).
    """

    def __init__(
        self,
        dim: int = 36,
        num_blocks=None,
        num_refinement_blocks: int = 4,
        heads=None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        use_checkpoint: bool = False,
        max_infer_size: int = 0,
    ) -> None:
        super().__init__()
        # If >0, inputs whose longer side exceeds max_infer_size are downsampled
        # before the Restormer, then pred_clean is upsampled back. Bounds tensor
        # size/memory at full-resolution inference (training crops are < cap, so
        # training is unaffected). Needed because full-res docs overflow CUDA
        # 32-bit indexing in this heavy net.
        self.max_infer_size = int(max_infer_size)
        self.restormer = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=dim,
            num_blocks=num_blocks if num_blocks is not None else [4, 6, 6, 8],
            num_refinement_blocks=num_refinement_blocks,
            heads=heads if heads is not None else [1, 2, 4, 8],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            use_checkpoint=use_checkpoint,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=False,
        )

    def forward(self, rgb: torch.Tensor) -> dict:
        H, W = rgb.shape[-2:]
        cap = self.max_infer_size
        if cap and max(H, W) > cap:
            scale = cap / float(max(H, W))
            nh, nw = max(8, int(round(H * scale))), max(8, int(round(W * scale)))
            rgb_small = F.interpolate(rgb, size=(nh, nw), mode="bilinear",
                                      align_corners=False)
            clean_small = self.restormer(rgb_small).clamp(0.0, 1.0)
            pred_clean = F.interpolate(clean_small, size=(H, W),
                                       mode="bilinear", align_corners=False)
        else:
            pred_clean = self.restormer(rgb).clamp(0.0, 1.0)
        pred_noise = rgb - pred_clean
        return {"noise": pred_noise, "pred_clean": pred_clean}


def build_noise_restormer(
    dim: int = 36,
    num_blocks=None,
    num_refinement_blocks: int = 4,
    heads=None,
    ffn_expansion_factor: float = 2.66,
    bias: bool = False,
    LayerNorm_type: str = "WithBias",
    use_checkpoint: bool = False,
    max_infer_size: int = 0,
) -> RestormerNoisePredictor:
    """Factory for RestormerNoisePredictor.

    Args:
        dim:                   Base feature width (default 36).
        num_blocks:            Transformer blocks per level (default [4,6,6,8]).
        num_refinement_blocks: Refinement-stage blocks (default 4).
        heads:                 Attention heads per level (default [1,2,4,8]).
        ffn_expansion_factor:  GDFN hidden expansion (default 2.66).
        bias:                  Use bias in conv layers (default False).
        LayerNorm_type:        'WithBias' | 'BiasFree' (default 'WithBias').

    Returns:
        RestormerNoisePredictor instance (un-initialised weights).
    """
    return RestormerNoisePredictor(
        dim=dim,
        num_blocks=num_blocks if num_blocks is not None else [4, 6, 6, 8],
        num_refinement_blocks=num_refinement_blocks,
        heads=heads if heads is not None else [1, 2, 4, 8],
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=LayerNorm_type,
        use_checkpoint=use_checkpoint,
        max_infer_size=max_infer_size,
    )
