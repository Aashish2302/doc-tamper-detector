"""NAFNet-based noise predictor for Stage-1 pre-training.

NAFNet reference: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
Architecture: compact NAFNet encoder-decoder predicting additive noise.
Forward: rgb (B,3,H,W) → {"noise": pred_noise (B,3,H,W), "pred_clean": pred_clean (B,3,H,W)}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Per-channel layer normalisation over spatial dimensions (H, W)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → permute to (B, H, W, C) → norm → permute back
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Gated activation: split channels in half and multiply elementwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet basic block.

    Two sub-paths:
      1. Spatial mixing (depthwise conv + SimpleGate + channel attention)
      2. Channel mixing (pointwise FFN with SimpleGate)
    Each sub-path is preceded by LayerNorm2d and its output is scaled by a
    learnable parameter (beta / gamma) before being added to the residual.
    """

    def __init__(self,
                 c: int,
                 dw_expand: int = 2,
                 ffn_expand: int = 2) -> None:
        super().__init__()
        dw_ch = c * dw_expand
        ffn_ch = c * ffn_expand

        # --- Spatial mixing path ---
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)  # expand
        self.conv2 = nn.Conv2d(dw_ch,
                               dw_ch,
                               3,
                               padding=1,
                               groups=dw_ch,
                               bias=True)  # depthwise
        self.sg1 = SimpleGate()  # dw_ch → c
        # Simple Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=False),
        )
        self.conv3 = nn.Conv2d(c, c, 1)  # project back
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

        # --- Channel mixing path (FFN) ---
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)  # expand
        self.sg2 = SimpleGate()  # ffn_ch → c
        self.conv5 = nn.Conv2d(c, c, 1)  # project back
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # --- Spatial mixing ---
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)  # (B, c, H, W)
        x = x * self.sca(x)  # channel attention
        x = self.conv3(x)
        y = inp + self.beta * x  # residual

        # --- Channel mixing (FFN) ---
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)  # (B, c, H, W)
        x = self.conv5(x)
        return y + self.gamma * x


class NAFNet(nn.Module):
    """NAFNet encoder-decoder for image restoration.

    Args:
        img_channel:    Number of input/output channels (3 for RGB).
        width:          Base feature width (number of channels after intro conv).
        middle_blk_num: Number of NAFBlocks at the bottleneck.
        enc_blks:       List giving the number of NAFBlocks per encoder stage.
        dec_blks:       List giving the number of NAFBlocks per decoder stage.
                        len(enc_blks) must equal len(dec_blks).

    forward(inp) → restored image in [0, 1] (same spatial size as inp).
    """

    def __init__(self,
                 img_channel: int = 3,
                 width: int = 32,
                 middle_blk_num: int = 8,
                 enc_blks: list[int] | None = None,
                 dec_blks: list[int] | None = None) -> None:
        super().__init__()
        if enc_blks is None:
            enc_blks = [2, 2, 4]
        if dec_blks is None:
            dec_blks = [2, 2, 2]

        assert len(enc_blks) == len(dec_blks), (
            "enc_blks and dec_blks must have the same length")

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        self.encoders: nn.ModuleList = nn.ModuleList()
        self.downs: nn.ModuleList = nn.ModuleList()
        self.middle_blks = nn.Sequential(*[
            NAFBlock(width * (2**len(enc_blks))) for _ in range(middle_blk_num)
        ])
        self.ups: nn.ModuleList = nn.ModuleList()
        self.decoders: nn.ModuleList = nn.ModuleList()

        chan = width
        for num in enc_blks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        # At bottleneck: chan = width * 2^len(enc_blks)

        for num in dec_blks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                ))
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self._n_stages = len(enc_blks)

    # ------------------------------------------------------------------
    # Padding helper: ensures H and W are multiples of 2^n_stages so the
    # stride-2 downsampling steps never see odd sizes.
    # ------------------------------------------------------------------
    def _pad(
            self,
            x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        _, _, H, W = x.shape
        factor = 2**self._n_stages
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        # reflect padding: (left, right, top, bottom) for F.pad
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x_padded, (0, pad_w, 0, pad_h)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, H, W = inp.shape
        inp_padded, _ = self._pad(inp)

        x = self.intro(inp_padded)

        # Encode
        enc_skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle_blks(x)

        # Decode
        for decoder, up, skip in zip(self.decoders, self.ups,
                                     reversed(enc_skips)):
            x = up(x)
            x = x + skip  # element-wise add (not concat)
            x = decoder(x)

        x = self.ending(x)
        out = (inp_padded + x).clamp(0.0, 1.0)

        # Crop back to original size
        return out[:, :, :H, :W]


class NAFNetNoisePredictor(nn.Module):
    """Wraps NAFNet as a noise predictor compatible with the noise_only_v0 API.

    forward(rgb) → {
        "noise":      pred_noise  (B, 3, H, W) — estimated additive noise,
        "pred_clean": pred_clean  (B, 3, H, W) — denoised reconstruction,
    }

    The "noise" key is present so existing code using out["noise"] works
    unchanged.  pred_clean = nafnet(rgb)  and  pred_noise = rgb - pred_clean.
    """

    def __init__(self,
                 width: int = 32,
                 middle_blk_num: int = 8,
                 enc_blks: list[int] | None = None,
                 dec_blks: list[int] | None = None) -> None:
        super().__init__()
        self.nafnet = NAFNet(
            img_channel=3,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blks=enc_blks if enc_blks is not None else [2, 2, 4],
            dec_blks=dec_blks if dec_blks is not None else [2, 2, 2],
        )

    def forward(self, rgb: torch.Tensor) -> dict:
        pred_clean = self.nafnet(rgb)
        pred_noise = rgb - pred_clean
        return {"noise": pred_noise, "pred_clean": pred_clean}


def build_noise_nafnet(
    width: int = 32,
    middle_blk_num: int = 8,
    enc_blks: list[int] | None = None,
    dec_blks: list[int] | None = None,
) -> NAFNetNoisePredictor:
    """Factory function for NAFNetNoisePredictor.

    Args:
        width:          Base feature width (default 32).
        middle_blk_num: NAFBlocks at bottleneck (default 8).
        enc_blks:       NAFBlocks per encoder stage (default [2, 2, 4]).
        dec_blks:       NAFBlocks per decoder stage (default [2, 2, 2]).

    Returns:
        NAFNetNoisePredictor instance (un-initialised weights).
    """
    return NAFNetNoisePredictor(
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blks=enc_blks if enc_blks is not None else [2, 2, 4],
        dec_blks=dec_blks if dec_blks is not None else [2, 2, 2],
    )
