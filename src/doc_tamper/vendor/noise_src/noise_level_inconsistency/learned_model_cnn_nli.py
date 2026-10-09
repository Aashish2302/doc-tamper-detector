"""CNN NP-NLI — standalone noise-level inconsistency model (Branch A + Branch B).

Scaled-down CNN-only architecture for probing whether a noise branch can converge
independently (no PVT backbone). Used in probe runs P1/P2.

Branch A (RGB context): 3ch → DoubleConv → 16ch → 32ch encoder
Branch B (Bayar+SRM noise): 3ch → 45ch → DoubleConv → 32ch → 64ch encoder
Combined: concat(A_feat, B_feat) → 96ch → 128ch → 256ch encoder
Decoder mirrors encoder with skip connections.
Output heads: pixel, confidence, boundary, trust, n_hp_pred (+ sigma, jpeg stubs).

~6-8M parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_level_inconsistency.learned_model_v2 import MultiScaleBayarConv, SRMFilterBank


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.GELU(),
    )


class _BranchA(nn.Module):
    """RGB context branch — 3ch → 16ch → 32ch, outputs at H and H/2."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _double_conv(3, 16)  # H
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = _double_conv(16, 32)  # H/2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.conv1(x)  # (B, 16, H, W)
        f2 = self.conv2(self.pool1(f1))  # (B, 32, H/2, W/2)
        return f1, f2


class _BranchB(nn.Module):
    """Bayar+SRM noise branch — 3ch → 45ch → 32ch → 64ch, outputs at H and H/2."""

    def __init__(self, bayar_target: str = "rgb") -> None:
        super().__init__()
        self.bayar_target = bayar_target
        noise_in = 1 if bayar_target == "y" else 3
        self.bayar = MultiScaleBayarConv(in_channels=noise_in,
                                         out_channels_per_scale=10)  # 30ch
        self.srm = SRMFilterBank(in_channels=noise_in, out_channels=15)  # 15ch
        self.conv1 = _double_conv(45, 32)  # H
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = _double_conv(32, 64)  # H/2

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bayar_target == "y":
            inp = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:,
                                                                          2:3]
        else:
            inp = rgb
        noise = torch.cat([self.bayar(inp), self.srm(inp)],
                          dim=1)  # (B, 45, H, W)
        f1 = self.conv1(noise)  # (B, 32, H, W)
        f2 = self.conv2(self.pool1(f1))  # (B, 64, H/2, W/2)
        return f1, f2


class _CNNEncoder(nn.Module):
    """Fused encoder: concat(A, B) → 4-level feature pyramid."""

    def __init__(self, bayar_target: str = "rgb") -> None:
        super().__init__()
        self.branch_a = _BranchA()
        self.branch_b = _BranchB(bayar_target)
        # After concat at H: 16+32=48ch; at H/2: 32+64=96ch
        self.fuse_h = _double_conv(48, 64)  # H    (skip for decoder)
        self.fuse_h2 = _double_conv(96, 96)  # H/2  (skip)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _double_conv(96, 128)  # H/4  (skip)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _double_conv(128, 256)  # H/8  (bottleneck)

    def forward(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        a1, a2 = self.branch_a(rgb)
        b1, b2 = self.branch_b(rgb)
        s1 = self.fuse_h(torch.cat([a1, b1], dim=1))  # (B, 64,  H,   W)
        s2 = self.fuse_h2(torch.cat([a2, b2], dim=1))  # (B, 96,  H/2, W/2)
        s3 = self.enc3(self.pool2(s2))  # (B, 128, H/4, W/4)
        s4 = self.enc4(self.pool3(s3))  # (B, 256, H/8, W/8)
        return [s1, s2, s3, s4]


class _UpBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=False)
        self.conv = _double_conv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x,
                              size=skip.shape[2:],
                              mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CNNNLIModel(nn.Module):
    """CNN-based NLI model with dual-branch encoder and UNet decoder."""

    def __init__(self,
                 decoder_out_ch: int = 64,
                 bayar_target: str = "rgb") -> None:
        super().__init__()
        self.encoder = _CNNEncoder(bayar_target)

        # Decoder: H/8 → H/4 → H/2 → H → full
        self.up3 = _UpBlock(256, 128, 128)  # H/8 + H/4 skip → H/4
        self.up2 = _UpBlock(128, 96, 96)  # H/4 + H/2 skip → H/2
        self.up1 = _UpBlock(96, 64, 64)  # H/2 + H   skip → H
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, decoder_out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, decoder_out_ch), decoder_out_ch),
            nn.GELU(),
        )
        self.upsample_full = nn.Upsample(scale_factor=2,
                                         mode="bilinear",
                                         align_corners=False)

        # Trust head from bottleneck
        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # Output heads
        self.pixel_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.confidence_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.boundary_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.noise_residual_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch,
                      decoder_out_ch // 2,
                      3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(min(8, decoder_out_ch // 2), decoder_out_ch // 2),
            nn.GELU(),
            nn.Conv2d(decoder_out_ch // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.sigma_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch,
                      decoder_out_ch // 2,
                      3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(min(8, decoder_out_ch // 2), decoder_out_ch // 2),
            nn.GELU(),
            nn.Conv2d(decoder_out_ch // 2, 1, 1),
            nn.ReLU(),
        )
        self.jpeg_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch,
                      decoder_out_ch // 2,
                      3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(min(8, decoder_out_ch // 2), decoder_out_ch // 2),
            nn.GELU(),
            nn.Conv2d(decoder_out_ch // 2, 1, 1),
            nn.Sigmoid(),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        for head in [
                self.pixel_head, self.confidence_head, self.boundary_head
        ]:
            nn.init.zeros_(head.bias)
            nn.init.kaiming_normal_(head.weight)
        for m in self.trust_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        s1, s2, s3, s4 = self.encoder(rgb)

        trust = self.trust_head(s4)

        d3 = self.up3(s4, s3)  # H/4
        d2 = self.up2(d3, s2)  # H/2
        d1 = self.up1(d2, s1)  # H
        decoded = self.upsample_full(
            self.final_conv(d1))  # 2× → full resolution

        # Align to input size
        if decoded.shape[2:] != rgb.shape[2:]:
            decoded = F.interpolate(decoded,
                                    size=rgb.shape[2:],
                                    mode="bilinear",
                                    align_corners=False)

        return {
            "pixel": self.pixel_head(decoded),
            "confidence": self.confidence_head(decoded),
            "boundary": self.boundary_head(decoded),
            "trust": trust,
            "aux": [],
            "n_hp_pred": self.noise_residual_head(decoded),
            "n_sigma_pred": self.sigma_head(decoded),
            "e_jpeg_pred": self.jpeg_head(decoded),
        }


def build_cnn_nli(decoder_out_ch: int = 64,
                  bayar_target: str = "rgb") -> CNNNLIModel:
    return CNNNLIModel(decoder_out_ch=decoder_out_ch,
                       bayar_target=bayar_target)
