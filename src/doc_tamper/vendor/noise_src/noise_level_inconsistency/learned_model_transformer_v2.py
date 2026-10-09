"""NP-NLI Transformer V2 — PVT-B2 + CNN Branch-B noise encoder + gated fusion.

Architecture change from V1:
  V1: noise stem fused only at H/4 (stage 1 only; stages 2/3/4 are pure RGB).
  V2: 4-stage CNN noise encoder produces features at H/4, H/8, H/16, H/32.
      Each level is injected into the corresponding decoder level via a learned
      scalar gate (tanh-gated residual, initialised to 0 = neutral).

Goal: break the RGB shortcut — forcing noise features to compete with content
features at every decoder level, not just the first.

Zero changes to learned_model_transformer.py. Selected via
  ablation:
    architecture_version: "v2"
in the training config.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_level_inconsistency.pvt_v2 import build_pvt_v2_b2, build_pvt_v2_b1, build_pvt_v2_b0
from noise_level_inconsistency.learned_model_v2 import MultiScaleBayarConv, SRMFilterBank
from noise_level_inconsistency.learned_model_transformer import (
    BidirectionalDecoder, )


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.GELU(),
    )


class CNNNoiseEncoder(nn.Module):
    """4-stage CNN noise encoder (Branch B only).

    Outputs features at H/4, H/8, H/16, H/32 matching PVT-B2 embed_dims
    [64, 128, 320, 512]. No Branch A (RGB context) — PVT backbone handles that.
    """

    def __init__(self,
                 bayar_target: str = "rgb",
                 embed_dims: list[int] | None = None) -> None:
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 320, 512]
        self.bayar_target = bayar_target
        d0, d1, d2, d3 = embed_dims

        noise_in = 1 if bayar_target == "y" else 3
        self.bayar = MultiScaleBayarConv(in_channels=noise_in,
                                         out_channels_per_scale=10)
        self.srm = SRMFilterBank(in_channels=noise_in, out_channels=15)

        # Stage 0: 45ch → d0@H/4 (two stride-2 convs to match PVT stage-1 stride-4)
        self.stage0 = nn.Sequential(
            nn.Conv2d(45, d0, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, d0), d0),
            nn.GELU(),
            nn.Conv2d(d0, d0, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, d0), d0),
            nn.GELU(),
        )  # → d0@H/4

        # Stage 1: d0 → d1@H/8
        self.stage1 = _conv_block(d0, d1, stride=2)

        # Stage 2: d1 → d2@H/16
        self.stage2 = _conv_block(d1, d2, stride=2)

        # Stage 3: d2 → d3@H/32
        self.stage3 = _conv_block(d2, d3, stride=2)

    def forward(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        if self.bayar_target == "y":
            inp = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:,
                                                                          2:3]
        else:
            inp = rgb
        noise = torch.cat([self.bayar(inp), self.srm(inp)],
                          dim=1)  # (B, 45, H, W)
        n0 = self.stage0(noise)  # (B, d0, H/4,  W/4)
        n1 = self.stage1(n0)  # (B, d1, H/8,  W/8)
        n2 = self.stage2(n1)  # (B, d2, H/16, W/16)
        n3 = self.stage3(n2)  # (B, d3, H/32, W/32)
        return [n0, n1, n2, n3]


class NPNLITransformerV2(nn.Module):
    """NP-NLI Transformer V2: PVT-B2 content backbone + CNN noise encoder.

    Gated fusion at all 4 feature levels:
        fused_i = rgb_feat_i + tanh(gate_i) * noise_feat_i
    Gates initialised to 0 (tanh(0)=0) so training starts from pure PVT baseline.
    """

    def __init__(
        self,
        backbone_size: str = "b2",
        decoder_out_ch: int = 64,
        bayar_target: str = "rgb",
    ) -> None:
        super().__init__()
        builders = {
            "b0": build_pvt_v2_b0,
            "b1": build_pvt_v2_b1,
            "b2": build_pvt_v2_b2,
        }
        self.rgb_encoder = builders[backbone_size](in_channels=3)
        embed_dims: list[
            int] = self.rgb_encoder.embed_dims  # [64, 128, 320, 512] for B2

        self.noise_encoder = CNNNoiseEncoder(bayar_target=bayar_target,
                                             embed_dims=embed_dims)

        # Per-level tanh gates (init=0 → neutral, no noise injection at start)
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in embed_dims])

        self.decoder = BidirectionalDecoder(embed_dims, out_ch=decoder_out_ch)

        # Output heads (identical to V1)
        self.pixel_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.confidence_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.boundary_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dims[3], 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
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
        rgb_feats = self.rgb_encoder(rgb)  # [f0@H/4, f1@H/8, f2@H/16, f3@H/32]
        noise_feats = self.noise_encoder(
            rgb)  # [n0@H/4, n1@H/8, n2@H/16, n3@H/32]

        # Gated injection: noise_feat may differ in spatial size due to rounding — align
        fused = []
        for i, (rf, nf,
                gate) in enumerate(zip(rgb_feats, noise_feats, self.gates)):
            if nf.shape[2:] != rf.shape[2:]:
                nf = F.interpolate(nf,
                                   size=rf.shape[2:],
                                   mode="bilinear",
                                   align_corners=False)
            fused.append(rf + torch.tanh(gate) * nf)

        trust = self.trust_head(fused[3])
        decoded, aux = self.decoder(fused)

        return {
            "pixel": self.pixel_head(decoded),
            "confidence": self.confidence_head(decoded),
            "boundary": self.boundary_head(decoded),
            "trust": trust,
            "aux": aux if self.training else [],
            "n_hp_pred": self.noise_residual_head(decoded),
            "n_sigma_pred": self.sigma_head(decoded),
            "e_jpeg_pred": self.jpeg_head(decoded),
        }


def build_nli_transformer_v2(
    backbone_size: str = "b2",
    decoder_out_ch: int = 64,
    bayar_target: str = "rgb",
) -> NPNLITransformerV2:
    return NPNLITransformerV2(
        backbone_size=backbone_size,
        decoder_out_ch=decoder_out_ch,
        bayar_target=bayar_target,
    )
