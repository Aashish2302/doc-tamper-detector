"""V0.5-OCR decoder-only ablation — encoder is RGB-only, OCR enters ONLY
at the pixel head via late fusion.

ISOLATED from the main `learned_model_transformer_v05_ocr.py` module —
this file is a self-contained variant for the encoder=3ch + decoder OCR-
fusion experiment. No imports or modifications touch the main module.

Differences vs the main V0.5-OCR model:
  1. PVT backbone built with in_channels=3 (RGB only).
  2. CNNNoiseEncoder still sees RGB (unchanged).
  3. forward(): rgb_encoder receives x[:, :3]; OCR channel (x[:, 3:4]) is
     reserved and concatenated with the upsampled decoder features
     immediately before the pixel-head 1x1 conv (always on — there is no
     other place OCR is used in this variant).

Input contract is identical to the main model: forward expects a
(B, 4, H, W) tensor with channels [R, G, B, OCR]; the OCR channel just
takes a different path through the network.

Why isolated: the user wants this as an ablation without touching the
main shared model code. To revert: delete this file (no other module
imports it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_level_inconsistency.learned_model_transformer import BidirectionalDecoder
from noise_level_inconsistency.learned_model_transformer_v2 import CNNNoiseEncoder
from noise_level_inconsistency.pvt_v2 import (
    build_pvt_v2_b0,
    build_pvt_v2_b1,
    build_pvt_v2_b2,
)


class NPNLITransformerV05OCRDecOnly(nn.Module):
    """V0.5-OCR decoder-only OCR variant.

    Encoder sees only RGB. OCR mask enters only at the pixel head as a
    full-resolution channel concatenated with the upsampled decoder
    features before the final 1x1 conv.
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
        if backbone_size not in builders:
            raise ValueError(
                f"backbone_size must be one of {list(builders)}, got '{backbone_size}'"
            )

        # PVT backbone: 3-channel input (RGB only) — the I-vs-this ablation.
        self.rgb_encoder = builders[backbone_size](in_channels=3)
        embed_dims: list[int] = self.rgb_encoder.embed_dims

        # CNN noise encoder: 3-channel RGB only (identical to main model).
        self.noise_encoder = CNNNoiseEncoder(bayar_target=bayar_target,
                                             embed_dims=embed_dims)

        self.gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in embed_dims])

        self.decoder = BidirectionalDecoder(embed_dims, out_ch=decoder_out_ch)

        # Pixel head accepts (decoder_out_ch + 1) channels: decoder features
        # + full-resolution OCR mask concatenated at the head input.
        self.pixel_head = nn.Conv2d(decoder_out_ch + 1, 1, 1)

        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dims[3], 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dims[3], 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        self.noise_recon_head = nn.Sequential(
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
        nn.init.kaiming_normal_(self.pixel_head.weight)
        nn.init.zeros_(self.pixel_head.bias)
        for m in list(self.bbox_head.modules()) + list(
                self.trust_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        x : Tensor (B, 4, H, W) — [R, G, B, OCR].
            The first 3 channels go to BOTH the PVT encoder and the CNN
            noise encoder. The OCR channel is held aside until the pixel
            head, where it is concatenated full-resolution with the
            upsampled decoder features.
        """
        rgb = x[:, :3]  # (B, 3, H, W) — both encoders consume this.

        rgb_feats = self.rgb_encoder(rgb)
        noise_feats = self.noise_encoder(rgb)

        fused: list[torch.Tensor] = []
        for rf, nf, gate in zip(rgb_feats, noise_feats, self.gates):
            if nf.shape[2:] != rf.shape[2:]:
                nf_aligned = F.interpolate(nf,
                                           size=rf.shape[2:],
                                           mode="bilinear",
                                           align_corners=False)
            else:
                nf_aligned = nf
            fused.append(rf + torch.tanh(gate) * nf_aligned.detach())

        decoded, _aux = self.decoder(fused)

        noise_decoded, _ = self.decoder(list(noise_feats))
        noise_recon = self.noise_recon_head(noise_decoded)
        if noise_recon.shape[2:] != x.shape[2:]:
            noise_recon = F.interpolate(noise_recon,
                                        size=x.shape[2:],
                                        mode="bilinear",
                                        align_corners=False)

        # Pixel head: upsample decoder features to full res, concat OCR.
        if decoded.shape[2:] != x.shape[2:]:
            decoded_full = F.interpolate(decoded,
                                         size=x.shape[2:],
                                         mode="bilinear",
                                         align_corners=False)
        else:
            decoded_full = decoded
        ocr_chw = x[:, 3:4]  # (B, 1, H, W) — full-resolution OCR mask.
        pixel_in = torch.cat([decoded_full, ocr_chw], dim=1)
        pixel = self.pixel_head(pixel_in)

        bbox = self.bbox_head(fused[3])
        trust = self.trust_head(fused[3])

        return {
            "pixel": pixel,
            "bbox": bbox,
            "trust": trust,
            "noise_recon": noise_recon,
        }


def build_nli_transformer_v05_ocr_dec_only(
    backbone_size: str = "b2",
    decoder_out_ch: int = 64,
    bayar_target: str = "rgb",
) -> NPNLITransformerV05OCRDecOnly:
    """Factory for the decoder-only OCR ablation model."""
    return NPNLITransformerV05OCRDecOnly(
        backbone_size=backbone_size,
        decoder_out_ch=decoder_out_ch,
        bayar_target=bayar_target,
    )
