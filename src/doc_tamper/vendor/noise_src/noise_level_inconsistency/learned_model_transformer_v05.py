"""NLI Transformer V0.5 — pure-noise V2 with stop-grad + pruned heads.

Differences from V2 (learned_model_transformer_v2.py):
  1. Stop-grad on noise_feats before fusion: L_pixel/L_trust gradients
     cannot reshape the noise CNN. Noise encoder's only teacher is L_noise
     (patch-L1 vs CI-GT) flowing through its own decoder pass + recon head.
  2. Pruned heads: pixel mask + bbox + trust + internal noise_recon only.
     Removed: confidence, boundary, n_sigma, e_jpeg.
  3. Bbox head: 4-D regression (cx, cy, w, h) on the deepest fused feature.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_level_inconsistency.learned_model_transformer import BidirectionalDecoder
from noise_level_inconsistency.learned_model_transformer_v2 import CNNNoiseEncoder
from noise_level_inconsistency.pvt_v2 import (build_pvt_v2_b0, build_pvt_v2_b1,
                                              build_pvt_v2_b2)


class NPNLITransformerV05(nn.Module):

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
            "b2": build_pvt_v2_b2
        }
        self.rgb_encoder = builders[backbone_size](in_channels=3)
        embed_dims = self.rgb_encoder.embed_dims
        self.noise_encoder = CNNNoiseEncoder(bayar_target=bayar_target,
                                             embed_dims=embed_dims)
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in embed_dims])
        self.decoder = BidirectionalDecoder(embed_dims, out_ch=decoder_out_ch)

        self.pixel_head = nn.Conv2d(decoder_out_ch, 1, 1)
        self.bbox_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(embed_dims[3], 256),
                                       nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(256, 4), nn.Sigmoid())
        self.trust_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(embed_dims[3], 256),
                                        nn.GELU(), nn.Dropout(0.3),
                                        nn.Linear(256, 1))
        self.noise_recon_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch,
                      decoder_out_ch // 2,
                      3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(min(8, decoder_out_ch // 2), decoder_out_ch // 2),
            nn.GELU(), nn.Conv2d(decoder_out_ch // 2, 1, 1), nn.Sigmoid())
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

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb_feats = self.rgb_encoder(rgb)
        noise_feats = self.noise_encoder(rgb)

        # Localization path: noise features detached so L_pixel/L_bbox/L_trust
        # cannot push gradients into the noise CNN.
        fused = []
        for rf, nf, gate in zip(rgb_feats, noise_feats, self.gates):
            if nf.shape[2:] != rf.shape[2:]:
                nf_d = F.interpolate(nf,
                                     size=rf.shape[2:],
                                     mode="bilinear",
                                     align_corners=False)
            else:
                nf_d = nf
            fused.append(rf + torch.tanh(gate) * nf_d.detach())

        decoded, _aux = self.decoder(fused)

        # Noise reconstruction path: full-grad through noise_feats.
        # Use the noise features alone (no RGB) — the recon head's only signal
        # is what the noise CNN produces.
        noise_decoded, _ = self.decoder(list(noise_feats))
        noise_recon = self.noise_recon_head(noise_decoded)
        if noise_recon.shape[2:] != rgb.shape[2:]:
            noise_recon = F.interpolate(noise_recon,
                                        size=rgb.shape[2:],
                                        mode="bilinear",
                                        align_corners=False)

        pixel = self.pixel_head(decoded)
        if pixel.shape[2:] != rgb.shape[2:]:
            pixel = F.interpolate(pixel,
                                  size=rgb.shape[2:],
                                  mode="bilinear",
                                  align_corners=False)
        bbox = self.bbox_head(fused[3])
        trust = self.trust_head(fused[3])

        return {
            "pixel": pixel,
            "bbox": bbox,
            "trust": trust,
            "noise_recon": noise_recon,
        }


def build_nli_transformer_v05(
    backbone_size: str = "b2",
    decoder_out_ch: int = 64,
    bayar_target: str = "rgb",
) -> NPNLITransformerV05:
    return NPNLITransformerV05(backbone_size=backbone_size,
                               decoder_out_ch=decoder_out_ch,
                               bayar_target=bayar_target)


def warm_start_v05(
    model: NPNLITransformerV05,
    run_f_ckpt: str | Path,
    run_d_ckpt: str | Path | None = None,
) -> Tuple[int, int, int]:
    """Load rgb_encoder + decoder from run_f, noise_encoder from run_d.

    Returns (rgb_loaded, noise_loaded, skipped) key counts.
    """
    ms = model.state_dict()
    rgb_loaded = 0
    noise_loaded = 0
    skipped = 0

    f_ckpt = torch.load(run_f_ckpt, map_location="cpu", weights_only=True)
    f_state = f_ckpt.get("model_state", f_ckpt.get("model_state_dict", f_ckpt))
    rgb_filtered = {}
    for k, v in f_state.items():
        if k.startswith(("rgb_encoder.", "decoder.")):
            if k in ms and v.shape == ms[k].shape:
                rgb_filtered[k] = v
                rgb_loaded += 1
            else:
                skipped += 1
    model.load_state_dict(rgb_filtered, strict=False)

    if run_d_ckpt is not None:
        d_ckpt = torch.load(run_d_ckpt, map_location="cpu", weights_only=True)
        d_state = d_ckpt.get("model_state",
                             d_ckpt.get("model_state_dict", d_ckpt))
        noise_filtered = {}
        for k, v in d_state.items():
            if k.startswith("noise_encoder."):
                if k in ms and v.shape == ms[k].shape:
                    noise_filtered[k] = v
                    noise_loaded += 1
                else:
                    skipped += 1
        model.load_state_dict(noise_filtered, strict=False)

    return rgb_loaded, noise_loaded, skipped
