"""NLI Transformer V0.5-OCR — 4-channel (R, G, B, OCR_mask) input variant.

Differences from V0.5 (learned_model_transformer_v05.py):
  1. rgb_encoder (PVT backbone) receives 4-channel input: in_channels=4.
     The extra channel is the OCR binary mask concatenated to RGB.
  2. noise_encoder (CNNNoiseEncoder) still receives only the 3-channel RGB
     (channels 0-2 of x) — its Bayar/SRM filters are designed for raw pixel
     residuals and should not see the OCR mask.
  3. forward() signature changes: x is (B, 4, H, W) rather than (B, 3, H, W).
  4. All other modules (gates, decoder, heads) are identical to V0.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_level_inconsistency.learned_model_transformer import BidirectionalDecoder
from noise_level_inconsistency.learned_model_transformer_v2 import CNNNoiseEncoder
from noise_level_inconsistency.pvt_v2 import build_pvt_v2_b0, build_pvt_v2_b1, build_pvt_v2_b2


class NPNLITransformerV05OCR(nn.Module):
    """NLI Transformer V0.5 with 4-channel (RGB + OCR mask) backbone input.

    The PVT backbone receives the full 4-channel tensor so it can attend to
    text regions via the OCR mask.  The CNN noise encoder only sees RGB so
    its forensic filter bank (Bayar + SRM) operates on unaltered pixel values.

    Parameters
    ----------
    backbone_size : str
        PVT variant: 'b0', 'b1', or 'b2'.  Default 'b2'.
    decoder_out_ch : int
        Output channels from the BidirectionalDecoder.  Default 64.
    bayar_target : str
        Input mode for CNNNoiseEncoder: 'rgb' (3-ch) or 'y' (1-ch luma).
        Default 'rgb'.
    """

    def __init__(
        self,
        backbone_size: str = "b2",
        decoder_out_ch: int = 64,
        bayar_target: str = "rgb",
        use_ocr_decoder_fusion: bool = False,
    ) -> None:
        """
        use_ocr_decoder_fusion:
            Option I — late-fusion injection of the OCR mask at the decoder's
            pixel-head input. When True, the OCR channel (full image resolution)
            is concatenated with the upsampled decoder feature map *immediately
            before* the pixel-head 1x1 conv, so the pixel head sees both the
            encoder-distilled features AND a crisp pixel-level OCR prior.
            Mitigates the PVT-v2 backbone's 4-32x spatial downsampling, which
            smears the OCR boundary by the time it reaches the decoder.
            When False (default): legacy V0.5-OCR behaviour — OCR only enters
            via the 4-channel encoder input.
        """
        super().__init__()
        self.use_ocr_decoder_fusion = use_ocr_decoder_fusion

        builders = {
            "b0": build_pvt_v2_b0,
            "b1": build_pvt_v2_b1,
            "b2": build_pvt_v2_b2,
        }
        if backbone_size not in builders:
            raise ValueError(
                f"backbone_size must be one of {list(builders)}, got '{backbone_size}'"
            )

        # PVT backbone: 4-channel input (RGB + OCR mask).
        self.rgb_encoder = builders[backbone_size](in_channels=4)
        embed_dims: list[
            int] = self.rgb_encoder.embed_dims  # e.g. [64, 128, 320, 512]

        # CNN noise encoder: 3-channel RGB only (unchanged from V0.5).
        self.noise_encoder = CNNNoiseEncoder(bayar_target=bayar_target,
                                             embed_dims=embed_dims)

        # One learnable scalar gate per pyramid level (init 0 → tanh(0)=0 → neutral).
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in embed_dims])

        self.decoder = BidirectionalDecoder(embed_dims, out_ch=decoder_out_ch)

        # ── Heads (identical to V0.5 except optional decoder OCR fusion) ─────
        # Pixel head: 1x1 conv on decoder features.
        # When use_ocr_decoder_fusion=True, the input channels are
        # (decoder_out_ch + 1) so the head also consumes a full-resolution
        # OCR mask channel concatenated at the head input.
        pixel_in_ch = decoder_out_ch + 1 if use_ocr_decoder_fusion else decoder_out_ch
        self.pixel_head = nn.Conv2d(pixel_in_ch, 1, 1)

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

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_heads(self) -> None:
        nn.init.kaiming_normal_(self.pixel_head.weight)
        nn.init.zeros_(self.pixel_head.bias)
        for m in list(self.bbox_head.modules()) + list(
                self.trust_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model.

        Parameters
        ----------
        x : Tensor  shape (B, 4, H, W)
            Channels 0-2 are RGB (float, normalised), channel 3 is the OCR
            binary mask (float 0/1).

        Returns
        -------
        dict with keys:
            'pixel'       (B, 1, H, W)  — raw logits for tamper mask
            'bbox'        (B, 4)         — normalised cx/cy/w/h in [0,1]
            'trust'       (B, 1)         — raw trust logit
            'noise_recon' (B, 1, H, W)  — noise-level reconstruction in [0,1]
        """
        rgb = x[:, :3]  # (B, 3, H, W) — for the CNN noise encoder
        full = x  # (B, 4, H, W) — for the PVT backbone

        # Backbone features (4 scales).
        rgb_feats = self.rgb_encoder(full)

        # Noise-encoder features (3-ch RGB only).
        noise_feats = self.noise_encoder(rgb)

        # Localization path: noise features detached so L_pixel/L_bbox/L_trust
        # cannot push gradients back into the noise CNN.
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

        # Noise reconstruction path: full gradients through noise_feats.
        noise_decoded, _ = self.decoder(list(noise_feats))
        noise_recon = self.noise_recon_head(noise_decoded)
        if noise_recon.shape[2:] != x.shape[2:]:
            noise_recon = F.interpolate(noise_recon,
                                        size=x.shape[2:],
                                        mode="bilinear",
                                        align_corners=False)

        # Pixel head — Option I: late-fuse the OCR channel at full resolution
        # so the pixel head sees a CRISP OCR prior (the encoder-side OCR has
        # been spatially smeared by 4-32x downsampling by this point).
        if self.use_ocr_decoder_fusion:
            # Upsample decoder feature map to full resolution first.
            if decoded.shape[2:] != x.shape[2:]:
                decoded_full = F.interpolate(decoded,
                                             size=x.shape[2:],
                                             mode="bilinear",
                                             align_corners=False)
            else:
                decoded_full = decoded
            # Concat with the OCR mask channel from the input (channel 3),
            # which is already at full resolution.
            ocr_chw = x[:, 3:4]  # (B, 1, H, W)
            pixel_in = torch.cat([decoded_full, ocr_chw], dim=1)
            pixel = self.pixel_head(pixel_in)
        else:
            pixel = self.pixel_head(decoded)
            if pixel.shape[2:] != x.shape[2:]:
                pixel = F.interpolate(pixel,
                                      size=x.shape[2:],
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


# ── Factory ───────────────────────────────────────────────────────────────────


def build_nli_transformer_v05_ocr(
    backbone_size: str = "b2",
    decoder_out_ch: int = 64,
    bayar_target: str = "rgb",
    use_ocr_decoder_fusion: bool = False,
) -> NPNLITransformerV05OCR:
    """Instantiate an NPNLITransformerV05OCR model.

    Parameters
    ----------
    backbone_size : str
        PVT size: 'b0', 'b1', or 'b2'.
    decoder_out_ch : int
        Output channels for BidirectionalDecoder.
    bayar_target : str
        Noise encoder input mode: 'rgb' or 'y'.
    use_ocr_decoder_fusion : bool
        Option I — late-fuse the OCR mask channel at the pixel-head input
        (full image resolution) in addition to the existing 4-channel
        encoder injection. Default False (legacy V0.5-OCR behaviour).

    Returns
    -------
    NPNLITransformerV05OCR
    """
    return NPNLITransformerV05OCR(
        backbone_size=backbone_size,
        decoder_out_ch=decoder_out_ch,
        bayar_target=bayar_target,
        use_ocr_decoder_fusion=use_ocr_decoder_fusion,
    )


def warm_start_v05_ocr(
    model: NPNLITransformerV05OCR,
    run_f_ckpt: str | Path,
    run_d_ckpt: str | Path | None = None,
) -> Tuple[int, int, int]:
    """Load rgb_encoder + decoder from a V0.5 checkpoint; noise_encoder from a
    separate checkpoint if provided.

    The rgb_encoder patch_embed1 weight has shape (embed_dim, 3, 7, 7) in the
    V0.5 checkpoint but (embed_dim, 4, 7, 7) here — that key is skipped
    automatically (shape mismatch) so the extra OCR channel is randomly
    initialised, which is the correct behaviour.

    Returns
    -------
    (rgb_loaded, noise_loaded, skipped) key counts.
    """
    ms = model.state_dict()
    rgb_loaded = 0
    noise_loaded = 0
    skipped = 0

    f_ckpt = torch.load(run_f_ckpt, map_location="cpu", weights_only=True)
    f_state = f_ckpt.get("model_state", f_ckpt.get("model_state_dict", f_ckpt))
    rgb_filtered: dict[str, torch.Tensor] = {}
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
        noise_filtered: dict[str, torch.Tensor] = {}
        for k, v in d_state.items():
            if k.startswith("noise_encoder."):
                if k in ms and v.shape == ms[k].shape:
                    noise_filtered[k] = v
                    noise_loaded += 1
                else:
                    skipped += 1
        model.load_state_dict(noise_filtered, strict=False)

    return rgb_loaded, noise_loaded, skipped
