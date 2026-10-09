"""TruForDinoBPredCleanModel — Run C architecture.

DINOv3 ViT-B/16 (frozen + LoRA) fused with a trainable pred_clean noise branch
via CMX (FRM+FFM), decoded by a SegFormer MLP head.

Key design choice: the noise branch predicts the additive noise component via
  pred_clean = rgb - noise_predictor(rgb)["noise"]
This residual is the forensic signal (noise inconsistency map). The DINO branch
provides high-level semantic context; the noise branch resolves fine-scale tamper
traces below DINO's 16px patch floor.

Forward returns:
  (logits, pred_clean, modal_x, aux)
  logits    : (B, num_classes, H, W) — upsampled to input resolution
  pred_clean: (B, 3, H, W) — denoised image (rgb - noise)
  modal_x   : (B, 3, H, W) — raw noise estimate from predictor
  aux       : None (reserved for future trust head)

Freeze behaviour:
  freeze_noise=True  : noise branch weights frozen (warm-start from ckpt, no grad)
  freeze_noise=False : noise branch fully trainable (Run C setting)

Usage:
  model = build_trufor_dinob_pred_clean(
      dinov3_ckpt=..., dinov3_repo_dir=...,
      noise_v0_ckpt=...,   # None → random init
      freeze_noise=False,
  )
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_level_inconsistency.trufor_dinob.dinob_cmx_encoder import (  # noqa: E402
    DinoBCMXEncoder, NECK_CHANNELS,
)
from noise_level_inconsistency.trufor_dinob.mlp_decoder import DecoderHead  # noqa: E402

REPO = None  # set at import time if needed


def detect_noise_backbone(sd: dict) -> str | None:
    """Infer the PVT-v2 backbone size (b0/b1/b2) of a noise_v0 predictor from a
    state dict.

    Works for both a raw Stage-1 checkpoint (keys prefixed 'encoder.') and a full
    Stage-2 model_state (keys prefixed 'noise_predictor.encoder.'). Returns None
    if no matching patch_embed1 tensor is found.

    Rule (see build_pvt_v2_b{0,1,2}):
        patch_embed1 out-channels == 32           -> b0  ([32,64,160,256])
        patch_embed1 out-channels == 64, depth1==3 -> b2 (depths [3,4,6,3])
        patch_embed1 out-channels == 64, depth1==2 -> b1 (depths [2,2,2,2])
    """
    import re
    suffix = "encoder.patch_embed1.proj.weight"
    pe1_key = next((k for k in sd if k.endswith(suffix)), None)
    if pe1_key is None:
        return None
    prefix = pe1_key[:-len(suffix)]  # "" or "noise_predictor."
    c1 = sd[pe1_key].shape[0]
    if c1 == 32:
        return "b0"
    depth1 = len({
        int(m.group(1))
        for k in sd
        for m in [re.match(re.escape(prefix) + r"encoder\.blocks1\.(\d+)\.", k)]
        if m
    })
    return "b2" if depth1 >= 3 else "b1"


class TruForDinoBPredCleanModel(nn.Module):
    """Run C model: DINOv3 + trainable pred_clean noise branch + CMX + MLP head."""

    def __init__(
        self,
        encoder: DinoBCMXEncoder,
        decoder: DecoderHead,
        noise_predictor: nn.Module,
        freeze_noise: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise_predictor = noise_predictor
        self.num_classes = num_classes

        if freeze_noise:
            for p in self.noise_predictor.parameters():
                p.requires_grad_(False)

        # Image-level score head (trust/authenticity)
        self.trust_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_classes, 1),
        )

    def forward(self, rgb: torch.Tensor):
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1]
        Returns:
            (logits, pred_clean, modal_x, trust_logit)
        """
        B, _, H, W = rgb.shape

        # Noise branch: modal_x = noise estimate; pred_clean = rgb - noise
        with torch.set_grad_enabled(
                not any(not p.requires_grad
                        for p in self.noise_predictor.parameters())):
            noise_out = self.noise_predictor(rgb)
            modal_x = noise_out["noise"]          # (B,3,H,W)
        pred_clean = rgb - modal_x                # (B,3,H,W)

        # Encoder: DINO + noise CMX fusion → 4 scales
        features = self.encoder(rgb, modal_x)

        # Decoder: 4-scale → logits at H/4
        logits_small = self.decoder(features)    # (B, C, H/4, W/4) approx

        # Upsample to input resolution
        logits = F.interpolate(logits_small, size=(H, W), mode="bilinear",
                               align_corners=False)

        # Trust score
        trust = self.trust_head(logits)          # (B, 1)

        return logits, pred_clean, modal_x, trust


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_tamper_maps(model, rgb: torch.Tensor, tile: int = 512,
                        overlap: int = 128, device=None):
    """Sliding-window inference → full-resolution (prob, logit) maps for the
    tamper class.

    The CMX FeatureFusionModule does global spatial attention (O((H*W)^2)), so
    full-page inference OOMs. The model is trained on `tile`-sized crops, so we
    run it on overlapping `tile` windows and average — memory stays flat at the
    training footprint regardless of page size. Returns two (H, W) CPU tensors:
    the class-1 probability map and the class-1 logit map.
    """
    model.eval()
    _, _, H, W = rgb.shape
    stride = max(1, tile - overlap)
    prob_acc = torch.zeros(H, W)
    logit_acc = torch.zeros(H, W)
    wsum = torch.zeros(H, W)
    for y0 in _tile_starts(H, tile, stride):
        for x0 in _tile_starts(W, tile, stride):
            patch = rgb[:, :, y0:y0 + tile, x0:x0 + tile]
            if device is not None:
                patch = patch.to(device)
            out, _, _, _ = model(patch)
            out = out[0].float().cpu()                 # (2, h, w)
            ph, pw = out.shape[-2:]
            prob_acc[y0:y0 + ph, x0:x0 + pw] += torch.softmax(out, dim=0)[1]
            logit_acc[y0:y0 + ph, x0:x0 + pw] += out[1]
            wsum[y0:y0 + ph, x0:x0 + pw] += 1.0
    wsum = wsum.clamp(min=1.0)
    return prob_acc / wsum, logit_acc / wsum


def build_trufor_dinob_pred_clean(
    dinov3_ckpt: str,
    dinov3_repo_dir: str,
    noise_v0_ckpt: str | None = None,
    freeze_noise: bool = False,
    num_classes: int = 2,
    decoder_embed_dim: int = 512,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_blocks: int = 6,
    noise_arch: str = "v0",
    noise_backbone: str = "auto",
    use_noise: bool = True,
) -> TruForDinoBPredCleanModel:
    """Factory for the Run C model.

    Args:
        dinov3_ckpt:      path to DINOv3 ViT-B/16 checkpoint
        dinov3_repo_dir:  path to dinov3 repo (for importable dinov3.hub.backbones)
        noise_v0_ckpt:    path to noise_only_v0 Stage-1 checkpoint (None = random)
        freeze_noise:     freeze noise branch weights
        num_classes:      2 (tamper / clean)
        decoder_embed_dim: SegFormer embed dim (512)
        lora_rank:        LoRA rank on DINO attention (8)
        lora_alpha:       LoRA alpha (16)
        lora_blocks:      number of last DINO blocks to apply LoRA (6)
        noise_arch:       which noise predictor to use ("v0" = noise_only_v0 Stage-1)
        noise_backbone:   PVT size for the "v0" noise branch: "b0"/"b1"/"b2", or
                          "auto" (default) to infer it from noise_v0_ckpt's tensor
                          shapes. "auto" with no ckpt falls back to "b2" (the
                          historical default). Callers loading a full Stage-2
                          checkpoint after build (e.g. eval) must pass the size
                          explicitly, detected via detect_noise_backbone(model_state).
    """
    # Build encoder
    encoder = DinoBCMXEncoder(
        dinov3_ckpt=dinov3_ckpt,
        dinov3_repo_dir=dinov3_repo_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_blocks=lora_blocks,
        freeze_backbone=True,
        use_noise=use_noise,
    )

    if not use_noise:
        # RGB-only mode: no noise branch at all. ZeroNoise has zero learnable
        # parameters and returns a zero map in the same {"noise": ...} dict
        # shape the rest of the forward path expects, so no other call site
        # (DualBranchTamperModel.forward, TruForDinoBPredCleanModel.forward)
        # needs to change: modal_x becomes literal zeros, pred_clean == rgb,
        # and recon_loss(modal_x=0, ...) is automatically 0 with no special
        # casing needed in the training loop.
        class ZeroNoise(nn.Module):
            def forward(self, rgb):
                return {"noise": torch.zeros_like(rgb)}
        noise_predictor = ZeroNoise()
        decoder = DecoderHead(in_channels=NECK_CHANNELS,
                              embed_dim=decoder_embed_dim, num_classes=num_classes)
        return TruForDinoBPredCleanModel(
            encoder=encoder, decoder=decoder, noise_predictor=noise_predictor,
            freeze_noise=freeze_noise, num_classes=num_classes,
        )

    # Build noise predictor
    if noise_arch == "v0":
        from noise_only_v0.learned_model_noise_v0 import build_noise_v0_stage1  # noqa: E402
        # Load the warm-start ckpt first (if any) so we can size the backbone to
        # match it — Stage-1 denoisers may be b0/b1/b2 and a mismatch here is a
        # hard load_state_dict error.
        sd = None
        if noise_v0_ckpt is not None:
            sd = torch.load(noise_v0_ckpt, map_location="cpu", weights_only=False)
            if "model_state" in sd:
                sd = sd["model_state"]
            elif "model" in sd:
                sd = sd["model"]
        bb = noise_backbone
        if bb == "auto":
            bb = (detect_noise_backbone(sd) if sd is not None else None) or "b2"
        noise_predictor = build_noise_v0_stage1(backbone_size=bb)
        if sd is not None:
            missing, unexp = noise_predictor.load_state_dict(sd, strict=False)
            print(f"[PredClean] noise branch (backbone={bb}) loaded from {noise_v0_ckpt}",
                  f"missing={len(missing)} unexp={len(unexp)}", flush=True)
        else:
            print(f"[PredClean] noise branch (backbone={bb}) random init", flush=True)
    elif noise_arch == "nafnet":
        from noise_only_v0.noise_nafnet import build_noise_nafnet  # noqa: PLC0415
        noise_predictor = build_noise_nafnet()
    elif noise_arch == "restormer":
        from noise_only_v0.noise_restormer import build_noise_restormer  # noqa: PLC0415
        noise_predictor = build_noise_restormer()
    else:
        raise ValueError(f"Unknown noise_arch: {noise_arch}")

    # Build decoder
    decoder = DecoderHead(
        in_channels=NECK_CHANNELS,
        embed_dim=decoder_embed_dim,
        num_classes=num_classes,
    )

    return TruForDinoBPredCleanModel(
        encoder=encoder,
        decoder=decoder,
        noise_predictor=noise_predictor,
        freeze_noise=freeze_noise,
        num_classes=num_classes,
    )
