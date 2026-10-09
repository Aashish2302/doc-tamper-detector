"""DINOv3 ViT-B/16 + CMX multi-scale encoder with noise modality.

Architecture:
  Backbone : DINOv3 ViT-B/16 (embed_dim=768, patch=16, 12 blocks)
             Optionally frozen + LoRA r/α on last N blocks.
             lora_rank=0 → no LoRA (floor / frozen-DINO baseline).
  Necks    : 4 lightweight heads that reshape the single /16 token grid to
             the 4 standard FPN resolutions:
               Neck0 : /16 → /4   (2× ConvTranspose2d)  → 64 ch
               Neck1 : /16 → /8   (1× ConvTranspose2d)  → 128 ch
               Neck2 : /16 → /16  (1×1 Conv)             → 320 ch
               Neck3 : /16 → /32  (stride-2 Conv)        → 512 ch
  Noise embedder: 4 parallel lightweight conv stacks that embed the
               pred_clean noise map (B,3,H,W) at the same 4 resolutions.
  FRM + FFM: applied at each of the 4 scales to fuse DINO + noise features.

Output: list of 4 fused tensors at [/4,/8,/16,/32] with channels [64,128,320,512].

Note: DINOv3 (dinov3_vitb16) is imported from the dinov3 repo at runtime so the
(non-commercial) code is imported only when actually needed.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from noise_level_inconsistency.trufor_dinob.cmx_fusion import (  # noqa: E402
    FeatureRectifyModule, FeatureFusionModule,
)

DINO_EMBED = 768
NECK_CHANNELS = [64, 128, 320, 512]
DINO_BLOCKS = 12
# Block indices tapped for multi-scale features (0-based)
FEATURE_BLOCKS = [2, 5, 8, 11]


# ── LoRA ─────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter (frozen base weight)."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.base = linear
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    # DINOv3's attention reads .in_features/.out_features off the qkv projection
    # (e.g. `C = self.qkv.in_features`). Expose them transparently so this wrapper
    # is a drop-in replacement for the nn.Linear it wraps.
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:  # some code paths read .weight for shape/dtype
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def _apply_lora_to_attention(block: nn.Module, rank: int, alpha: float):
    """Replace qkv projection in a DINOv3 attention block with LoRA wrapper.
    No-op if rank == 0 (floor / frozen baseline).
    """
    if rank == 0:
        return
    attn = getattr(block, "attn", None)
    if attn is None:
        return
    for name in ("qkv", "q", "k", "v"):
        lin = getattr(attn, name, None)
        if isinstance(lin, nn.Linear):
            setattr(attn, name, LoRALinear(lin, rank, alpha))


# ── Noise embedder ───────────────────────────────────────────────────────────

def _noise_conv_stack(out_ch: int, stride: int) -> nn.Sequential:
    """Lightweight conv stack: RGB (3ch) → out_ch at 1/stride resolution."""
    layers: list[nn.Module] = []
    in_ch = 3
    s_remaining = stride
    while s_remaining > 1:
        s = min(s_remaining, 2)
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        in_ch = out_ch
        s_remaining //= s
    if not layers:
        layers += [
            nn.Conv2d(3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


# ── Main encoder ─────────────────────────────────────────────────────────────

class DinoBCMXEncoder(nn.Module):
    """DINOv3 ViT-B/16 + per-scale CMX fusion with a noise (pred_clean) modality."""

    def __init__(
        self,
        dinov3_ckpt: str,
        dinov3_repo_dir: str,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_blocks: int = 6,
        freeze_backbone: bool = True,
        use_noise: bool = True,
    ):
        super().__init__()
        self.use_noise = use_noise

        # ── Load DINOv3 backbone ──────────────────────────────────────────
        if dinov3_repo_dir not in sys.path:
            sys.path.insert(0, dinov3_repo_dir)
        from dinov3.hub.backbones import dinov3_vitb16  # noqa: PLC0415
        backbone = dinov3_vitb16(pretrained=False)
        sd = torch.load(dinov3_ckpt, map_location="cpu", weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        missing, _ = backbone.load_state_dict(sd, strict=False)
        if missing:
            print(f"[DinoBCMX] missing keys: {len(missing)}", flush=True)

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        # Apply LoRA to last `lora_blocks` transformer blocks (skip if rank=0)
        if lora_rank > 0:
            blocks = list(backbone.blocks)
            for blk in blocks[DINO_BLOCKS - lora_blocks:]:
                _apply_lora_to_attention(blk, lora_rank, lora_alpha)

        self.backbone = backbone

        # ── Necks (DINO /16 tokens → 4 FPN scales) ────────────────────────
        self.neck0 = nn.Sequential(
            nn.ConvTranspose2d(DINO_EMBED, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, NECK_CHANNELS[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(NECK_CHANNELS[0]), nn.ReLU(inplace=True),
        )
        self.neck1 = nn.Sequential(
            nn.ConvTranspose2d(DINO_EMBED, NECK_CHANNELS[1], kernel_size=2, stride=2),
            nn.BatchNorm2d(NECK_CHANNELS[1]), nn.ReLU(inplace=True),
        )
        self.neck2 = nn.Sequential(
            nn.Conv2d(DINO_EMBED, NECK_CHANNELS[2], 1, bias=False),
            nn.BatchNorm2d(NECK_CHANNELS[2]), nn.ReLU(inplace=True),
        )
        self.neck3 = nn.Sequential(
            nn.Conv2d(DINO_EMBED, NECK_CHANNELS[3], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(NECK_CHANNELS[3]), nn.ReLU(inplace=True),
        )

        # ── Noise embedder + CMX (RGB-only mode: skip entirely -- these
        # modules exist iff use_noise=True, so an RGB-only checkpoint never
        # carries their weights and load_state_dict(strict=False) reports
        # them as unexpected/missing exactly like any other removed module) ──
        if use_noise:
            self.noise_embedder = nn.ModuleList([
                _noise_conv_stack(NECK_CHANNELS[i], stride)
                for i, stride in enumerate([4, 8, 16, 32])
            ])
            self.frm = nn.ModuleList([FeatureRectifyModule(c) for c in NECK_CHANNELS])
            self.ffm = nn.ModuleList([FeatureFusionModule(c) for c in NECK_CHANNELS])
        else:
            self.noise_embedder = None
            self.frm = None
            self.ffm = None

    def forward(self, rgb: torch.Tensor, noise_map: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            rgb:       (B, 3, H, W) original image, values in [0,1]
            noise_map: (B, 3, H, W) pred_clean noise estimate (rgb - pred_clean)
        Returns:
            list of 4 fused tensors at [/4, /8, /16, /32]
        """
        B, _, H, W = rgb.shape
        # Pad to multiples of 16
        pH = (16 - H % 16) % 16
        pW = (16 - W % 16) % 16
        if pH or pW:
            rgb = F.pad(rgb, (0, pW, 0, pH))
            noise_map = F.pad(noise_map, (0, pW, 0, pH))

        # ── DINO: extract 4 intermediate spatial feature maps ─────────────
        # get_intermediate_layers returns tuple of (B, C, H/16, W/16) tensors
        # when reshape=True. n=[2,5,8,11] taps blocks 2,5,8,11 (0-based).
        dino_feats = self.backbone.get_intermediate_layers(
            rgb, n=FEATURE_BLOCKS, reshape=True, norm=True,
        )  # tuple of 4 × (B, 768, H/16, W/16)

        # ── Build DINO 4-scale features via necks ─────────────────────────
        necks = [self.neck0, self.neck1, self.neck2, self.neck3]
        dino_scales = [neck(feat) for neck, feat in zip(necks, dino_feats)]

        if not self.use_noise:
            # RGB-only mode: noise_map (whatever the caller passed, typically
            # zeros from a stubbed noise_predictor) is never consumed. Return
            # the DINO features directly -- already at NECK_CHANNELS
            # [64,128,320,512], the exact shapes every downstream consumer
            # (decoder, Branch-B's GatedResidualFusion) already expects.
            return dino_scales

        # ── Noise 4-scale embeddings ───────────────────────────────────────
        noise_scales = [emb(noise_map) for emb in self.noise_embedder]

        # ── CMX fusion at each scale ───────────────────────────────────────
        fused = []
        for i in range(4):
            fa, fb = self.frm[i](dino_scales[i], noise_scales[i])
            fused.append(self.ffm[i](fa, fb))

        return fused


def build_dinob_cmx_encoder(
    dinov3_ckpt: str,
    dinov3_repo_dir: str,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_blocks: int = 6,
    use_noise: bool = True,
) -> DinoBCMXEncoder:
    return DinoBCMXEncoder(
        dinov3_ckpt=dinov3_ckpt,
        dinov3_repo_dir=dinov3_repo_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_blocks=lora_blocks,
        freeze_backbone=True,
        use_noise=use_noise,
    )
