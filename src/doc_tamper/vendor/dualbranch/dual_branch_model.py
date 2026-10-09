"""Dual-branch tamper localiser: DINOv3 vision branch + semantic-prior branch.

    RGB ──► Branch A (existing, untouched): DINOv3 ViT-B/16 + LoRA
            + PVT-v2 noise residual, CMX-fused           ──► F_v^k  (4 scales)
                                                                │
    S   ──► Branch B (new): SemanticEncoder               ──► F_s^k │
                                                                │   │
                                          MultiScaleFusion ◄────┴───┘
                                                                │
                                          SegFormer decoder ◄────┘  ──► logits

Branch A is instantiated by calling the *existing* factory unmodified, so
config A (single-branch) remains byte-identical to the current production
system. Everything new lives in this package.

Forward signature matches the original model -- `(logits, pred_clean, modal_x,
trust)` -- so existing evaluation code consumes it unchanged.

    IMPORTANT: `forward(rgb)` with no semantic map substitutes an all-zero map.
    That is correct-by-construction at initialisation (alpha=0) but NOT after
    training, when a trained Branch B will read the zero map as "no regions
    anywhere". Always pass the real semantic map at inference; use
    `predict_tamper_maps_dual` below, which tiles image and map together.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gated_fusion import MultiScaleFusion                     # noqa: E402
from semantic_dropout import SemanticChannelDropout           # noqa: E402
from semantic_encoder import (                                # noqa: E402
    NECK_CHANNELS, SEMANTIC_CHANNELS, SemanticEncoder, SemanticChannelGate,
)


def _add_repo_paths(root: str = None):
    """Put the vendored packages on sys.path (they are not pip-installed).
    Paths are resolved relative to this file's vendored location so the repo is portable."""
    _vendor = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../doc_tamper/vendor
    for p in (
        os.path.join(_vendor, "noise_src"),    # noise_level_inconsistency + noise_only_v0 packages
        os.path.join(_vendor, "dinov3_repo"),  # dinov3 package
        os.path.join(_vendor, "dualbranch"),   # gated_fusion / semantic_encoder helpers
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


class DualBranchTamperModel(nn.Module):
    """Branch A (vision) + Branch B (semantic prior), gated-residual fused."""

    def __init__(
        self,
        base_model: nn.Module,
        semantic_encoder: Optional[SemanticEncoder] = None,
        fusion_mode: str = "residual",
        semantic_channels: Optional[Sequence[str]] = None,
        dropout_p: float = 0.15,
        fusion_kernel: int = 1,
        alpha_init: float = 0.0,
        use_aux_head: bool = False,
        use_channel_gate: bool = False,
    ):
        super().__init__()
        self.base = base_model
        self.num_classes = getattr(base_model, "num_classes", 2)
        self.use_aux_head = use_aux_head
        self.use_channel_gate = use_channel_gate

        # Which semantic channels this config uses (C: text only, D: all five).
        # Disabled channels are *masked to zero* rather than removed, so the
        # input width stays 5 and checkpoints remain loadable across configs.
        enabled = list(semantic_channels) if semantic_channels is not None \
            else list(SEMANTIC_CHANNELS)
        unknown = [c for c in enabled if c not in SEMANTIC_CHANNELS]
        if unknown:
            raise ValueError(f"unknown semantic channels {unknown}; "
                             f"valid: {SEMANTIC_CHANNELS}")
        self.semantic_channels = enabled
        mask = torch.zeros(len(SEMANTIC_CHANNELS))
        for c in enabled:
            mask[SEMANTIC_CHANNELS.index(c)] = 1.0
        self.register_buffer("channel_mask", mask.view(1, -1, 1, 1))

        self.semantic_encoder = semantic_encoder or SemanticEncoder()
        self.channel_gate = SemanticChannelGate(len(SEMANTIC_CHANNELS)) \
            if use_channel_gate else None
        self.semantic_dropout = SemanticChannelDropout(p=dropout_p)
        self.fusion = MultiScaleFusion(
            NECK_CHANNELS, mode=fusion_mode,
            kernel_size=fusion_kernel, alpha_init=alpha_init,
        )
        # Grounding-loss aux head (opt-in): a 1x1 conv on the finest-scale
        # (/4) semantic feature, tapped BEFORE fusion, producing its own
        # coarse tamper logits P_sem. Only meaningful when semantic input is
        # real (configs D/F), and only built when requested so existing
        # checkpoints (which never had this layer) still load strict=True.
        # See train_dual_branch.py for how P_sem is used in the loss.
        self.aux_head = nn.Conv2d(NECK_CHANNELS[0], 2, kernel_size=1) \
            if use_aux_head else None

    # ---------------------------------------------------------------- helpers
    def alphas(self) -> List[float]:
        return self.fusion.alphas()

    def branch_b_parameters(self):
        yield from self.semantic_encoder.parameters()
        yield from self.fusion.parameters()

    def load_branch_a_state(self, state_dict: dict, verbose: bool = True):
        """Load an existing single-branch checkpoint into Branch A.

        Branch B / fusion params are new, so `strict=False` and we assert that
        every reported missing key belongs to the new modules -- if a Branch A
        key were missing that would be a silent, serious regression.
        """
        missing, unexpected = self.load_state_dict(
            {f"base.{k}" if not k.startswith("base.") else k: v
             for k, v in state_dict.items()}, strict=False)
        new_prefixes = ("semantic_encoder.", "fusion.", "semantic_dropout.",
                        "channel_mask", "aux_head.", "channel_gate.")
        unexplained = [k for k in missing if not k.startswith(new_prefixes)]
        if verbose:
            print(f"[DualBranch] loaded Branch A: {len(missing)} missing "
                  f"({len(missing) - len(unexplained)} are new Branch B params), "
                  f"{len(unexpected)} unexpected", flush=True)
        if unexplained:
            raise RuntimeError(
                f"{len(unexplained)} Branch A keys missing from checkpoint, "
                f"e.g. {unexplained[:5]}")
        return missing, unexpected

    def _prepare_semantic(self, rgb: torch.Tensor,
                          semantic: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, H, W = rgb.shape
        n_ch = len(SEMANTIC_CHANNELS)
        if semantic is None:
            return rgb.new_zeros(B, n_ch, H, W)
        if semantic.shape[1] != n_ch:
            raise ValueError(f"semantic map must have {n_ch} channels, "
                             f"got {semantic.shape[1]}")
        semantic = semantic.to(rgb.dtype)
        if semantic.shape[-2:] != (H, W):
            semantic = F.interpolate(semantic, size=(H, W), mode="nearest")
        semantic = semantic * self.channel_mask          # config C vs D
        if self.channel_gate is not None:
            semantic = self.channel_gate(semantic)
        return self.semantic_dropout(semantic)

    # ---------------------------------------------------------------- forward
    def forward(self, rgb: torch.Tensor, semantic: Optional[torch.Tensor] = None,
                return_aux: bool = False):
        """
        Args:
            rgb:      (B, 3, H, W) in [0, 1]
            semantic: (B, 5, H, W) in [0, 1], or None (-> zeros; see caveat above)
            return_aux: if True, also return `sem_logits` (B, 2, H, W) from the
                        grounding-loss aux head -- requires use_aux_head=True.
        Returns:
            (logits, pred_clean, modal_x, trust) -- same as the single-branch model,
            or (logits, pred_clean, modal_x, trust, sem_logits) if return_aux=True.
        """
        B, _, H, W = rgb.shape

        # ---- Branch A (unchanged code path) ----------------------------
        with torch.set_grad_enabled(
                not any(not p.requires_grad
                        for p in self.base.noise_predictor.parameters())):
            modal_x = self.base.noise_predictor(rgb)["noise"]
        pred_clean = rgb - modal_x
        vision_feats = self.base.encoder(rgb, modal_x)          # 4 scales

        # ---- Branch B --------------------------------------------------
        sem = self._prepare_semantic(rgb, semantic)
        semantic_feats = self.semantic_encoder(sem)

        # Branch B pads differently from the vision encoder (which pads to a
        # multiple of 16 internally), so sizes can differ by a pixel at odd
        # input sizes. Align to the vision feature grid, which is canonical.
        aligned = []
        for fv, fs in zip(vision_feats, semantic_feats):
            if fs.shape[-2:] != fv.shape[-2:]:
                fs = F.interpolate(fs, size=fv.shape[-2:], mode="bilinear",
                                   align_corners=False)
            aligned.append(fs)

        fused = self.fusion(vision_feats, aligned)

        # ---- Decoder (unchanged) ---------------------------------------
        logits_small = self.base.decoder(fused)
        logits = F.interpolate(logits_small, size=(H, W), mode="bilinear",
                               align_corners=False)
        trust = self.base.trust_head(logits)

        if return_aux:
            if self.aux_head is None:
                raise RuntimeError(
                    "return_aux=True but this model was built with "
                    "use_aux_head=False -- no aux_head layer exists")
            sem_logits_small = self.aux_head(aligned[0])
            sem_logits = F.interpolate(sem_logits_small, size=(H, W),
                                       mode="bilinear", align_corners=False)
            return logits, pred_clean, modal_x, trust, sem_logits
        return logits, pred_clean, modal_x, trust


# ---------------------------------------------------------------- factory
def build_dual_branch(
    dinov3_ckpt: str,
    dinov3_repo_dir: str,
    noise_v0_ckpt: Optional[str] = None,
    freeze_noise: bool = False,
    num_classes: int = 2,
    decoder_embed_dim: int = 512,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_blocks: int = 6,
    noise_backbone: str = "auto",
    fusion_mode: str = "residual",
    semantic_channels: Optional[Sequence[str]] = None,
    dropout_p: float = 0.15,
    alpha_init: float = 0.0,
    use_aux_head: bool = False,
    use_channel_gate: bool = False,
    use_noise: bool = True,
    base_model: Optional[nn.Module] = None,
) -> DualBranchTamperModel:
    """Build the dual-branch model.

    `base_model` lets a caller inject an already-built (or stubbed) Branch A,
    which is what the CPU smoke test uses to avoid loading a 342MB ViT.
    """
    if base_model is None:
        _add_repo_paths()
        from noise_level_inconsistency.trufor_dinob.trufor_dinob_pred_clean_model import (  # noqa: E402
            build_trufor_dinob_pred_clean,
        )
        base_model = build_trufor_dinob_pred_clean(
            dinov3_ckpt=dinov3_ckpt,
            dinov3_repo_dir=dinov3_repo_dir,
            noise_v0_ckpt=noise_v0_ckpt,
            freeze_noise=freeze_noise,
            num_classes=num_classes,
            decoder_embed_dim=decoder_embed_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_blocks=lora_blocks,
            noise_backbone=noise_backbone,
            use_noise=use_noise,
        )
    return DualBranchTamperModel(
        base_model=base_model,
        fusion_mode=fusion_mode,
        semantic_channels=semantic_channels,
        dropout_p=dropout_p,
        alpha_init=alpha_init,
        use_aux_head=use_aux_head,
        use_channel_gate=use_channel_gate,
    )


def _tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_tamper_maps_dual(model, rgb: torch.Tensor, semantic: torch.Tensor,
                             tile: int = 512, overlap: int = 256, device=None):
    """Sliding-window inference for the dual-branch model.

    Mirrors the single-branch `predict_tamper_maps`, but crops the semantic map
    with the *same* window as the image -- otherwise Branch B would see a prior
    that does not correspond to the patch Branch A is looking at.
    """
    model.eval()
    _, _, H, W = rgb.shape
    if semantic.shape[-2:] != (H, W):
        semantic = F.interpolate(semantic, size=(H, W), mode="nearest")
    stride = max(1, tile - overlap)
    prob_acc = torch.zeros(H, W)
    logit_acc = torch.zeros(H, W)
    wsum = torch.zeros(H, W)
    for y0 in _tile_starts(H, tile, stride):
        for x0 in _tile_starts(W, tile, stride):
            patch = rgb[:, :, y0:y0 + tile, x0:x0 + tile]
            spatch = semantic[:, :, y0:y0 + tile, x0:x0 + tile]
            if device is not None:
                patch, spatch = patch.to(device), spatch.to(device)
            out, _, _, _ = model(patch, spatch)
            out = out[0].float().cpu()
            ph, pw = out.shape[-2:]
            prob_acc[y0:y0 + ph, x0:x0 + pw] += torch.softmax(out, dim=0)[1]
            logit_acc[y0:y0 + ph, x0:x0 + pw] += out[1]
            wsum[y0:y0 + ph, x0:x0 + pw] += 1.0
    wsum = wsum.clamp(min=1.0)
    return prob_acc / wsum, logit_acc / wsum
