"""V0.5 losses — patch-L1 (noise), focal+dice (mask), GIoU (bbox), BCE (trust)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_l1_loss(pred: torch.Tensor,
                  target: torch.Tensor,
                  patch: int = 16) -> torch.Tensor:
    """Average L1 over non-overlapping patches.

    Patch averaging blocks the trivial 'predict zero' minimum that pixel-L1
    admits when the target is sparse — keeps signal on small high-energy regions.
    """
    p_avg = F.avg_pool2d(pred, kernel_size=patch, stride=patch)
    t_avg = F.avg_pool2d(target, kernel_size=patch, stride=patch)
    return F.l1_loss(p_avg, t_avg)


def focal_dice_loss(logits: torch.Tensor,
                    target: torch.Tensor,
                    alpha: float = 0.25,
                    gamma: float = 2.0,
                    dice_w: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    focal = (alpha * (1 - p_t).pow(gamma) * bce).mean()
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice = 1.0 - (2 * inter / denom).mean()
    return focal + dice_w * dice


def _cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                       dim=-1)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = _cxcywh_to_xyxy(pred)
    t = _cxcywh_to_xyxy(target)
    ix1 = torch.max(p[..., 0], t[..., 0])
    iy1 = torch.max(p[..., 1], t[..., 1])
    ix2 = torch.min(p[..., 2], t[..., 2])
    iy2 = torch.min(p[..., 3], t[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ap = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] -
                                                 p[..., 1]).clamp(min=0)
    at = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] -
                                                 t[..., 1]).clamp(min=0)
    union = ap + at - inter + 1e-6
    iou = inter / union
    ex1 = torch.min(p[..., 0], t[..., 0])
    ey1 = torch.min(p[..., 1], t[..., 1])
    ex2 = torch.max(p[..., 2], t[..., 2])
    ey2 = torch.max(p[..., 3], t[..., 3])
    enc = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + 1e-6
    giou = iou - (enc - union) / enc
    return (1 - giou).mean()


def bce_trust(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(-1).float(), label.float())
