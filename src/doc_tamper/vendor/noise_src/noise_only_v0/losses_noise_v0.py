"""Losses for noise_V0 training.

Stage 1 (denoiser):
  L = || (noisy - pred) - clean ||_1   (equivalent to ||pred - (noisy-clean)||_1)
  Optional: + lambda_cigt * patch_L1(|pred|, CI-GT) for the with-CI-GT variant.

Stage 2 (tamper heads on noise tensor): reuses focal+dice / GIoU / BCE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def denoise_l1(pred_noise: torch.Tensor, noisy: torch.Tensor,
               clean: torch.Tensor) -> torch.Tensor:
    """L = mean(|noisy - pred - clean|).

    The model predicts the additive noise such that pred ≈ noisy - clean.
    """
    residual = noisy - pred_noise - clean
    return residual.abs().mean()


def denoise_charbonnier(pred_noise: torch.Tensor,
                        noisy: torch.Tensor,
                        clean: torch.Tensor,
                        eps: float = 1e-3) -> torch.Tensor:
    residual = noisy - pred_noise - clean
    return torch.sqrt(residual.pow(2) + eps**2).mean()


def denoise_l2(pred_noise: torch.Tensor, noisy: torch.Tensor,
               clean: torch.Tensor) -> torch.Tensor:
    """L = MSE(pred_noise, noisy - clean) = mean((noisy - pred - clean)^2).

    Replaces L1 to break the median-collapse failure mode. L1 minimised by
    predicting the median residual (~0 for mostly-clean document pixels), so
    the model converged to pred_noise=0 / PSNR=25 dB. L2's gradient scales
    with residual magnitude, so the rare high-noise pixels (text edges,
    JPEG ringing) exert enough gradient to escape the trivial zero.
    """
    residual = noisy - pred_noise - clean
    return residual.pow(2).mean()


def denoise_l2_pool(pred_noise: torch.Tensor, noisy: torch.Tensor,
                    clean: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Region-pooled L2: average the residual over non-overlapping kxk blocks,
    then MSE on the block means.

    L = mean( AvgPool_kxk(noisy - pred - clean)^2 )

    Replaces the pixel-wise L2 with a region-level target: the model is only
    penalised for the MEAN error inside each kxk block, not per pixel. This
    tolerates high-frequency per-pixel error (blurrier pred_clean, lower PSNR
    by design) while still driving the low-/mid-frequency structure of the
    residual — the band that carries region-level noise-level differences.
    AvgPool is linear, so this equals MSE(pool(pred_clean), pool(clean)).
    """
    residual = noisy - pred_noise - clean
    pooled = F.avg_pool2d(residual, kernel_size=k, stride=k, ceil_mode=False)
    return pooled.pow(2).mean()


def ssim_loss_on_denoised(pred_noise: torch.Tensor, noisy: torch.Tensor,
                          clean: torch.Tensor) -> torch.Tensor:
    """1 - SSIM(noisy - pred_noise, clean).

    Structural pressure on the denoised reconstruction; complements MSE which
    only cares about pixel-wise magnitudes. Inputs in [0, 1].
    """
    denoised = (noisy - pred_noise).clamp(0.0, 1.0)
    return 1.0 - ssim(denoised, clean)


def patch_l1_to_cigt(pred_noise: torch.Tensor,
                     ci_gt: torch.Tensor,
                     patch: int = 16) -> torch.Tensor:
    """Patch-mean magnitude of pred should match the CI-GT proxy.

    pred_noise : (B, 3, H, W) — signed noise estimate
    ci_gt      : (B, 1, H, W) — proxy noise level in [0, 1]
    """
    mag = pred_noise.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
    mag_avg = F.avg_pool2d(mag, patch)
    gt_avg = F.avg_pool2d(ci_gt, patch)
    return (mag_avg - gt_avg).abs().mean()


def psnr(pred_clean: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """PSNR in dB. Inputs assumed in [0, 1]."""
    mse = ((pred_clean - clean)**2).mean()
    return -10.0 * torch.log10(mse.clamp(min=1e-10))


def ssim(pred_clean: torch.Tensor,
         clean: torch.Tensor,
         window: int = 11) -> torch.Tensor:
    """Simple single-scale SSIM. Inputs in [0, 1]."""
    c1 = 0.01**2
    c2 = 0.03**2
    pad = window // 2
    pred_g = pred_clean.mean(dim=1, keepdim=True)
    clean_g = clean.mean(dim=1, keepdim=True)
    mu_x = F.avg_pool2d(pred_g, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(clean_g, window, stride=1, padding=pad)
    sig_x = F.avg_pool2d(pred_g**2, window, stride=1, padding=pad) - mu_x**2
    sig_y = F.avg_pool2d(clean_g**2, window, stride=1, padding=pad) - mu_y**2
    sig_xy = (F.avg_pool2d(pred_g * clean_g, window, stride=1, padding=pad) -
              mu_x * mu_y)
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2)
    return (num / den.clamp(min=1e-10)).mean()


# ── Stage-2 losses (reused from V0.5) ────────────────────────────────────────


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
    x1 = torch.max(p[..., 0], t[..., 0])
    y1 = torch.max(p[..., 1], t[..., 1])
    x2 = torch.min(p[..., 2], t[..., 2])
    y2 = torch.min(p[..., 3], t[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    pa = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    ta = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    union = pa + ta - inter
    iou = inter / union.clamp(min=1e-6)
    ex1 = torch.min(p[..., 0], t[..., 0])
    ey1 = torch.min(p[..., 1], t[..., 1])
    ex2 = torch.max(p[..., 2], t[..., 2])
    ey2 = torch.max(p[..., 3], t[..., 3])
    enc = (ex2 - ex1) * (ey2 - ey1)
    giou = iou - (enc - union) / enc.clamp(min=1e-6)
    return (1 - giou).mean()


def bce_trust(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.flatten(),
                                              target.flatten())
