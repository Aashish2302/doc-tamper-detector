"""Loss functions for V0.5-OCR training.

Differences from losses_v05.py:
  - ocr_masked_pixel_loss: replaces focal_dice_loss for the pixel head.
    Upweights OCR-region pixels (where forgeries are most likely) and computes
    Dice only inside the OCR mask to keep gradients focused on text regions.
  - giou_with_iou_penalty: extends standard GIoU with a soft IoU lower-bound
    penalty so the bbox head is penalised when predicted/GT overlap is too low.
  - bce_trust, noise_recon_loss: identical semantics to losses_v05.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ── Pixel mask loss ───────────────────────────────────────────────────────────


def balanced_focal_dice_loss(
    pixel_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    ocr_mask: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    focal_gamma: float = 2.0,
    dice_w: float = 1.0,
) -> torch.Tensor:
    """Group-balanced focal BCE + OCR-region Dice for the pixel tamper head.

    Replaces ``ocr_masked_pixel_loss`` (which used a `.mean()` over all
    pixels and got collapsed by 99% negative-pixel dominance regardless of
    focal_alpha or outside_weight). Here positives and negatives are
    averaged SEPARATELY at batch level, then combined with explicit
    pos/neg weights — so the per-pixel count imbalance can no longer
    drown out the positive gradient.

    Per pixel:
        bce[p]   = BCEWithLogits(logit[p], gt[p])
        focal[p] = (1 - p_t[p])^gamma · bce[p]   # focusing, no alpha here
    Group means (batch-level):
        pos_loss = mean(focal[gt > 0.5])  if any positive pixels exist
                 = 0                       otherwise (all-authentic batch)
        neg_loss = mean(focal[gt <= 0.5])
    Final:
        L_bal = pos_weight * pos_loss + neg_weight * neg_loss
        L     = L_bal + dice_w * dice(prob, gt, ocr_mask)

    Notes:
      * No outside_weight asymmetry — the group-balanced formulation makes
        OCR-side reweighting redundant (and was a workaround for the
        original count-imbalance failure mode anyway).
      * Dice is unchanged: ratio-based inside the OCR region, already
        balanced by construction.
      * Negatives outside the OCR mask are still part of the negative
        group, so predicting tamper there still costs loss — it just no
        longer dominates the gradient by sheer pixel count.

    Parameters
    ----------
    pixel_logits : Tensor  (B, 1, H, W)  raw (unactivated) logits.
    gt_mask      : Tensor  (B, 1, H, W)  binary ground-truth (float 0/1).
    ocr_mask     : Tensor  (B, 1, H, W)  binary OCR region mask (Dice only).
    pos_weight   : float   weight on the positive-group mean (default 1.0).
    neg_weight   : float   weight on the negative-group mean (default 1.0).
    focal_gamma  : float   focal focusing exponent (default 2.0).
    dice_w       : float   weight on the OCR-region Dice term.

    Returns
    -------
    Scalar loss tensor.
    """
    prob = torch.sigmoid(pixel_logits)
    bce = F.binary_cross_entropy_with_logits(pixel_logits,
                                             gt_mask,
                                             reduction="none")
    p_t = prob * gt_mask + (1.0 - prob) * (1.0 - gt_mask)
    focal = (1.0 - p_t).pow(focal_gamma) * bce  # (B, 1, H, W)

    pos_mask = gt_mask > 0.5
    neg_mask = ~pos_mask
    # Tensor-safe: avoid NaN when a batch has no positive pixels (all-authentic).
    if pos_mask.any():
        pos_loss = focal[pos_mask].mean()
    else:
        pos_loss = focal.new_zeros(())
    # Negative group is always non-empty in practice (every image has bg pixels).
    neg_loss = focal[neg_mask].mean() if neg_mask.any() else focal.new_zeros(
        ())

    balanced = pos_weight * pos_loss + neg_weight * neg_loss

    # OCR-region Dice (unchanged from ocr_masked_pixel_loss).
    ocr_pred = prob * ocr_mask
    ocr_gt = gt_mask * ocr_mask
    dice_num = 2.0 * (ocr_pred * ocr_gt).sum() + 1e-6
    dice_den = ocr_pred.sum() + ocr_gt.sum() + 1e-6
    dice = 1.0 - dice_num / dice_den

    return balanced + dice_w * dice


def ocr_masked_pixel_loss(
    pixel_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    ocr_mask: torch.Tensor,
    outside_weight: float = 3.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Asymmetric **focal** loss + OCR-region Dice for the pixel tamper head.

    Why focal (Option C, replacing plain BCE used previously):
      The pixel tamper task is highly imbalanced — only ~1% of in-OCR pixels
      are GT-positive, and ~95% of all pixels lie outside the OCR mask (all
      GT=0 after cleanup). Plain BCE × outside_weight=3.0 collapsed the model
      to "predict 0 everywhere" because easy negatives dominated the gradient
      (last run: best IoU = 0.0052). Focal modulation (1-p_t)^γ down-weights
      easy/well-classified examples so the gradient is dominated by hard /
      misclassified pixels — the rare positives and any leftover FPs.

    The asymmetric OCR weight (``outside_weight`` outside, 1.0 inside) is
    preserved on top of focal so out-of-OCR predictions still pay extra.

    Parameters
    ----------
    pixel_logits : Tensor  (B, 1, H, W)  — raw (unactivated) logits.
    gt_mask      : Tensor  (B, 1, H, W)  — binary ground-truth (float 0/1).
    ocr_mask     : Tensor  (B, 1, H, W)  — binary OCR region mask (float 0/1).
    outside_weight : float — weight for non-OCR pixels (default 3.0).
    focal_alpha    : float — class-balancing factor for positive class
                              (default 0.25 = standard focal recipe).
    focal_gamma    : float — focusing exponent (default 2.0); higher values
                              down-weight easy examples more aggressively.

    Returns
    -------
    Scalar loss tensor.
    """
    # ── 1) Focal-modulated BCE per pixel ─────────────────────────────────────
    prob = torch.sigmoid(pixel_logits)
    bce = F.binary_cross_entropy_with_logits(pixel_logits,
                                             gt_mask,
                                             reduction="none")
    # p_t: probability assigned to the true class
    p_t = prob * gt_mask + (1.0 - prob) * (1.0 - gt_mask)
    # alpha_t: class-balance term — α for positives, (1-α) for negatives
    alpha_t = focal_alpha * gt_mask + (1.0 - focal_alpha) * (1.0 - gt_mask)
    focal = alpha_t * (1.0 - p_t).pow(focal_gamma) * bce  # (B,1,H,W)

    # ── 2) Asymmetric OCR weight on top of focal ─────────────────────────────
    w = ocr_mask + outside_weight * (1.0 - ocr_mask)
    focal_weighted = focal * w

    # ── 3) Dice computed on OCR region only (unchanged) ──────────────────────
    ocr_pred = prob * ocr_mask
    ocr_gt = gt_mask * ocr_mask
    dice_num = 2.0 * (ocr_pred * ocr_gt).sum() + 1e-6
    dice_den = ocr_pred.sum() + ocr_gt.sum() + 1e-6
    dice = 1.0 - dice_num / dice_den

    return focal_weighted.mean() + dice


# ── BBox losses ───────────────────────────────────────────────────────────────


def _cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2) normalised coordinates.

    Parameters
    ----------
    b : Tensor  (..., 4)

    Returns
    -------
    Tensor  (..., 4)
    """
    cx, cy, w, h = b.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                       dim=-1)


def _compute_iou(p_xyxy: torch.Tensor, t_xyxy: torch.Tensor) -> torch.Tensor:
    """Element-wise IoU between two sets of boxes.

    Parameters
    ----------
    p_xyxy : Tensor  (B, 4)  predicted boxes in x1/y1/x2/y2 form.
    t_xyxy : Tensor  (B, 4)  target  boxes in x1/y1/x2/y2 form.

    Returns
    -------
    iou : Tensor  (B,)  per-element IoU in [0, 1].
    """
    ix1 = torch.max(p_xyxy[..., 0], t_xyxy[..., 0])
    iy1 = torch.max(p_xyxy[..., 1], t_xyxy[..., 1])
    ix2 = torch.min(p_xyxy[..., 2], t_xyxy[..., 2])
    iy2 = torch.min(p_xyxy[..., 3], t_xyxy[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ap = (p_xyxy[..., 2] - p_xyxy[..., 0]).clamp(
        min=0) * (p_xyxy[..., 3] - p_xyxy[..., 1]).clamp(min=0)
    at = (t_xyxy[..., 2] - t_xyxy[..., 0]).clamp(
        min=0) * (t_xyxy[..., 3] - t_xyxy[..., 1]).clamp(min=0)
    union = ap + at - inter + 1e-6
    return inter / union


def giou_with_iou_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    iou_threshold: float = 0.9,
    penalty_weight: float = 0.5,
) -> torch.Tensor:
    """GIoU loss with a soft lower-bound penalty on IoU.

    The standard GIoU term drives the boxes to overlap correctly.  The
    penalty term adds an extra signal when IoU is below ``iou_threshold``,
    preventing the model from converging to a solution where boxes partially
    overlap but never tightly fit the tampered region.

    Parameters
    ----------
    pred   : Tensor  (B, 4)  predicted bbox as normalised cx/cy/w/h in [0, 1].
    target : Tensor  (B, 4)  ground-truth bbox as normalised cx/cy/w/h in [0, 1].
    iou_threshold  : float   IoU threshold below which the penalty fires.
    penalty_weight : float   Scale factor for the IoU penalty term.

    Returns
    -------
    Scalar loss: giou_loss + penalty.
    """
    p = _cxcywh_to_xyxy(pred)
    t = _cxcywh_to_xyxy(target)

    # ── Intersection ─────────────────────────────────────────────────────────
    ix1 = torch.max(p[..., 0], t[..., 0])
    iy1 = torch.max(p[..., 1], t[..., 1])
    ix2 = torch.min(p[..., 2], t[..., 2])
    iy2 = torch.min(p[..., 3], t[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # ── Union ─────────────────────────────────────────────────────────────────
    ap = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] -
                                                 p[..., 1]).clamp(min=0)
    at = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] -
                                                 t[..., 1]).clamp(min=0)
    union = ap + at - inter + 1e-6

    # ── IoU ───────────────────────────────────────────────────────────────────
    iou = inter / union

    # ── Enclosing box ─────────────────────────────────────────────────────────
    ex1 = torch.min(p[..., 0], t[..., 0])
    ey1 = torch.min(p[..., 1], t[..., 1])
    ex2 = torch.max(p[..., 2], t[..., 2])
    ey2 = torch.max(p[..., 3], t[..., 3])
    enc = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + 1e-6

    # ── GIoU loss ─────────────────────────────────────────────────────────────
    giou = iou - (enc - union) / enc
    giou_loss = (1.0 - giou).mean()

    # ── IoU lower-bound penalty ───────────────────────────────────────────────
    # relu(threshold - iou) > 0 only when iou < threshold.
    penalty = penalty_weight * F.relu(iou_threshold - iou).mean()

    return giou_loss + penalty


# ── Trust / noise-reconstruction losses ──────────────────────────────────────


def bce_trust(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for the trust head.

    Parameters
    ----------
    logits : Tensor  (B, 1) or (B,)  — raw trust logits.
    target : Tensor  (B, 1) or (B,)  — binary labels (float 0/1).

    Returns
    -------
    Scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(logits.flatten(),
                                              target.flatten().float())


def noise_recon_loss(pred: torch.Tensor, ci_gt: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss for the noise-level head.

    Parameters
    ----------
    pred  : Tensor  (B, 1, H, W)  — model output (sigmoid-activated, in [0,1]).
    ci_gt : Tensor  (B, 1, H, W)  — CI-GT noise-level proxy (in [0,1]).

    Returns
    -------
    Scalar L1 loss.
    """
    return F.l1_loss(pred, ci_gt)
