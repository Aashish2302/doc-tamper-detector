"""V0.5-OCR dataset — 4-channel (RGB + OCR mask) input for outputs_100k.

Returns per sample:
  image    : (4, H, W) float32   channels 0-2 = RGB/255, channel 3 = OCR mask/255
  mask     : (1, H, W) float32   tamper GT mask in {0, 1}
  bbox     : (4,)  float32       normalised cx/cy/w/h in [0,1]; zeros if authentic
  label    : scalar float32      0.0 or 1.0
  ocr_mask : (1, H, W) float32   same as channel-3, kept separate for loss gating

Split JSON format (produced by make_splits_synthetic.py):
  {
    "doc_id": "orig_020647",
    "label": 0,
    "noisy_path": "/abs/.../originals/images/orig_020647.jpg",
    "clean_path": "/abs/.../clean/originals/orig_020647.png",
    "mask_path": null,
    "tamper_bboxes": []
  }
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Reuse the V2 smart-crop helpers (positive / hard-neg / random tile modes)
# and the full augment pipeline. These already lockstep an extras list of
# (H,W) float maps with the image crop/rotate/flip — perfect for the OCR mask.
from noise_level_inconsistency.docforge_dataset_v2 import (
    _augment_nli_v2 as _augment_v2_lockstep, )

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SyntheticOCREntry:
    doc_id: str
    label: int  # 0 = authentic, 1 = tampered
    noisy_path: Path
    mask_path: Optional[Path]
    tamper_bboxes: list[list[int]]  # pixel [x1, y1, x2, y2] bboxes


# ---------------------------------------------------------------------------
# Split loader
# ---------------------------------------------------------------------------


def load_synthetic_entries(
    splits_root: str | Path,
    split_name: str,
) -> list[SyntheticOCREntry]:
    """Load entries from <splits_root>/<split_name>.json.

    Args:
        splits_root: Directory that contains {train,val,test}.json files.
        split_name:  One of "train", "val", "test".

    Returns:
        List of SyntheticOCREntry.
    """
    splits_root = Path(splits_root)
    split_path = splits_root / f"{split_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    raw: list[dict] = json.loads(split_path.read_text(encoding="utf-8"))
    entries: list[SyntheticOCREntry] = []
    for item in raw:
        mask_raw = item.get("mask_path")
        entries.append(
            SyntheticOCREntry(
                doc_id=item["doc_id"],
                label=int(item["label"]),
                noisy_path=Path(item["noisy_path"]),
                mask_path=Path(mask_raw) if mask_raw else None,
                tamper_bboxes=item.get("tamper_bboxes") or [],
            ))
    return entries


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _union_bboxes_normalised(
    bboxes: list[list[int]],
    h: int,
    w: int,
) -> np.ndarray:
    """Convert a list of [x1,y1,x2,y2] pixel bboxes to a single normalised
    cx/cy/w/h bbox (float32 array of shape (4,)).  Returns zeros if empty."""
    if not bboxes:
        return np.zeros(4, dtype=np.float32)
    xs0 = min(b[0] for b in bboxes)
    ys0 = min(b[1] for b in bboxes)
    xs1 = max(b[2] for b in bboxes)
    ys1 = max(b[3] for b in bboxes)
    cx = ((xs0 + xs1) / 2.0) / w
    cy = ((ys0 + ys1) / 2.0) / h
    bw = (xs1 - xs0) / w
    bh = (ys1 - ys0) / h
    return np.array([cx, cy, bw, bh], dtype=np.float32)


def _bbox_from_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Derive normalised cx/cy/w/h from a binary mask via connected components.

    Returns zeros if the mask is empty.
    """
    h, w = mask_bin.shape
    m8 = (mask_bin > 0.5).astype(np.uint8)
    if m8.sum() == 0:
        return np.zeros(4, dtype=np.float32)

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    # Collect all foreground component bboxes and take their union.
    bboxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 1:
            continue
        bboxes.append([x, y, x + bw, y + bh])

    if not bboxes:
        return np.zeros(4, dtype=np.float32)

    return _union_bboxes_normalised(bboxes, h, w)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _random_crop_or_pad(
    arrays: list[np.ndarray],
    target_size: int,
    rng: random.Random,
) -> list[np.ndarray]:
    """Apply the same random crop (with reflect-padding if needed) to all arrays.

    Each array in `arrays` must have shape (H, W) or (H, W, C).
    All arrays must share the same (H, W).
    """
    h, w = arrays[0].shape[:2]
    ts = target_size

    # Pad if smaller than target.
    if h < ts or w < ts:
        pad_h = max(0, ts - h)
        pad_w = max(0, ts - w)
        ph0, ph1 = pad_h // 2, pad_h - pad_h // 2
        pw0, pw1 = pad_w // 2, pad_w - pad_w // 2
        padded = []
        for arr in arrays:
            if arr.ndim == 2:
                arr = np.pad(arr, ((ph0, ph1), (pw0, pw1)), mode="reflect")
            else:
                arr = np.pad(arr, ((ph0, ph1), (pw0, pw1), (0, 0)),
                             mode="reflect")
            padded.append(arr)
        arrays = padded
        h, w = arrays[0].shape[:2]

    # Random crop.
    y0 = rng.randint(0, h - ts)
    x0 = rng.randint(0, w - ts)
    cropped = []
    for arr in arrays:
        if arr.ndim == 2:
            cropped.append(arr[y0:y0 + ts, x0:x0 + ts])
        else:
            cropped.append(arr[y0:y0 + ts, x0:x0 + ts, :])
    return cropped


def _augment(
    rgb: np.ndarray,  # (H, W, 3) uint8
    mask: np.ndarray,  # (H, W)    uint8
    ocr_mask: np.ndarray,  # (H, W)    uint8
    target_size: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop + horizontal flip.  Returns (rgb, mask, ocr_mask)."""
    rgb_out, mask_out, ocr_out = _random_crop_or_pad([rgb, mask, ocr_mask],
                                                     target_size, rng)
    # Horizontal flip (p=0.5).
    if rng.random() < 0.5:
        rgb_out = rgb_out[:, ::-1, :].copy()
        mask_out = mask_out[:, ::-1].copy()
        ocr_out = ocr_out[:, ::-1].copy()
    return rgb_out, mask_out, ocr_out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SyntheticOCRDataset(Dataset):
    """PyTorch Dataset for V0.5-OCR training on outputs_100k synthetic data.

    Each sample exposes a 4-channel image tensor (RGB + OCR text mask) together
    with the tamper ground-truth mask, bounding box, and class label.

    Args:
        entries:        List of SyntheticOCREntry (from load_synthetic_entries).
        ocr_masks_root: Path to outputs_100k/ocr_masks/.
        target_size:    Spatial size (H=W) of returned tensors (default 256).
        augment:        Enable random crop + flip augmentation (default True).
        seed:           Base RNG seed for reproducibility (default 42).
        tiles_per_image: Number of random crops sampled per image (default 1).
                         When >1 the Dataset length is len(entries)*tiles_per_image;
                         __getitem__ maps index → (entry, tile_index) internally.
    """

    def __init__(
        self,
        entries: list[SyntheticOCREntry],
        ocr_masks_root: Path,
        target_size: int = 256,
        augment: bool = True,
        seed: int = 42,
        tiles_per_image: int = 1,
        ci_gt_root: Path | None = None,
        use_tile_sampling: bool = False,
        crop_sizes: tuple[int, ...] = (256, 384, 512),
        full_page_prob: float = 0.3,
    ) -> None:
        """
        use_tile_sampling:
            PDF §9 within-doc spatial-aware K-tile cropping. When True AND
            entry has tamper_bboxes:
              tile 0 = forced POSITIVE (crop guaranteed to overlap a tamper bbox)
              tile 1 = forced HARD-NEG (crop avoids all tamper bboxes)
              tiles 2+: 50% positive / 30% hard-neg / 20% uniform random
            Ignored when augment=False or when tamper_bboxes is empty.
        crop_sizes:
            Pool of crop sizes sampled per tile; ported from the V0.5
            (non-OCR) recipe.
        full_page_prob:
            Probability of full-page downscale instead of cropping. With
            use_tile_sampling on, this should be small (e.g. 0.1) so most
            tiles preserve the positive-forcing signal.
        """
        super().__init__()
        self.entries = entries
        self.ocr_masks_root = Path(ocr_masks_root)
        self.target_size = target_size
        self.augment = augment
        self.seed = seed
        self.tiles_per_image = max(1, int(tiles_per_image))
        self.ci_gt_root = Path(ci_gt_root) if ci_gt_root is not None else None
        self.use_tile_sampling = use_tile_sampling
        self.crop_sizes = tuple(crop_sizes)
        self.full_page_prob = full_page_prob if augment else 0.0

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries) * self.tiles_per_image

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ocr_mask(self, entry: SyntheticOCREntry, h: int,
                       w: int) -> np.ndarray:
        """Load OCR text mask (uint8, same size as image).

        Falls back to all-ones (conservative: treat everything as potential text)
        if the file is missing.
        """
        bucket = "originals" if entry.label == 0 else "tampered"
        ocr_path = self.ocr_masks_root / bucket / f"{entry.doc_id}.png"
        if ocr_path.exists():
            ocr = cv2.imread(str(ocr_path), cv2.IMREAD_GRAYSCALE)
            if ocr is not None:
                if ocr.shape != (h, w):
                    ocr = cv2.resize(ocr, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
                return ocr
        # Missing → conservative all-ones.
        return np.full((h, w), 255, dtype=np.uint8)

    def _compute_bbox(
        self,
        entry: SyntheticOCREntry,
        mask_bin: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Return normalised cx/cy/w/h bbox tensor (float32, shape (4,)).

        Priority:
          1. tamper_bboxes from entry (pre-computed by make_splits_synthetic).
          2. If label==1 and tamper_bboxes is empty: derive from mask via CCs.
          3. Otherwise: zeros.
        """
        if entry.tamper_bboxes:
            return _union_bboxes_normalised(entry.tamper_bboxes, h, w)
        if entry.label == 1:
            return _bbox_from_mask(mask_bin)
        return np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        entry_idx = index % len(self.entries)
        tile_idx = index // len(self.entries)
        entry = self.entries[entry_idx]

        # --- Load noisy image ---
        try:
            bgr = cv2.imread(str(entry.noisy_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(
                    f"cv2.imread returned None for {entry.noisy_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            # Graceful skip: return next sample.
            return self.__getitem__((index + 1) % len(self))

        h, w = rgb.shape[:2]

        # --- Load tamper GT mask ---
        if entry.mask_path is not None and entry.mask_path.exists():
            gt_mask = cv2.imread(str(entry.mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                gt_mask = np.zeros((h, w), dtype=np.uint8)
            elif gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        # --- Load OCR mask ---
        ocr_mask = self._load_ocr_mask(entry, h, w)

        # --- Augmentation or deterministic resize ---
        if self.augment:
            rng = random.Random(self.seed + entry_idx * 31 + tile_idx)

            # Smart K-tile sampling (PDF §9): tile 0 forced positive, tile 1
            # forced hard-neg, tiles 2+ stochastic mix. Falls back to random
            # crop when entry has no tamper bboxes.
            tile_mode: str | None = None
            if self.use_tile_sampling and entry.tamper_bboxes:
                if tile_idx == 0:
                    tile_mode = "positive"
                elif tile_idx == 1 and self.tiles_per_image >= 2:
                    tile_mode = "hard_neg"
                else:
                    r = rng.random()
                    if r < 0.50:
                        tile_mode = "positive"
                    elif r < 0.80:
                        tile_mode = "hard_neg"

            if tile_mode is not None:
                # Lockstep crop/rotate/flip via the V2 augment helper —
                # OCR mask rides along as the single float-map extra.
                ocr_f32 = ocr_mask.astype(np.float32)
                rgb, gt_mask, extras_out = _augment_v2_lockstep(
                    rgb,
                    gt_mask,
                    self.crop_sizes,
                    self.full_page_prob,
                    self.target_size,
                    rng,
                    extras=[ocr_f32],
                    tile_mode=tile_mode,
                    bboxes=entry.tamper_bboxes,
                )
                ocr_mask = extras_out[0].astype(np.uint8)
            else:
                rgb, gt_mask, ocr_mask = _augment(rgb, gt_mask, ocr_mask,
                                                  self.target_size, rng)
        else:
            ts = self.target_size
            if (h, w) != (ts, ts):
                rgb = cv2.resize(rgb, (ts, ts),
                                 interpolation=cv2.INTER_LANCZOS4)
                gt_mask = cv2.resize(gt_mask, (ts, ts),
                                     interpolation=cv2.INTER_NEAREST)
                ocr_mask = cv2.resize(ocr_mask, (ts, ts),
                                      interpolation=cv2.INTER_NEAREST)

        # Ensure exact target size (safety net after augmentation).
        ts = self.target_size
        if rgb.shape[0] != ts or rgb.shape[1] != ts:
            rgb = cv2.resize(rgb, (ts, ts), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, (ts, ts),
                                 interpolation=cv2.INTER_NEAREST)
            ocr_mask = cv2.resize(ocr_mask, (ts, ts),
                                  interpolation=cv2.INTER_NEAREST)

        # --- Convert to float arrays ---
        rgb_f = rgb.astype(np.float32) / 255.0  # (H, W, 3)
        ocr_f = ocr_mask.astype(np.float32) / 255.0  # (H, W)
        mask_bin = (gt_mask > 127).astype(np.float32)  # (H, W) binary

        # --- Compute bbox (before tensorisation so we have H, W) ---
        crop_h, crop_w = rgb_f.shape[:2]
        bbox_np = self._compute_bbox(entry, mask_bin, crop_h, crop_w)

        # --- Build 4-channel image tensor: (4, H, W) ---
        rgb_chw = rgb_f.transpose(2, 0, 1)  # (3, H, W)
        ocr_chw = ocr_f[None, :, :]  # (1, H, W)
        image_t = torch.from_numpy(
            np.concatenate([rgb_chw, ocr_chw], axis=0)  # (4, H, W)
        )

        mask_t = torch.from_numpy(mask_bin[None, :, :])  # (1, H, W)
        ocr_t = torch.from_numpy(ocr_chw.astype(np.float32))  # (1, H, W)
        bbox_t = torch.from_numpy(bbox_np)  # (4,)
        label_t = torch.tensor(float(entry.label), dtype=torch.float32)

        out = {
            "image": image_t,  # (4, H, W) float32
            "mask": mask_t,  # (1, H, W) float32
            "bbox": bbox_t,  # (4,)      float32  cx/cy/w/h normalised
            "label": label_t,  # scalar    float32
            "ocr_mask": ocr_t,  # (1, H, W) float32  same as channel-3
        }

        # CI-GT (optional) — resized to target_size, no crop alignment needed
        if self.ci_gt_root is not None:
            bucket = "originals" if entry.label == 0 else "tampered"
            cg_path = self.ci_gt_root / bucket / f"{entry.doc_id}.png"
            cg = cv2.imread(str(cg_path), cv2.IMREAD_GRAYSCALE)
            if cg is not None:
                ts = self.target_size
                if cg.shape != (ts, ts):
                    cg = cv2.resize(cg, (ts, ts),
                                    interpolation=cv2.INTER_LINEAR)
                out["ci_gt"] = torch.from_numpy(cg.astype(np.float32) /
                                                255.0).unsqueeze(0)  # (1,H,W)

        return out


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def build_weighted_sampler(
    entries: list[SyntheticOCREntry],
    positive_frac: float = 0.5,
    tiles_per_image: int = 1,
) -> WeightedRandomSampler:
    """Build a 50/50 (authentic / tampered) WeightedRandomSampler.

    When ``tiles_per_image > 1`` the sampler is expanded to cover the full
    K-tile dataset (length = len(entries) * K) so smart-tile sampling in
    `SyntheticOCRDataset` actually fires across all tile_idx values per
    entry — previously the sampler was capped at len(entries), so the model
    only ever saw tile_idx=0 and tiles 1..K-1 were unreachable.

    Args:
        entries:         List of SyntheticOCREntry for one split.
        positive_frac:   Fraction of draws that should be tampered (default 0.5).
        tiles_per_image: K. Replicates per-entry weights K× and bumps
                         num_samples to len(entries) * K.

    Returns:
        WeightedRandomSampler over len(entries) * K samples with replacement.
    """
    negative_frac = 1.0 - positive_frac
    n_pos = sum(1 for e in entries if e.label == 1)
    n_neg = sum(1 for e in entries if e.label == 0)

    w_pos = positive_frac / max(1, n_pos)
    w_neg = negative_frac / max(1, n_neg)

    per_entry_weights = [w_pos if e.label == 1 else w_neg for e in entries]
    K = max(1, int(tiles_per_image))
    # Replicate per-entry weights for each tile_idx slot. Dataset indexing is
    #   entry_idx = idx % N,  tile_idx = idx // N
    # so tile-0 indices are [0..N), tile-1 indices are [N..2N), etc.
    weights = per_entry_weights * K

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(entries) * K,
        replacement=True,
    )
