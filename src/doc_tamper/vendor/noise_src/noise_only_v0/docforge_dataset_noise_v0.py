"""Paired (noisy, clean) dataset for noise_V0 training.

Reads splits in the format produced by `make_splits_synthetic.py` (which
walks outputs_100k/originals + outputs_100k/tampered and pairs each
noisy image with its corresponding clean PNG under outputs_100k/clean/).

Stage 1 returns: image (noisy), clean, label (0/1).
Stage 2 returns: image (noisy), mask, bbox, label.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NoiseV0Entry:
    doc_id: str
    label: int  # 0 = authentic, 1 = tampered
    noisy_path: Path
    clean_path: Path
    mask_path: Path | None  # None for authentic
    tamper_bboxes: list[list[int]]  # pixel bboxes (only for tampered)


def load_split_entries(splits_root: Path,
                       split_name: str) -> list[NoiseV0Entry]:
    raw = json.loads((splits_root / f"{split_name}.json").read_text())
    out: list[NoiseV0Entry] = []
    for e in raw:
        out.append(
            NoiseV0Entry(
                doc_id=e["doc_id"],
                label=int(e["label"]),
                noisy_path=Path(e["noisy_path"]),
                clean_path=Path(e["clean_path"]),
                mask_path=Path(e["mask_path"]) if e.get("mask_path") else None,
                tamper_bboxes=e.get("tamper_bboxes", []),
            ))
    return out


def _random_crop(rgb: np.ndarray, clean: np.ndarray, mask: np.ndarray,
                 size: int, rng: random.Random) -> tuple:
    h, w = rgb.shape[:2]
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        clean = cv2.copyMakeBorder(clean, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask,
                                  0,
                                  ph,
                                  0,
                                  pw,
                                  cv2.BORDER_CONSTANT,
                                  value=0)
        h, w = rgb.shape[:2]
    y = rng.randint(0, h - size)
    x = rng.randint(0, w - size)
    return (rgb[y:y + size, x:x + size], clean[y:y + size,
                                               x:x + size], mask[y:y + size,
                                                                 x:x + size])


def _tile_is_blank(clean_crop: np.ndarray, white_thresh: int = 245,
                   white_frac: float = 0.90) -> bool:
    """True if the crop is mostly blank/white document background.

    A pixel counts as 'white' when all channels exceed `white_thresh`; the
    tile is blank when the white fraction exceeds `white_frac`. Cheap enough
    to run per candidate crop.
    """
    white = (clean_crop.min(axis=2) > white_thresh)
    return float(white.mean()) > white_frac


def _content_random_crop(rgb: np.ndarray, clean: np.ndarray, mask: np.ndarray,
                         size: int, rng: random.Random,
                         reject_prob: float = 0.0, max_tries: int = 6) -> tuple:
    """Random crop with epoch-scheduled rejection of blank tiles.

    With probability `reject_prob`, a candidate crop that `_tile_is_blank`
    reports as mostly-white is resampled (up to `max_tries`). reject_prob is
    ramped from a base value down to 0 over the warmup epochs by the dataset,
    so early training focuses on text/content tiles and late training sees the
    full distribution (blanks included) — no test-time skew.
    """
    crop = _random_crop(rgb, clean, mask, size, rng)
    if reject_prob <= 0.0:
        return crop
    for _ in range(max_tries):
        if not _tile_is_blank(crop[1]):
            return crop
        if rng.random() >= reject_prob:
            return crop
        crop = _random_crop(rgb, clean, mask, size, rng)
    return crop


def _positive_crop(rgb: np.ndarray, clean: np.ndarray, mask: np.ndarray,
                   size: int, bboxes: list, rng: random.Random) -> tuple:
    if not bboxes:
        return _random_crop(rgb, clean, mask, size, rng)
    h, w = rgb.shape[:2]
    bx0, by0, bx1, by1 = bboxes[rng.randint(0, len(bboxes) - 1)]
    cx = (bx0 + bx1) // 2
    cy = (by0 + by1) // 2
    jitter = size // 4
    cx += rng.randint(-jitter, jitter)
    cy += rng.randint(-jitter, jitter)
    x = max(0, min(w - size, cx - size // 2))
    y = max(0, min(h - size, cy - size // 2))
    return (rgb[y:y + size, x:x + size], clean[y:y + size,
                                               x:x + size], mask[y:y + size,
                                                                 x:x + size])


def _hard_neg_crop(rgb: np.ndarray,
                   clean: np.ndarray,
                   mask: np.ndarray,
                   size: int,
                   bboxes: list,
                   rng: random.Random,
                   max_attempts: int = 8) -> tuple:
    """Rejection-sampled crop that AVOIDS every tamper bbox.

    Falls back to `_random_crop` after `max_attempts` or when no bboxes
    exist. Used by Stage 2's tile_idx=1 slot for forced hard-negatives:
    same document, similar texture, but no tamper signal — pushes the
    tamper-vs-not boundary harder than easy negatives from authentic docs.
    """
    if not bboxes:
        return _random_crop(rgb, clean, mask, size, rng)
    h, w = rgb.shape[:2]
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        clean = cv2.copyMakeBorder(clean, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask,
                                  0,
                                  ph,
                                  0,
                                  pw,
                                  cv2.BORDER_CONSTANT,
                                  value=0)
        h, w = rgb.shape[:2]
    for _ in range(max_attempts):
        y = rng.randint(0, h - size)
        x = rng.randint(0, w - size)
        overlaps = any(
            x < bx1 and x + size > bx0 and y < by1 and y + size > by0
            for bx0, by0, bx1, by1 in bboxes)
        if not overlaps:
            return (rgb[y:y + size,
                        x:x + size], clean[y:y + size,
                                           x:x + size], mask[y:y + size,
                                                             x:x + size])
    # Couldn't find a clean spot — fall back to random.
    return _random_crop(rgb, clean, mask, size, rng)


class NoiseV0Dataset(Dataset):
    """Returns (noisy, clean, mask, bbox, label) for noise_V0 training.

    For Stage 1, only noisy + clean are used (mask/bbox/label ignored).
    For Stage 2, noisy + mask + bbox + label are used (clean ignored).
    Both stages share this dataset to avoid duplication.

    ci_gt_root: optional path to outputs_100k/ci_gt/. When set, each batch
    item includes a "ci_gt" tensor (B,1,H,W) resized to target_size.
    """

    def __init__(
        self,
        entries: list[NoiseV0Entry],
        target_size: int = 384,
        augment: bool = True,
        seed: int = 42,
        positive_crop_prob: float = 0.5,
        tiles_per_image: int = 1,
        ci_gt_root: Path | None = None,
        use_tile_sampling: bool = False,
    ) -> None:
        """
        use_tile_sampling:
            Stage-2 K-tile mode mix (PDF §9). When True AND entry has
            tamper_bboxes:
              tile 0 = forced POSITIVE crop (overlaps a tamper bbox)
              tile 1 = forced HARD-NEG crop (rejection-sampled to avoid bboxes)
              tiles 2+ : 50% positive / 30% hard-neg / 20% random
            When False (Stage-1 default): flat coin-flip — `positive_crop_prob`
            chance of positive crop per tile, else random crop. Stage 1 is a
            denoiser and doesn't benefit from forced tamper-side coverage.
        """
        self.entries = entries
        self.target_size = target_size
        self.augment = augment
        self.seed = seed
        self.positive_crop_prob = positive_crop_prob
        self.tiles_per_image = max(1, int(tiles_per_image))
        self.ci_gt_root = ci_gt_root
        self.use_tile_sampling = use_tile_sampling
        # Blank-tile curriculum (set via set_epoch each epoch). base>0 enables.
        self.blank_reject_base = 0.0
        self.blank_reject_warmup_frac = 0.30
        self.blank_reject_prob = 0.0

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Update the blank-tile rejection probability for this epoch.

        Linearly decays from `blank_reject_base` at epoch 1 to 0 at
        `blank_reject_warmup_frac * total_epochs`, then stays 0. Called by the
        trainer before each epoch; with non-persistent workers the updated
        value is picked up when workers re-fork.
        """
        if self.blank_reject_base <= 0.0:
            self.blank_reject_prob = 0.0
            return
        warm = max(1.0, self.blank_reject_warmup_frac * float(total_epochs))
        frac = max(0.0, 1.0 - (float(epoch) - 1.0) / warm)
        self.blank_reject_prob = self.blank_reject_base * frac

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, entry: NoiseV0Entry) -> tuple:
        bgr = cv2.imread(str(entry.noisy_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read noisy: {entry.noisy_path}")
        noisy = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        bgr_c = cv2.imread(str(entry.clean_path), cv2.IMREAD_COLOR)
        if bgr_c is None:
            raise RuntimeError(f"failed to read clean: {entry.clean_path}")
        clean = cv2.cvtColor(bgr_c, cv2.COLOR_BGR2RGB)

        if clean.shape[:2] != noisy.shape[:2]:
            clean = cv2.resize(clean, (noisy.shape[1], noisy.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        if entry.mask_path is not None and entry.mask_path.exists():
            mask = cv2.imread(str(entry.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(noisy.shape[:2], dtype=np.uint8)
            elif mask.shape != noisy.shape[:2]:
                mask = cv2.resize(mask, (noisy.shape[1], noisy.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(noisy.shape[:2], dtype=np.uint8)

        return noisy, clean, mask

    def _pick_tile_mode(self, tile_idx: int, has_bboxes: bool,
                        rng: random.Random) -> str:
        """Return one of {'positive', 'hard_neg', 'random'} for tile_idx.

        Smart mix (use_tile_sampling=True, has_bboxes=True):
          tile 0 = forced positive
          tile 1 = forced hard-neg (if K >= 2)
          tiles 2+ : 50% pos / 30% hard-neg / 20% random
        Default coin-flip (use_tile_sampling=False):
          `positive_crop_prob` chance positive (only if has_bboxes), else random.
        """
        if not self.use_tile_sampling or not has_bboxes:
            if has_bboxes and rng.random() < self.positive_crop_prob:
                return "positive"
            return "random"
        if tile_idx == 0:
            return "positive"
        if tile_idx == 1 and self.tiles_per_image >= 2:
            return "hard_neg"
        r = rng.random()
        if r < 0.50:
            return "positive"
        if r < 0.80:
            return "hard_neg"
        return "random"

    def _process_tile(self,
                      noisy,
                      clean,
                      mask,
                      entry,
                      rng,
                      tile_idx: int = 0) -> tuple:
        if self.augment:
            mode = self._pick_tile_mode(tile_idx, bool(entry.tamper_bboxes),
                                        rng)
            if mode == "positive":
                noisy_c, clean_c, mask_c = _positive_crop(
                    noisy, clean, mask, self.target_size, entry.tamper_bboxes,
                    rng)
            elif mode == "hard_neg":
                noisy_c, clean_c, mask_c = _hard_neg_crop(
                    noisy, clean, mask, self.target_size, entry.tamper_bboxes,
                    rng)
            else:
                noisy_c, clean_c, mask_c = _content_random_crop(
                    noisy, clean, mask, self.target_size, rng,
                    reject_prob=self.blank_reject_prob)

            # Random horizontal flip
            if rng.random() < 0.5:
                noisy_c = noisy_c[:, ::-1].copy()
                clean_c = clean_c[:, ::-1].copy()
                mask_c = mask_c[:, ::-1].copy()
        else:
            ts = self.target_size
            noisy_c = cv2.resize(noisy, (ts, ts),
                                 interpolation=cv2.INTER_LANCZOS4)
            clean_c = cv2.resize(clean, (ts, ts),
                                 interpolation=cv2.INTER_LANCZOS4)
            mask_c = cv2.resize(mask, (ts, ts),
                                interpolation=cv2.INTER_NEAREST)
        return noisy_c, clean_c, mask_c

    @staticmethod
    def _to_tensors(noisy_c, clean_c, mask_c) -> tuple:
        noisy_t = torch.from_numpy(
            noisy_c.transpose(2, 0, 1).astype(np.float32) / 255.0)
        clean_t = torch.from_numpy(
            clean_c.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask_bin = (mask_c > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_bin[None, :, :])
        bbox = NoiseV0Dataset._mask_to_bbox(mask_bin)
        bbox_t = torch.from_numpy(bbox)
        return noisy_t, clean_t, mask_t, bbox_t

    @staticmethod
    def _mask_to_bbox(mask01: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask01 > 0.5)
        if ys.size == 0:
            return np.zeros(4, dtype=np.float32)
        h, w = mask01.shape
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return np.array([
            ((x0 + x1) / 2.0) / w,
            ((y0 + y1) / 2.0) / h,
            (x1 - x0 + 1) / w,
            (y1 - y0 + 1) / h,
        ],
                        dtype=np.float32)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        try:
            noisy, clean, mask = self._load(entry)
        except RuntimeError:
            return self.__getitem__((index + 1) % len(self))

        K = self.tiles_per_image if self.augment else 1
        noisy_list, clean_list, mask_list, bbox_list = [], [], [], []
        for k in range(K):
            rng = random.Random(self.seed + index * 31 + k)
            noisy_c, clean_c, mask_c = self._process_tile(noisy,
                                                          clean,
                                                          mask,
                                                          entry,
                                                          rng,
                                                          tile_idx=k)
            noisy_t, clean_t, mask_t, bbox_t = self._to_tensors(
                noisy_c, clean_c, mask_c)
            noisy_list.append(noisy_t)
            clean_list.append(clean_t)
            mask_list.append(mask_t)
            bbox_list.append(bbox_t)

        # Load CI-GT (once per image, same map reused for all K tiles)
        ci_gt_t = None
        if self.ci_gt_root is not None:
            bucket = "tampered" if entry.label == 1 else "originals"
            ci_gt_path = self.ci_gt_root / bucket / f"{entry.doc_id}.png"
            cg = cv2.imread(str(ci_gt_path), cv2.IMREAD_GRAYSCALE)
            if cg is not None:
                ts = self.target_size
                if cg.shape != (ts, ts):
                    cg = cv2.resize(cg, (ts, ts),
                                    interpolation=cv2.INTER_LINEAR)
                ci_gt_t = torch.from_numpy(cg.astype(np.float32) /
                                           255.0).unsqueeze(0)  # (1,H,W)

        if K == 1:
            out = {
                "image": noisy_list[0],
                "clean": clean_list[0],
                "mask": mask_list[0],
                "bbox": bbox_list[0],
                "label": torch.tensor(float(entry.label), dtype=torch.float32),
            }
            if ci_gt_t is not None:
                out["ci_gt"] = ci_gt_t
            return out

        out = {
            "image": torch.stack(noisy_list, 0),
            "clean": torch.stack(clean_list, 0),
            "mask": torch.stack(mask_list, 0),
            "bbox": torch.stack(bbox_list, 0),
            "label": torch.full((K, ), float(entry.label),
                                dtype=torch.float32),
            "_k_stacked": True,
        }
        if ci_gt_t is not None:
            # repeat same CI-GT for all K tiles
            out["ci_gt"] = ci_gt_t.unsqueeze(0).expand(K, -1, -1, -1)
        return out


def build_weighted_sampler(entries: list[NoiseV0Entry],
                           positive_frac: float = 0.5,
                           tiles_per_image: int = 1):
    """50/50 sampler for tampered vs authentic — used by Stage 2.

    Note: unlike `SyntheticOCRDataset` (V0.5-OCR), `NoiseV0Dataset.__len__`
    returns N (not N*K). The K tile expansion happens INSIDE __getitem__
    by stacking K tiles into one item, then `_flatten_k_collate` unrolls
    them in the DataLoader. So this sampler must keep num_samples = N;
    every drawn index already produces K training examples.

    The `tiles_per_image` arg is kept for API parity with V0.5-OCR but
    intentionally does not multiply num_samples here.
    """
    from torch.utils.data import WeightedRandomSampler
    n_pos = sum(1 for e in entries if e.label == 1)
    n_neg = len(entries) - n_pos
    w_pos = positive_frac / max(1, n_pos)
    w_neg = (1 - positive_frac) / max(1, n_neg)
    weights = [w_pos if e.label == 1 else w_neg for e in entries]
    return WeightedRandomSampler(weights=weights,
                                 num_samples=len(entries),
                                 replacement=True)
