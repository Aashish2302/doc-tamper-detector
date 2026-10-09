"""V0.5 dataset — subclass of DocForgeNLIDatasetV2.

Returns:
  image: (3, H, W) float32 in [0, 1]
  ci_gt: (1, H, W) float32 in [0, 1]   — content-invariant noise GT
  mask:  (1, H, W) float32 in {0, 1}
  bbox:  (4,) float32 in [0, 1]        — (cx, cy, w, h), zeros if authentic
  label: scalar float32 in {0, 1}
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch

from noise_level_inconsistency.docforge_dataset_v2 import (
    DocForgeEntry,
    DocForgeNLIDatasetV2,
    _augment_nli_v2,
    load_split_entries_v2,
)


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


class DocForgeNLIDatasetV05(DocForgeNLIDatasetV2):

    def __init__(
        self,
        entries: list[DocForgeEntry],
        *,
        augment: bool,
        dataset_root: str | Path,
        ci_gt_root: str | Path | None = None,
        layout: str = "outputs_30k",
        crop_sizes: tuple[int, ...] = (256, 384, 512),
        target_size: int = 256,
        full_page_prob: float = 0.3,
        seed: int = 42,
        use_tile_sampling: bool = False,
        tiles_per_image: int = 1,
    ) -> None:
        super().__init__(
            entries,
            augment=augment,
            crop_sizes=crop_sizes,
            target_size=target_size,
            full_page_prob=full_page_prob,
            seed=seed,
            prefer_precomputed_gt=False,
            use_tile_sampling=use_tile_sampling,
            n_hp_override_dir=None,
        )
        self.dataset_root = Path(dataset_root)
        self.ci_gt_root = Path(ci_gt_root) if ci_gt_root else (
            self.dataset_root / "ci_gt")
        self.layout = layout
        self.tiles_per_image = max(1, int(tiles_per_image))

    def _ci_gt_path(self, entry: DocForgeEntry) -> Path:
        bucket = "originals" if entry.label == 0 else "tampered"
        return self.ci_gt_root / bucket / f"{entry.image_path.stem}.png"

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]

        try:
            bgr = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                return self.__getitem__((index + 1) % len(self))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        if entry.mask_path is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        else:
            mask = cv2.imread(str(entry.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            elif mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        H, W = rgb.shape[:2]

        ci_gt_p = self._ci_gt_path(entry)
        if not ci_gt_p.exists():
            return self.__getitem__((index + 1) % len(self))
        ci_gt_u = cv2.imread(str(ci_gt_p), cv2.IMREAD_GRAYSCALE)
        if ci_gt_u is None:
            return self.__getitem__((index + 1) % len(self))
        if ci_gt_u.shape != (H, W):
            ci_gt_u = cv2.resize(ci_gt_u, (W, H),
                                 interpolation=cv2.INTER_LINEAR)
        extras_in: list[np.ndarray] = [ci_gt_u.astype(np.float32) / 255.0]

        K = self.tiles_per_image if self.augment else 1
        rgb_list, ci_list, mask_list, bbox_list = [], [], [], []
        for k in range(K):
            ci_u_k = extras_in[0].copy()
            if self.augment:
                rng_k = random.Random(self.seed + index * 31 + k)
                tile_mode: str | None = None
                if self.use_tile_sampling and entry.tamper_bboxes:
                    if k == 0:
                        tile_mode = "positive"
                    elif k == 1 and K >= 2:
                        tile_mode = "hard_neg"
                    else:
                        r = rng_k.random()
                        if r < 0.50:
                            tile_mode = "positive"
                        elif r < 0.80:
                            tile_mode = "hard_neg"
                rgb_k, mask_k, extras_out_k = _augment_nli_v2(
                    rgb.copy(),
                    mask.copy(),
                    self.crop_sizes,
                    self.full_page_prob,
                    self.target_size,
                    rng_k,
                    extras=[ci_u_k],
                    tile_mode=tile_mode,
                    bboxes=entry.tamper_bboxes if tile_mode else None,
                )
            else:
                rgb_k = cv2.resize(rgb, (self.target_size, self.target_size),
                                   interpolation=cv2.INTER_LANCZOS4)
                mask_k = cv2.resize(mask, (self.target_size, self.target_size),
                                    interpolation=cv2.INTER_NEAREST)
                extras_out_k = [
                    cv2.resize(ci_u_k, (self.target_size, self.target_size),
                               interpolation=cv2.INTER_LINEAR)
                ]

            if (rgb_k.shape[0] != self.target_size
                    or rgb_k.shape[1] != self.target_size):
                rgb_k = cv2.resize(rgb_k, (self.target_size, self.target_size),
                                   interpolation=cv2.INTER_LINEAR)
                mask_k = cv2.resize(mask_k,
                                    (self.target_size, self.target_size),
                                    interpolation=cv2.INTER_NEAREST)
                extras_out_k = [
                    cv2.resize(e, (self.target_size, self.target_size),
                               interpolation=cv2.INTER_LINEAR)
                    for e in extras_out_k
                ]

            mask_bin_k = (mask_k > 127).astype(np.float32)
            rgb_list.append(
                rgb_k.transpose(2, 0, 1).astype(np.float32) / 255.0)
            ci_list.append(extras_out_k[0][None, :, :].astype(np.float32))
            mask_list.append(mask_bin_k[None, :, :])
            bbox_list.append(_mask_to_bbox(mask_bin_k))

        rgb_t = torch.from_numpy(np.stack(rgb_list, axis=0))
        ci_gt_t = torch.from_numpy(np.stack(ci_list, axis=0))
        mask_t = torch.from_numpy(np.stack(mask_list, axis=0))
        bbox_t = torch.from_numpy(np.stack(bbox_list, axis=0))
        label_t = torch.full((K, ), float(entry.label), dtype=torch.float32)

        if K == 1:
            return {
                "image": rgb_t[0],
                "ci_gt": ci_gt_t[0],
                "mask": mask_t[0],
                "bbox": bbox_t[0],
                "label": label_t[0],
                "description": f"{entry.description}:{entry.image_path.name}",
            }
        return {
            "image": rgb_t,
            "ci_gt": ci_gt_t,
            "mask": mask_t,
            "bbox": bbox_t,
            "label": label_t,
            "description": f"{entry.description}:{entry.image_path.name}",
            "_k_stacked": True,
        }


def load_split_entries_v05(dataset_root: str | Path,
                           split_name: str) -> list[DocForgeEntry]:
    return load_split_entries_v2(dataset_root, split_name)
