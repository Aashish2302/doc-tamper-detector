"""V0.4 — DocForgeNLIDatasetV04 with teacher-aux 4th input channel + GT swap.

Wraps DocForgeNLIDatasetV2's __init__/__len__ but overrides __getitem__ to:

1. Load the teacher's precomputed n_hp_pred PNG from
   `teacher_n_hp_root/{originals|tampered}/{group_id}.png` and concat it as
   channel 3 of the returned image tensor (so output is (4, H, W) instead of
   (3, H, W)).

2. Replace the standard n_hp GT (high-pass residual of the input) with one of:
     gt_mode="diff_hp" → diff_hp_authentic/{group_id}.png for label=0,
                          diff_hp_forgery/{group_id}.png for label=1
     gt_mode="ci_gt"   → ci_gt/originals/{group_id}.png for label=0,
                          ci_gt/tampered/{group_id}.png for label=1

   n_sigma and e_jpeg GTs keep their standard precomputed engine outputs.

Both the GT n_hp swap and the teacher channel pass through `_augment_nli_v2` in
lockstep with the RGB image (same crop/rotate/flip), so all spatial alignment
is preserved.
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
    _compute_e_jpeg,
    _compute_n_hp_proxy,
    _compute_sigma_map,
    _make_boundary,
    load_split_entries_v2,
)


class DocForgeNLIDatasetV04(DocForgeNLIDatasetV2):
    """V0.4 dataset: 4-channel image (RGB + teacher_n_hp), n_hp GT swapped per gt_mode."""

    def __init__(
        self,
        entries: list[DocForgeEntry],
        *,
        augment: bool,
        teacher_n_hp_root: str | Path,
        dataset_root: str | Path,
        gt_mode: str = "diff_hp",
        crop_sizes: tuple[int, ...] = (256, 384, 512),
        target_size: int = 256,
        full_page_prob: float = 0.3,
        seed: int = 42,
        use_tile_sampling: bool = False,
    ) -> None:
        """gt_mode: 'diff_hp' or 'ci_gt' — selects the n_hp GT source."""
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
        self.teacher_n_hp_root = Path(teacher_n_hp_root)
        self.dataset_root = Path(dataset_root)
        if gt_mode not in {"diff_hp", "ci_gt"}:
            raise ValueError(
                f"gt_mode must be 'diff_hp' or 'ci_gt', got {gt_mode!r}")
        self.gt_mode = gt_mode

    def _bucket(self, entry: DocForgeEntry) -> str:
        return "originals" if entry.label == 0 else "tampered"

    def _gt_n_hp_path(self, entry: DocForgeEntry) -> Path:
        if self.gt_mode == "diff_hp":
            sub = "diff_hp_authentic" if entry.label == 0 else "diff_hp_forgery"
            return self.dataset_root / sub / f"{entry.group_id}.png"
        # ci_gt
        return self.dataset_root / "ci_gt" / self._bucket(
            entry) / f"{entry.group_id}.png"

    def _teacher_n_hp_path(self, entry: DocForgeEntry) -> Path:
        return self.teacher_n_hp_root / self._bucket(
            entry) / f"{entry.group_id}.png"

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]

        # Load image
        try:
            bgr = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                return self.__getitem__((index + 1) % len(self))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        # Load mask
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

        # Resolve and load V0.4-specific maps:
        #   index 0: n_hp GT (diff_hp or ci_gt, per gt_mode)
        #   index 1: n_sigma GT (engine precomputed, fall back to compute)
        #   index 2: e_jpeg GT (engine precomputed, fall back to compute)
        #   index 3: teacher_n_hp (precomputed teacher output)
        gt_p = self._gt_n_hp_path(entry)
        teacher_p = self._teacher_n_hp_path(entry)
        if not gt_p.exists() or not teacher_p.exists():
            # Skip this sample — V0.4 requires both
            return self.__getitem__((index + 1) % len(self))

        gt_n_hp_u = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)
        teacher_n_hp_u = cv2.imread(str(teacher_p), cv2.IMREAD_GRAYSCALE)
        if gt_n_hp_u is None or teacher_n_hp_u is None:
            return self.__getitem__((index + 1) % len(self))
        if gt_n_hp_u.shape != (H, W):
            gt_n_hp_u = cv2.resize(gt_n_hp_u, (W, H),
                                   interpolation=cv2.INTER_LINEAR)
        if teacher_n_hp_u.shape != (H, W):
            teacher_n_hp_u = cv2.resize(teacher_n_hp_u, (W, H),
                                        interpolation=cv2.INTER_LINEAR)

        # n_sigma / e_jpeg GTs from engine when present
        if (entry.n_sigma_path is not None and entry.n_sigma_path.exists()
                and entry.e_jpeg_path is not None
                and entry.e_jpeg_path.exists()):
            n_sigma_u = cv2.imread(str(entry.n_sigma_path),
                                   cv2.IMREAD_GRAYSCALE)
            e_jpeg_u = cv2.imread(str(entry.e_jpeg_path), cv2.IMREAD_GRAYSCALE)
            if n_sigma_u is not None and n_sigma_u.shape != (H, W):
                n_sigma_u = cv2.resize(n_sigma_u, (W, H),
                                       interpolation=cv2.INTER_LINEAR)
            if e_jpeg_u is not None and e_jpeg_u.shape != (H, W):
                e_jpeg_u = cv2.resize(e_jpeg_u, (W, H),
                                      interpolation=cv2.INTER_LINEAR)
        else:
            n_sigma_u = None
            e_jpeg_u = None

        extras_in: list[np.ndarray] = [
            gt_n_hp_u.astype(np.float32) / 255.0,
            (n_sigma_u.astype(np.float32) /
             255.0 if n_sigma_u is not None else np.zeros(
                 (H, W), dtype=np.float32)),
            (e_jpeg_u.astype(np.float32) /
             255.0 if e_jpeg_u is not None else np.zeros(
                 (H, W), dtype=np.float32)),
            teacher_n_hp_u.astype(np.float32) / 255.0,
        ]
        n_sigma_present = n_sigma_u is not None
        e_jpeg_present = e_jpeg_u is not None

        # Augment / center-crop — extras carry through structurally
        if self.augment:
            rng = random.Random(self.seed + index)
            tile_mode: str | None = None
            if self.use_tile_sampling and entry.tamper_bboxes:
                r = rng.random()
                if r < 0.50:
                    tile_mode = "positive"
                elif r < 0.80:
                    tile_mode = "hard_neg"
            rgb, mask, extras_out = _augment_nli_v2(
                rgb,
                mask,
                self.crop_sizes,
                self.full_page_prob,
                self.target_size,
                rng,
                extras=extras_in,
                tile_mode=tile_mode,
                bboxes=entry.tamper_bboxes if tile_mode else None,
            )
        else:
            rgb = cv2.resize(rgb, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (self.target_size, self.target_size),
                              interpolation=cv2.INTER_NEAREST)
            extras_out = [
                cv2.resize(e, (self.target_size, self.target_size),
                           interpolation=cv2.INTER_LINEAR) for e in extras_in
            ]

        # Resize fallback
        if rgb.shape[0] != self.target_size or rgb.shape[1] != self.target_size:
            rgb = cv2.resize(rgb, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_size, self.target_size),
                              interpolation=cv2.INTER_NEAREST)
            if extras_out is not None:
                extras_out = [
                    cv2.resize(e, (self.target_size, self.target_size),
                               interpolation=cv2.INTER_LINEAR)
                    for e in extras_out
                ]

        # Boundary target
        bnd_radius = 2 if self.target_size <= 384 else 3
        boundary = _make_boundary(mask, radius=bnd_radius)

        # Unpack extras
        n_hp = extras_out[0]
        n_sigma = extras_out[1] if n_sigma_present else _compute_sigma_map(
            n_hp, block_size=16)
        e_jpeg = extras_out[2] if e_jpeg_present else _compute_e_jpeg(rgb)
        teacher_n_hp = extras_out[3]

        # To tensors — RGB is 3ch, then concat teacher as 4th channel
        rgb_t = torch.from_numpy(
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)  # (3, H, W)
        teacher_t = torch.from_numpy(teacher_n_hp[None, :, :].astype(
            np.float32))  # (1, H, W)
        image_t = torch.cat([rgb_t, teacher_t], dim=0)  # (4, H, W)

        mask_t = torch.from_numpy((mask > 127).astype(np.float32)[None, :, :])
        boundary_t = torch.from_numpy((boundary
                                       > 127).astype(np.float32)[None, :, :])
        label_t = torch.tensor(float(entry.label), dtype=torch.float32)
        n_hp_t = torch.from_numpy(n_hp[None, :, :].copy())
        n_sigma_t = torch.from_numpy(n_sigma[None, :, :].copy())
        e_jpeg_t = torch.from_numpy(e_jpeg[None, :, :].copy())
        alpha_t = torch.tensor(entry.alpha, dtype=torch.float32)

        return {
            "image": image_t,
            "mask": mask_t,
            "boundary": boundary_t,
            "label": label_t,
            "description": f"{entry.description}:{entry.image_path.name}",
            "n_hp_gt": n_hp_t,
            "n_sigma_gt": n_sigma_t,
            "e_jpeg_gt": e_jpeg_t,
            "alpha": alpha_t,
        }


def load_split_entries_v04(dataset_root: str | Path,
                           split_name: str) -> list[DocForgeEntry]:
    """Pass-through wrapper around the V2 split loader for naming consistency."""
    return load_split_entries_v2(dataset_root, split_name)
