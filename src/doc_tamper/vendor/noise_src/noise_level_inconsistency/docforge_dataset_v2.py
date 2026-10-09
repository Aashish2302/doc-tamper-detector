"""Dataset loader v2 for 100k doc_forge_engine outputs.

Changes from v1 (docforge_dataset.py):
  - Handles outputs_30k/splits/ format (100k entries, some may lack tampered)
  - Computes boundary target (dilate XOR erode) for the new boundary head
  - Supports full-page downscale mode (30% of training samples)
  - Lighter augmentation to preserve noise signal (NO JPEG roundtrip)
  - Falls back gracefully on missing/corrupt entries
  - [ablation] On-the-fly GT proxies: N_hp, N_sigma, E_jpeg
  - [ablation] Alpha-weighted anomaly target (base_clean → alpha=0.4 or 1.0)
  - [ablation] 50/30/20 WeightedRandomSampler (positive/hard-neg/easy-neg)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Authentic appearance profiles treated as hard negatives
_HARD_NEG_PROFILES = frozenset({
    "jpeg_q80",
    "heavy_jpeg",
    "double_jpeg",
    "scan_like",
    "mobile_capture",
    "print_scan",
})


@dataclass(slots=True)
class DocForgeEntry:
    group_id: str
    image_path: Path
    mask_path: Path | None
    label: int
    description: str
    base_clean: bool = False
    alpha: float = 1.0
    appearance_profile_name: str = "clean"
    is_hard_neg: bool = False
    # Precomputed GT-side noise profiles (engine-emitted, optional).
    # Present iff the manifest carries a `noise_profile` block.
    n_hp_path: Path | None = None
    n_sigma_path: Path | None = None
    e_jpeg_path: Path | None = None
    # PDF §3.3 difficulty flags (tampered entries only)
    forensic_equalized: bool = False
    # PDF §9 within-doc tile sampling: tamper region bboxes [x0, y0, x1, y1]
    tamper_bboxes: list = dc_field(default_factory=list)


def _resolve_path(raw: str, root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for base in [root.parent, root]:
        c = base / p
        if c.exists():
            return c
    return p


def load_split_entries_v2(dataset_root: str | Path,
                          split_name: str) -> list[DocForgeEntry]:
    """Load entries from 100k split files.  Handles missing tampered gracefully."""
    root = Path(dataset_root)
    split_path = root / "splits" / f"{split_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    pairs = json.loads(split_path.read_text(encoding="utf-8"))
    entries: list[DocForgeEntry] = []

    def _resolve_profile_paths(
            section: dict) -> tuple[Path | None, Path | None, Path | None]:
        np_block = section.get("noise_profile") or {}
        n_hp = np_block.get("n_hp_path")
        n_sigma = np_block.get("n_sigma_path")
        e_jpeg = np_block.get("e_jpeg_path")
        return (
            _resolve_path(n_hp, root) if n_hp else None,
            _resolve_path(n_sigma, root) if n_sigma else None,
            _resolve_path(e_jpeg, root) if e_jpeg else None,
        )

    for pair in pairs:
        orig = pair.get("original")
        tamp = pair.get("tampered")

        if orig:
            profile_name = (orig.get("appearance_profile")
                            or {}).get("name", "clean")
            is_hn = profile_name in _HARD_NEG_PROFILES
            n_hp_p, n_sigma_p, e_jpeg_p = _resolve_profile_paths(orig)
            entries.append(
                DocForgeEntry(
                    group_id=orig["doc_id"],
                    image_path=_resolve_path(orig["image_path"], root),
                    mask_path=None,
                    label=0,
                    description="authentic",
                    base_clean=False,
                    alpha=0.0,
                    appearance_profile_name=profile_name,
                    is_hard_neg=is_hn,
                    n_hp_path=n_hp_p,
                    n_sigma_path=n_sigma_p,
                    e_jpeg_path=e_jpeg_p,
                ))

        if tamp and tamp.get("image_path") and tamp.get("mask_path"):
            bc = bool(tamp.get("base_clean", False))
            fe = bool(tamp.get("forensic_equalized", False))
            profile_name = (tamp.get("appearance_profile")
                            or {}).get("name", "clean")
            alpha = 0.4 if bc else 1.0
            n_hp_p, n_sigma_p, e_jpeg_p = _resolve_profile_paths(tamp)
            bboxes = [
                r["bbox"] for r in tamp.get("tamper_regions", [])
                if "bbox" in r
            ]
            entries.append(
                DocForgeEntry(
                    group_id=tamp.get("parent_doc_id",
                                      orig["doc_id"] if orig else "unknown"),
                    image_path=_resolve_path(tamp["image_path"], root),
                    mask_path=_resolve_path(tamp["mask_path"], root),
                    label=1,
                    description="tampered",
                    base_clean=bc,
                    alpha=alpha,
                    appearance_profile_name=profile_name,
                    is_hard_neg=False,
                    n_hp_path=n_hp_p,
                    n_sigma_path=n_sigma_p,
                    e_jpeg_path=e_jpeg_p,
                    forensic_equalized=fe,
                    tamper_bboxes=bboxes,
                ))

    return entries


def build_nli_weighted_sampler(
    entries: list[DocForgeEntry],
    positive_frac: float = 0.50,
    hard_neg_frac: float = 0.30,
    random_neg_frac: float = 0.20,
) -> WeightedRandomSampler:
    """Build a 50/30/20 WeightedRandomSampler (positive/hard-neg/easy-neg)."""
    n_pos = sum(1 for e in entries if e.label == 1)
    n_hn = sum(1 for e in entries if e.label == 0 and e.is_hard_neg)
    n_rn = sum(1 for e in entries if e.label == 0 and not e.is_hard_neg)

    w_pos = positive_frac / max(1, n_pos)
    w_hn = hard_neg_frac / max(1, n_hn)
    w_rn = random_neg_frac / max(1, n_rn)

    weights = []
    for e in entries:
        if e.label == 1:
            weights.append(w_pos)
        elif e.is_hard_neg:
            weights.append(w_hn)
        else:
            weights.append(w_rn)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(entries),
        replacement=True,
    )


def build_difficulty_curriculum_sampler(
    entries: list[DocForgeEntry],
    easy_frac: float = 0.40,
    hard_frac: float = 0.40,
    auth_frac: float = 0.20,
) -> WeightedRandomSampler:
    """PDF §3.3 attack-difficulty curriculum: 40% easy / 40% hard / 20% authentic.

    Easy tampered: base_clean=False (no forensic laundering).
    Hard tampered: base_clean=True (forensic laundering applied).
    Authentic: label=0 (all authentic entries weighted equally).
    """
    n_easy = sum(1 for e in entries if e.label == 1 and not e.base_clean)
    n_hard = sum(1 for e in entries if e.label == 1 and e.base_clean)
    n_auth = sum(1 for e in entries if e.label == 0)

    w_easy = easy_frac / max(1, n_easy)
    w_hard = hard_frac / max(1, n_hard)
    w_auth = auth_frac / max(1, n_auth)

    weights = []
    for e in entries:
        if e.label == 1:
            weights.append(w_easy if not e.base_clean else w_hard)
        else:
            weights.append(w_auth)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(entries),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# On-the-fly GT proxy computations (no host_clean image needed)
# ---------------------------------------------------------------------------


def _compute_n_hp_proxy(rgb_uint8: np.ndarray) -> np.ndarray:
    """High-pass noise residual proxy: |I_gray - GaussianBlur(I_gray, σ=2)|.

    Returns float32 (H, W) in [0, 1].
    """
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(
        np.float32) / 255.0
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return np.abs(gray - blurred)  # already in [0, ~0.5], clipped below


def _compute_sigma_map(n_hp: np.ndarray, block_size: int = 16) -> np.ndarray:
    """MAD-based local noise sigma map from high-pass residual.

    σ_i = 1.4826 * MAD(block) for each block_size×block_size tile.
    Returns float32 (H, W) in [0, 1] — normalized by dividing by 0.05.
    """
    H, W = n_hp.shape
    # Pad to multiple of block_size
    Hp = ((H + block_size - 1) // block_size) * block_size
    Wp = ((W + block_size - 1) // block_size) * block_size
    padded = cv2.copyMakeBorder(n_hp, 0, Hp - H, 0, Wp - W,
                                cv2.BORDER_REFLECT_101)

    # Reshape to (num_blocks_h, block_size, num_blocks_w, block_size)
    blocks = padded.reshape(Hp // block_size, block_size, Wp // block_size,
                            block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size * block_size)

    # MAD per block → sigma estimate
    medians = np.median(blocks, axis=1, keepdims=True)
    mad = np.median(np.abs(blocks - medians), axis=1)
    sigma = 1.4826 * mad  # (num_blocks,)

    # Reshape back to block grid and upsample to (H, W)
    nBh = Hp // block_size
    nBw = Wp // block_size
    sigma_grid = sigma.reshape(nBh, nBw).astype(np.float32)
    sigma_map = cv2.resize(sigma_grid, (Wp, Hp),
                           interpolation=cv2.INTER_NEAREST)
    sigma_map = sigma_map[:H, :W]

    # Normalize: σ=0.05 in n_hp units ≈ strong noise; clip to [0, 1]
    return np.clip(sigma_map / 0.05, 0.0, 1.0)


def _compute_e_jpeg(rgb_uint8: np.ndarray) -> np.ndarray:
    """JPEG artifact map: |I_float - JPEG_recompress(I, Q=75)| / 0.1.

    Returns float32 (H, W) in [0, 1] — mean over channels for grayscale map.
    """
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR),
                          encode_param)
    recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    recompressed_rgb = cv2.cvtColor(recompressed, cv2.COLOR_BGR2RGB).astype(
        np.float32) / 255.0
    original_f = rgb_uint8.astype(np.float32) / 255.0
    diff = np.abs(original_f - recompressed_rgb).mean(
        axis=2)  # mean over RGB → (H, W)
    return np.clip(diff / 0.1, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Boundary and spatial helpers
# ---------------------------------------------------------------------------


def _make_boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Boundary = dilate(mask) XOR erode(mask)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    return ((dilated > 127) ^ (eroded > 127)).astype(np.uint8) * 255


def _random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    extras: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """Random crop applied identically to image, mask, and any float-map extras.

    extras: optional list of (H, W) float32 maps (n_hp, n_sigma, e_jpeg)
            cropped with the same coords so they stay aligned with the image.
    """
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h, pad_w = max(0, crop_size - h), max(0, crop_size - w)
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w,
                                  cv2.BORDER_REFLECT_101)
        if extras is not None:
            extras = [
                cv2.copyMakeBorder(e, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_REFLECT_101) for e in extras
            ]
        h, w = image.shape[:2]
    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    image = image[y:y + crop_size, x:x + crop_size]
    mask = mask[y:y + crop_size, x:x + crop_size]
    if extras is not None:
        extras = [e[y:y + crop_size, x:x + crop_size] for e in extras]
    return image, mask, extras


def _pad_to_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    extras: list[np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None, int, int]:
    """Pad image/mask/extras to at least crop_size and return new h, w."""
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h, pad_w = max(0, crop_size - h), max(0, crop_size - w)
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w,
                                  cv2.BORDER_REFLECT_101)
        if extras is not None:
            extras = [
                cv2.copyMakeBorder(e, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_REFLECT_101) for e in extras
            ]
        h, w = image.shape[:2]
    return image, mask, extras, h, w


def _positive_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    bboxes: list,
    rng: random.Random,
    extras: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """PDF §9 positive crop: guaranteed to overlap a tamper bbox (±25% jitter).

    Falls back to random crop when no bboxes are available.
    """
    if not bboxes:
        return _random_crop(image, mask, crop_size, extras)
    image, mask, extras, h, w = _pad_to_crop(image, mask, crop_size, extras)
    x0, y0, x1, y1 = rng.choice(bboxes)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    jitter = crop_size // 4
    cx += rng.randint(-jitter, jitter)
    cy += rng.randint(-jitter, jitter)
    x = max(0, min(w - crop_size, cx - crop_size // 2))
    y = max(0, min(h - crop_size, cy - crop_size // 2))
    out_img = image[y:y + crop_size, x:x + crop_size]
    out_msk = mask[y:y + crop_size, x:x + crop_size]
    out_ext = ([e[y:y + crop_size, x:x + crop_size]
                for e in extras] if extras is not None else None)
    return out_img, out_msk, out_ext


def _hard_neg_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    bboxes: list,
    rng: random.Random,
    extras: list[np.ndarray] | None = None,
    max_attempts: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """PDF §9 hard-neg crop: rejection-sampled to avoid all tamper bboxes.

    Falls back to random crop after max_attempts or when no bboxes available.
    """
    if not bboxes:
        return _random_crop(image, mask, crop_size, extras)
    image, mask, extras, h, w = _pad_to_crop(image, mask, crop_size, extras)
    for _ in range(max_attempts):
        y = rng.randint(0, h - crop_size)
        x = rng.randint(0, w - crop_size)
        overlap = any(
            x < bx1 and x + crop_size > bx0 and y < by1 and y + crop_size > by0
            for bx0, by0, bx1, by1 in bboxes)
        if not overlap:
            out_img = image[y:y + crop_size, x:x + crop_size]
            out_msk = mask[y:y + crop_size, x:x + crop_size]
            out_ext = ([e[y:y + crop_size, x:x + crop_size]
                        for e in extras] if extras is not None else None)
            return out_img, out_msk, out_ext
    return _random_crop(image, mask, crop_size, extras)


def _augment_nli_v2(
    image: np.ndarray,
    mask: np.ndarray,
    crop_sizes: tuple[int, ...],
    full_page_prob: float,
    target_size: int,
    rng: random.Random,
    extras: list[np.ndarray] | None = None,
    tile_mode: str | None = None,
    bboxes: list | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """NLI-specific augmentation (lighter to preserve noise signal).

    When ``extras`` is provided (precomputed (H,W) float32 GT maps:
    n_hp, n_sigma, e_jpeg), only the *structural* transforms (resize,
    crop, rotate, flip) are applied — to all five channels in lockstep.
    The pixel-modifying transforms (brightness, blur, additive noise,
    grayscale) are SKIPPED, because those would invalidate the
    precomputed GT (which was computed on the un-augmented image).
    """
    has_extras = extras is not None and len(extras) > 0

    # Full-page downscale (30% of time)
    if rng.random() < full_page_prob:
        image = cv2.resize(image, (target_size, target_size),
                           interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (target_size, target_size),
                          interpolation=cv2.INTER_NEAREST)
        if has_extras:
            extras = [
                cv2.resize(e, (target_size, target_size),
                           interpolation=cv2.INTER_LINEAR) for e in extras
            ]
    else:
        crop_size = rng.choice(crop_sizes)
        if tile_mode == "positive" and bboxes:
            image, mask, extras = _positive_crop(image, mask, crop_size,
                                                 bboxes, rng, extras)
        elif tile_mode == "hard_neg" and bboxes:
            image, mask, extras = _hard_neg_crop(image, mask, crop_size,
                                                 bboxes, rng, extras)
        else:
            image, mask, extras = _random_crop(image, mask, crop_size, extras)

    # Rotation ±5°
    if rng.random() < 0.4:
        angle = rng.uniform(-5, 5)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image,
                               M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask,
                              M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REFLECT_101)
        if has_extras:
            extras = [
                cv2.warpAffine(e,
                               M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
                for e in extras
            ]

    # Horizontal flip
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        if has_extras:
            extras = [cv2.flip(e, 1) for e in extras]

    # ── Pixel-modifying augmentations ──────────────────────────────────
    # Skipped when precomputed GT is supplied — they would alter Y/noise
    # statistics and invalidate the GT maps we just loaded.
    if not has_extras:
        # Brightness/contrast ±6% (light — preserve noise statistics)
        if rng.random() < 0.4:
            alpha = rng.uniform(0.94, 1.06)
            beta = rng.uniform(-0.06, 0.06) * 255.0
            image = np.clip(alpha * image.astype(np.float32) + beta, 0,
                            255).astype(np.uint8)

        # Gaussian blur (light — sigma 0.3-1.0)
        if rng.random() < 0.2:
            sigma = rng.uniform(0.3, 1.0)
            ksize = max(3, int(2 * round(2 * sigma) + 1) | 1)
            image = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        # Additive Gaussian noise (sigma 1-3)
        if rng.random() < 0.2:
            noise = np.random.default_rng().normal(0, rng.uniform(
                1, 3), image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0,
                            255).astype(np.uint8)

        # NO JPEG roundtrip — would corrupt the noise signal

        # Grayscale 10%
        if rng.random() < 0.1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return image, mask, extras


class DocForgeNLIDatasetV2(Dataset):
    """PyTorch Dataset for NP-NLI v2 training on 100k doc_forge_engine output.

    Returns dict:
        image:      (3, H, W) float32 [0, 1]
        mask:       (1, H, W) float32 binary
        boundary:   (1, H, W) float32 binary — dilate XOR erode of mask
        label:      scalar float32 {0, 1}
        description: str
        n_hp_gt:    (1, H, W) float32 — high-pass noise residual proxy [0, 1]
        n_sigma_gt: (1, H, W) float32 — per-block MAD sigma map [0, 1]
        e_jpeg_gt:  (1, H, W) float32 — JPEG artifact map [0, 1]
        alpha:      scalar float32 — anomaly weight (0.0=authentic, 0.4=hard, 1.0=easy)
    """

    def __init__(
        self,
        entries: list[DocForgeEntry],
        *,
        augment: bool,
        crop_sizes: tuple[int, ...] = (256, 384, 512),
        target_size: int = 256,
        full_page_prob: float = 0.3,
        seed: int = 42,
        prefer_precomputed_gt: bool = False,
        use_tile_sampling: bool = False,
        n_hp_override_dir: str | Path | None = None,
    ) -> None:
        """
        prefer_precomputed_gt:
            When True, load engine-emitted GT PNGs (`entry.n_hp_path` etc.)
            and pass them through `_augment_nli_v2` in lockstep with the
            image (same crop / rotate / flip applied). Pixel-modifying
            augmentations (brightness, blur, additive noise, grayscale)
            are skipped to keep the GT valid. Falls back to on-the-fly
            compute when any path is missing on disk.
            Default False — preserves the historical proxy-on-augmented
            behavior. Eliminates the per-sample JPEG re-encode in
            `_compute_e_jpeg` (the dataloader bottleneck).
        use_tile_sampling:
            PDF §9 within-doc spatial-aware cropping. For tampered samples
            with known bbox annotations: 50% crops overlap a tamper region
            (positive), 30% are rejection-sampled to avoid tamper regions
            (hard-neg), 20% are uniform random. Ignored when augment=False.
        """
        self.entries = entries
        self.augment = augment
        self.crop_sizes = crop_sizes
        self.target_size = target_size
        self.full_page_prob = full_page_prob if augment else 0.0
        self.seed = seed
        self.prefer_precomputed_gt = prefer_precomputed_gt
        self.use_tile_sampling = use_tile_sampling
        self.n_hp_override_dir = (Path(n_hp_override_dir)
                                  if n_hp_override_dir else None)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]

        # Load image — skip corrupt/empty/broken files by trying next valid entry
        try:
            bgr = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                return self.__getitem__((index + 1) % len(self))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        # Load or create mask
        if entry.mask_path is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        else:
            mask = cv2.imread(str(entry.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            elif mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # ── Pre-augment: try to load precomputed GT at native resolution ──
        # We feed (image, mask, [n_hp, n_sigma, e_jpeg]) through the augment
        # together so the structural transforms apply in lockstep. This
        # eliminates the per-sample JPEG re-encode in `_compute_e_jpeg` —
        # the dataloader bottleneck.
        extras_in: list[np.ndarray] | None = None

        # Diagnostic override: swap n_hp GT for an externally-provided map
        # (e.g. diff_hp_authentic / diff_hp_forgery for D3/D4). Looked up by
        # image stem. Missing override file → skip this entry.
        if self.n_hp_override_dir is not None:
            override_p = self.n_hp_override_dir / f"{entry.image_path.stem}.png"
            if not override_p.exists():
                return self.__getitem__((index + 1) % len(self))
            n_hp_override = cv2.imread(str(override_p), cv2.IMREAD_GRAYSCALE)
            if n_hp_override is None:
                return self.__getitem__((index + 1) % len(self))
            H, W = rgb.shape[:2]
            if n_hp_override.shape != (H, W):
                n_hp_override = cv2.resize(n_hp_override, (W, H),
                                           interpolation=cv2.INTER_LINEAR)
            extras_in = [
                n_hp_override.astype(np.float32) / 255.0,
                np.zeros((H, W), dtype=np.float32),  # n_sigma — unused (w=0)
                np.zeros((H, W), dtype=np.float32),  # e_jpeg — unused (w=0)
            ]
        elif (self.prefer_precomputed_gt and entry.n_hp_path is not None
              and entry.n_sigma_path is not None
              and entry.e_jpeg_path is not None and entry.n_hp_path.exists()
              and entry.n_sigma_path.exists() and entry.e_jpeg_path.exists()):
            n_hp_u = cv2.imread(str(entry.n_hp_path), cv2.IMREAD_GRAYSCALE)
            n_sigma_u = cv2.imread(str(entry.n_sigma_path),
                                   cv2.IMREAD_GRAYSCALE)
            e_jpeg_u = cv2.imread(str(entry.e_jpeg_path), cv2.IMREAD_GRAYSCALE)
            if (n_hp_u is not None and n_sigma_u is not None
                    and e_jpeg_u is not None):
                H, W = rgb.shape[:2]
                # Engine emits GT at saved-image native resolution; align if
                # the manifest's image_path was resized after-the-fact.
                if n_hp_u.shape != (H, W):
                    n_hp_u = cv2.resize(n_hp_u, (W, H),
                                        interpolation=cv2.INTER_LINEAR)
                    n_sigma_u = cv2.resize(n_sigma_u, (W, H),
                                           interpolation=cv2.INTER_LINEAR)
                    e_jpeg_u = cv2.resize(e_jpeg_u, (W, H),
                                          interpolation=cv2.INTER_LINEAR)
                extras_in = [
                    n_hp_u.astype(np.float32) / 255.0,
                    n_sigma_u.astype(np.float32) / 255.0,
                    e_jpeg_u.astype(np.float32) / 255.0,
                ]

        # Augment or center-crop — extras carry through structurally
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
            if extras_in is not None:
                extras_out = [
                    cv2.resize(e, (self.target_size, self.target_size),
                               interpolation=cv2.INTER_LINEAR)
                    for e in extras_in
                ]
            else:
                extras_out = None

        # Resize fallback if augment didn't normalize size
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

        # GT-side noise profiles: precomputed (transformed in lockstep) or
        # fall back to on-the-fly recomputation on the augmented image.
        if extras_out is not None:
            n_hp, n_sigma, e_jpeg = extras_out
        else:
            n_hp = _compute_n_hp_proxy(rgb)
            n_sigma = _compute_sigma_map(n_hp, block_size=16)
            e_jpeg = _compute_e_jpeg(rgb)

        # To tensors
        image_t = torch.from_numpy(
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
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
