"""Dataset loader for doc_forge_engine output -- NLI-specific variant.

Uses the pre-built train/val/test splits produced by ``doc-forge split``::

    outputs_domicile/
    +-- originals/images/   orig_000000.jpg  ...
    +-- tampered/images/    tampered_orig_000000.png ...
    +-- tampered/masks/     tampered_orig_000000.png ...
    +-- splits/
        +-- train.json      (document pairs)
        +-- val.json        (document pairs)
        +-- test.json       (document pairs)

Each split JSON is a list of ``{"original": {...}, "tampered": {...}}`` pairs.
Image paths inside the JSON are relative to the repo root.

NLI-specific augmentation is LIGHTER than ELA to avoid corrupting the noise
signal:
    - Multi-scale random crop from {256, 384, 512}
    - Rotation +/-5 deg, horizontal flip
    - Brightness/contrast +/-6%
    - Gaussian blur sigma 0.3-1.0
    - Additive Gaussian noise sigma 1-3
    - NO JPEG roundtrip (would corrupt noise signal)
    - Grayscale 10% chance
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


# ---------------------------------------------------------------------------
# Data entry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DocForgeEntry:
    group_id: str
    image_path: Path
    mask_path: Path | None   # None for authentic (zero mask)
    label: int               # 0 = authentic, 1 = tampered
    description: str


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_image_path(raw_path: str, dataset_root: Path) -> Path:
    """Resolve an image path from the split JSON.

    Paths in the JSON may be absolute or relative.  If relative they are
    typically relative to the doc_forge_engine project root (one level above
    dataset_root).  We try several strategies:
    1. Already absolute and exists.
    2. Relative to dataset_root's parent.
    3. Relative to dataset_root itself.
    4. Return raw path as-is (let cv2 report the error later).
    """
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = dataset_root.parent / p
    if candidate.exists():
        return candidate
    candidate = dataset_root / p
    if candidate.exists():
        return candidate
    return p


# ---------------------------------------------------------------------------
# Split loader
# ---------------------------------------------------------------------------

def load_split_entries(dataset_root: str | Path, split_name: str) -> list[DocForgeEntry]:
    """Load entries from a pre-built split file (train.json / val.json / test.json)."""
    root = Path(dataset_root)
    split_path = root / "splits" / f"{split_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    pairs = json.loads(split_path.read_text(encoding="utf-8"))
    entries: list[DocForgeEntry] = []

    for pair in pairs:
        orig = pair["original"]
        tamp = pair["tampered"]
        group_id = orig["doc_id"]

        # Authentic entry (zero mask)
        entries.append(DocForgeEntry(
            group_id=group_id,
            image_path=_resolve_image_path(orig["image_path"], root),
            mask_path=None,
            label=0,
            description="docforge_authentic",
        ))

        # Tampered entry (with mask)
        entries.append(DocForgeEntry(
            group_id=group_id,
            image_path=_resolve_image_path(tamp["image_path"], root),
            mask_path=_resolve_image_path(tamp["mask_path"], root),
            label=1,
            description="docforge_tampered",
        ))

    if not entries:
        raise ValueError(f"No entries in split file: {split_path}")
    return entries


# ---------------------------------------------------------------------------
# Augmentation helpers (NLI-specific, lighter than ELA)
# ---------------------------------------------------------------------------

def _random_crop(image: np.ndarray, mask: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Random crop with zero-padding if the image is smaller than crop_size."""
    h, w = image.shape[:2]

    # Pad if smaller
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )
        mask = cv2.copyMakeBorder(
            mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )
        h, w = image.shape[:2]

    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    return image[y : y + crop_size, x : x + crop_size], mask[y : y + crop_size, x : x + crop_size]


def _augment_nli(
    image: np.ndarray,
    mask: np.ndarray,
    crop_sizes: tuple[int, ...] = (256, 384, 512),
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply NLI-specific augmentation.

    Returns (image, mask) as uint8 arrays.
    """
    if rng is None:
        rng = random.Random()

    # --- Multi-scale random crop ---
    crop_size = rng.choice(crop_sizes)
    image, mask = _random_crop(image, mask, crop_size)

    # --- Rotation +/- 5 degrees ---
    if rng.random() < 0.5:
        angle = rng.uniform(-5.0, 5.0)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    # --- Horizontal flip ---
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # --- Brightness / contrast +/- 6% ---
    if rng.random() < 0.5:
        alpha = rng.uniform(0.94, 1.06)  # contrast
        beta = rng.uniform(-0.06, 0.06) * 255.0  # brightness
        img_f = image.astype(np.float32)
        img_f = np.clip(alpha * img_f + beta, 0, 255)
        image = img_f.astype(np.uint8)

    # --- Gaussian blur (sigma 0.3 - 1.0) ---
    if rng.random() < 0.3:
        sigma = rng.uniform(0.3, 1.0)
        ksize = int(2 * round(2 * sigma) + 1)
        ksize = max(ksize, 3)
        if ksize % 2 == 0:
            ksize += 1
        image = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # --- Additive Gaussian noise (sigma 1-3) ---
    if rng.random() < 0.3:
        noise_sigma = rng.uniform(1.0, 3.0)
        noise = np.random.default_rng().normal(0, noise_sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --- Grayscale 10% chance (keep 3-channel by duplicating) ---
    if rng.random() < 0.1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return image, mask


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def _image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 RGB to CHW float32 in [0, 1]."""
    return torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)


def _mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert HW uint8 mask to (1, H, W) float32 in {0, 1}."""
    binary = (mask > 127).astype(np.float32)
    return torch.from_numpy(binary[np.newaxis, :, :])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class DocForgeNLIDataset(Dataset):
    """PyTorch Dataset for NLI training on doc_forge_engine output.

    Returns a dict:
        image: (3, H, W) float32 in [0, 1]
        mask:  (1, H, W) float32 binary
        label: scalar float32 (0 = authentic, 1 = tampered)
        description: str
    """

    def __init__(
        self,
        entries: list[DocForgeEntry],
        *,
        augment: bool,
        crop_sizes: tuple[int, ...] = (256, 384, 512),
        target_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.entries = entries
        self.augment = augment
        self.crop_sizes = crop_sizes
        self.target_size = target_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        entry = self.entries[index]

        # --- Load image ---
        image = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {entry.image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load or create mask ---
        if entry.mask_path is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        else:
            loaded_mask = cv2.imread(str(entry.mask_path), cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None:
                raise ValueError(f"Could not decode mask: {entry.mask_path}")
            # Ensure mask matches image size
            if loaded_mask.shape[:2] != rgb.shape[:2]:
                loaded_mask = cv2.resize(
                    loaded_mask, (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = loaded_mask

        # --- Augment (train) or center-crop (val) ---
        if self.augment:
            rng = random.Random(self.seed + index)
            rgb, mask = _augment_nli(rgb, mask, self.crop_sizes, rng)
        else:
            # Deterministic center crop for validation
            crop_size = self.target_size
            h, w = rgb.shape[:2]
            if h < crop_size or w < crop_size:
                pad_h = max(0, crop_size - h)
                pad_w = max(0, crop_size - w)
                rgb = cv2.copyMakeBorder(
                    rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
                )
                mask = cv2.copyMakeBorder(
                    mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
                )
                h, w = rgb.shape[:2]
            y_start = (h - crop_size) // 2
            x_start = (w - crop_size) // 2
            rgb = rgb[y_start : y_start + crop_size, x_start : x_start + crop_size]
            mask = mask[y_start : y_start + crop_size, x_start : x_start + crop_size]

        # --- Resize to target_size if augmented crop was different ---
        if rgb.shape[0] != self.target_size or rgb.shape[1] != self.target_size:
            rgb = cv2.resize(rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        return {
            "image": _image_to_tensor(rgb),
            "mask": _mask_to_tensor(mask),
            "label": torch.tensor(float(entry.label), dtype=torch.float32),
            "description": f"{entry.description}:{entry.image_path.name}",
        }
