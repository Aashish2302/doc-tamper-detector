from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class SyntheticForgery:
    image: Image.Image
    mask: Image.Image
    bbox: tuple[int, int, int, int]


def _base_document(width: int, height: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    canvas = 236.0 + 8.0 * np.sin(xv * 6.0) + 4.0 * np.cos(yv * 8.0)
    canvas -= ((np.mod(np.arange(height)[:, None], 28) < 2) * 10.0).astype(np.float32)
    canvas -= ((np.mod(np.arange(width)[None, :], 54) < 2) * 6.0).astype(np.float32)
    return np.clip(canvas, 0.0, 255.0)


def create_spliced_document(
    host_sigma: float,
    patch_sigma: float,
    *,
    seed: int = 0,
    size: tuple[int, int] = (256, 256),
    patch_bbox: tuple[int, int, int, int] = (88, 80, 176, 168),
) -> SyntheticForgery:
    rng = np.random.default_rng(seed)
    width, height = size
    base = _base_document(width, height)
    host_noise = rng.normal(0.0, host_sigma, size=(height, width)).astype(np.float32)
    image = base + host_noise
    x0, y0, x1, y1 = patch_bbox
    patch_noise = rng.normal(0.0, patch_sigma, size=(y1 - y0, x1 - x0)).astype(np.float32)
    image[y0:y1, x0:x1] = base[y0:y1, x0:x1] + patch_noise
    image = np.clip(image, 0.0, 255.0).astype("uint8")

    rgb = np.stack([image, image, image], axis=-1)
    mask = np.zeros((height, width), dtype="uint8")
    mask[y0:y1, x0:x1] = 255
    return SyntheticForgery(
        image=Image.fromarray(rgb),
        mask=Image.fromarray(mask),
        bbox=patch_bbox,
    )


def save_synthetic_forgery(
    output_image: str | Path,
    output_mask: str | Path,
    host_sigma: float,
    patch_sigma: float,
    *,
    seed: int,
) -> tuple[Path, Path]:
    sample = create_spliced_document(host_sigma=host_sigma, patch_sigma=patch_sigma, seed=seed)
    image_path = Path(output_image)
    mask_path = Path(output_mask)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    sample.image.save(image_path, format="PNG", dpi=(200, 200))
    sample.mask.save(mask_path, format="PNG")
    return image_path, mask_path
