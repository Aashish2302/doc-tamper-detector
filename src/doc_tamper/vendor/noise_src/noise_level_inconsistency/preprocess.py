from __future__ import annotations

from typing import cast

import numpy as np
from PIL import Image

from noise_level_inconsistency.config import PreprocessConfig


def resolve_image_dpi(image: Image.Image, fallback: int) -> int:
    raw = image.info.get("dpi")
    if isinstance(raw, tuple) and raw:
        return max(1, int(round(float(raw[0]))))
    if isinstance(raw, (int, float)):
        return max(1, int(round(float(raw))))
    return fallback


def normalize_input_image(
    image: Image.Image,
    config: PreprocessConfig,
) -> tuple[Image.Image, np.ndarray, int, list[str]]:
    rgb = image.convert("RGB")
    notes: list[str] = ["rgb_preserved=true", "grayscale_copy=true"]
    source_dpi = resolve_image_dpi(rgb, config.default_input_dpi)
    target_dpi = max(1, int(config.target_dpi))

    if min(rgb.size) < config.min_dimension:
        raise ValueError(f"Input is too small for NLI analysis: {rgb.size}")

    scale = target_dpi / source_dpi
    if abs(scale - 1.0) >= 0.05:
        width = max(config.min_dimension, int(round(rgb.width * scale)))
        height = max(config.min_dimension, int(round(rgb.height * scale)))
        rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        notes.append(f"dpi_normalized={source_dpi}->{target_dpi}")
        source_dpi = target_dpi
    else:
        notes.append(f"dpi_normalized={source_dpi}")

    gray_image = cast(Image.Image, rgb.convert("L"))
    gray = np.asarray(gray_image, dtype=np.float32)
    return rgb, gray, source_dpi, notes
