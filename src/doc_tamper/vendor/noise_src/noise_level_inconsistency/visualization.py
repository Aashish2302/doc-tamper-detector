from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

# ---------------------------------------------------------------------------
# Perceptual "hot" colormap (black → red → orange → yellow → white)
# Maps 0→background noise, 1→highest anomaly, without the all-blue problem.
# Pure numpy; no matplotlib dependency.
# ---------------------------------------------------------------------------
_HOT_R = np.array([0, 128, 255, 255, 255], dtype=np.float32)
_HOT_G = np.array([0,   0,   0, 200, 255], dtype=np.float32)
_HOT_B = np.array([0,   0,   0,   0, 255], dtype=np.float32)
_HOT_X = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def _apply_hot_colormap(normalized: np.ndarray) -> np.ndarray:
    """Map [0,1] float array → RGB uint8 using perceptual hot colormap."""
    flat = normalized.ravel()
    r = np.interp(flat, _HOT_X, _HOT_R).astype("uint8")
    g = np.interp(flat, _HOT_X, _HOT_G).astype("uint8")
    b = np.interp(flat, _HOT_X, _HOT_B).astype("uint8")
    return np.stack([r, g, b], axis=-1).reshape(normalized.shape + (3,))


def _normalize(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


def _normalize_percentile(array: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Normalize by percentile range so outliers don't collapse the colour scale."""
    if array.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    p_lo = float(np.percentile(array, lo))
    p_hi = float(np.percentile(array, hi))
    if p_hi - p_lo <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array.astype(np.float32) - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def to_grayscale_image(array: np.ndarray) -> Image.Image:
    normalized = (_normalize(array) * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(normalized)


def to_heatmap_image(array: np.ndarray) -> Image.Image:
    """Perceptual hot colormap normalised by [1st, 99th] percentile.

    The old min-max normalisation compressed almost all z-scores into the
    near-zero (blue) end because a handful of outlier blocks set the scale.
    Percentile clipping spreads colours meaningfully across the distribution.
    """
    normalized = _normalize_percentile(array, lo=1.0, hi=99.0)
    return Image.fromarray(_apply_hot_colormap(normalized))


def save_image(image: Image.Image, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    return target


def build_overlay_image(
    rgb_image: Image.Image,
    anomaly_mask: np.ndarray,
    regions: list[tuple[int, int, int, int]],
) -> Image.Image:
    """Draw anomalous block highlights and (small) region bounding boxes.

    Block highlights come from the actual per-block anomaly mask, so they
    follow the detection shape precisely.

    Bounding-box rectangles are only drawn when the region bbox covers less
    than 40 % of the image area — a region spanning the whole image is better
    communicated by the block highlights alone; a giant rectangle border just
    obscures the image.
    """
    base = rgb_image.convert("RGBA")
    img_area = base.width * base.height
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    red = ImageColor.getrgb("#ff3b30")
    draw = ImageDraw.Draw(overlay)

    # --- per-block highlights (semi-transparent red fill) ---
    if anomaly_mask.size > 0:
        ys, xs = np.where(anomaly_mask)
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((x, y), fill=red + (100,))

    # --- region outlines only for reasonably-sized regions ---
    for x0, y0, x1, y1 in regions:
        bbox_area = max(0, x1 - x0) * max(0, y1 - y0)
        if img_area > 0 and bbox_area / img_area > 0.40:
            continue  # skip giant bbox outlines; block highlights carry the info
        draw.rectangle((x0, y0, x1, y1), outline=red + (255,), width=2)

    return Image.alpha_composite(base, overlay).convert("RGB")


def save_overlay(
    rgb_image: Image.Image,
    anomaly_mask: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    path: str | Path,
) -> Path:
    return save_image(build_overlay_image(rgb_image, anomaly_mask, regions), path)


def save_debug_panel(images: list[tuple[str, Image.Image]], path: str | Path) -> Path:
    margin = 16
    title_height = 24
    width = sum(image.width for _, image in images) + margin * (len(images) + 1)
    height = max(image.height for _, image in images) + margin * 2 + title_height
    canvas = Image.new("RGB", (width, height), "#111111")
    draw = ImageDraw.Draw(canvas)
    x = margin
    for label, image in images:
        draw.text((x, margin), label, fill="white")
        canvas.paste(image.convert("RGB"), (x, margin + title_height))
        x += image.width + margin
    return save_image(canvas, path)
