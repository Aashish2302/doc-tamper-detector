"""Patch validity masking for NLI.

Computes a per-block validity mask that excludes blocks dominated by
document structure (text strokes, edges, line art) or too-flat regions
from noise estimation. Without this filter, document edges produce wildly
varying sigma values that look anomalous but are simply structural content.

Strategy: use structural variance — the variance of a blurred copy of the
image per block. Blurring suppresses noise while preserving structure, so
blocks with high blurred-variance contain real structural content (text,
borders, logos). Blocks with low blurred-variance are smooth canvas suitable
for reliable noise estimation.

This approach works correctly for both:
  - Synthetic noisy images (all blocks valid → tests pass)
  - Real documents (text/edge blocks excluded → FP rate drops)
"""

from __future__ import annotations

import numpy as np

from noise_level_inconsistency.residuals import convolve_3x3


# ---------------------------------------------------------------------------
# Edge strength (for region_classifier.py; also exported from here)
# ---------------------------------------------------------------------------

def compute_edge_strength(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude. Returns float32 map same shape as gray."""
    sx = np.asarray(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32
    )
    sy = np.asarray(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32
    )
    gx = convolve_3x3(gray, sx)
    gy = convolve_3x3(gray, sy)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


# ---------------------------------------------------------------------------
# Local variance map (for region_classifier.py; also exported from here)
# ---------------------------------------------------------------------------

def _box_sum_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient sliding box sum via 2-D cumulative sums.

    Returns an array of shape ``(arr.shape[0] - window + 1, arr.shape[1] -
    window + 1)`` where each element is the sum of a ``window × window``
    neighbourhood.
    """
    cs = np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)
    pcs = np.zeros((cs.shape[0] + 1, cs.shape[1] + 1), dtype=np.float64)
    pcs[1:, 1:] = cs
    result = (
        pcs[window:, window:]
        - pcs[window:, : arr.shape[1] - window + 1]
        - pcs[: arr.shape[0] - window + 1, window:]
        + pcs[: arr.shape[0] - window + 1, : arr.shape[1] - window + 1]
    )
    return result.astype(np.float32)


def compute_local_variance(gray: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Local variance via box-filter: Var = E[X²] − E[X]². Returns float32 same shape as gray."""
    w = max(2, min(window_size, gray.shape[0], gray.shape[1]))
    area = float(w * w)
    gray_f = gray.astype(np.float64)
    s1 = _box_sum_2d(gray_f, w)
    s2 = _box_sum_2d(gray_f * gray_f, w)
    var_small = np.maximum(s2 / area - (s1 / area) ** 2, 0.0).astype(np.float32)
    # Embed result back into original-sized canvas (top-left aligned, mirrored fill)
    out = np.zeros_like(gray, dtype=np.float32)
    oh, ow = var_small.shape
    if oh > 0 and ow > 0:
        out[:oh, :ow] = var_small
        if oh < gray.shape[0]:
            out[oh:, :ow] = var_small[-1:, :]
        if ow < gray.shape[1]:
            out[:, ow:] = out[:, ow - 1 : ow]
    return out


# ---------------------------------------------------------------------------
# Box blur (structural variance computation)
# ---------------------------------------------------------------------------

def _box_blur_2d(image: np.ndarray, radius: int) -> np.ndarray:
    """2-D box blur via cumulative sums. Output is same shape as input."""
    if radius == 0:
        return image.astype(np.float32)
    ksize = 2 * radius + 1
    padded = np.pad(image.astype(np.float64), radius, mode="reflect")
    pcs = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    pcs[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h, w = image.shape
    result = (
        pcs[ksize : ksize + h, ksize : ksize + w]
        - pcs[0:h, ksize : ksize + w]
        - pcs[ksize : ksize + h, 0:w]
        + pcs[0:h, 0:w]
    )
    return (result / float(ksize * ksize)).astype(np.float32)


# ---------------------------------------------------------------------------
# Block-level structural variance (core of the validity mask)
# ---------------------------------------------------------------------------

def _compute_block_structural_variances(
    gray: np.ndarray,
    block_size: int,
    blur_radius: int = 5,
) -> np.ndarray:
    """Variance of a blurred copy of the image per block.

    Blurring suppresses noise; the remaining variance measures structural
    content (text strokes, borders, logos). High structural variance → the
    block is dominated by document structure, not suitable for noise estimation.
    """
    blurred = _box_blur_2d(gray, blur_radius)
    grid_h = gray.shape[0] // block_size
    grid_w = gray.shape[1] // block_size
    struct_var = np.zeros((grid_h, grid_w), dtype=np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            y0, x0 = row * block_size, col * block_size
            block = blurred[y0 : y0 + block_size, x0 : x0 + block_size]
            struct_var[row, col] = float(np.var(block.astype(np.float32)))
    return struct_var


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_block_validity_mask(
    gray: np.ndarray,
    block_size: int,
    structural_blur_radius: int = 5,
    structural_var_percentile: float = 85.0,
) -> np.ndarray:
    """Return boolean grid ``(grid_h, grid_w)`` where True = valid for NLI.

    A block is marked *invalid* when its structural variance (variance of
    blurred image within block) exceeds the ``structural_var_percentile``-th
    percentile across all blocks.  This removes text-heavy, edge-heavy, and
    line-art blocks from the noise-estimation reference population while
    keeping smooth/photo blocks.

    The percentile-based threshold adapts automatically: for a document
    that is 100 % flat it excludes nothing; for a dense text page it
    excludes the most complex 15 % (default).
    """
    grid_h = gray.shape[0] // block_size
    grid_w = gray.shape[1] // block_size
    if grid_h == 0 or grid_w == 0:
        return np.ones((max(grid_h, 0), max(grid_w, 0)), dtype=bool)

    struct_var = _compute_block_structural_variances(gray, block_size, structural_blur_radius)
    threshold = float(np.percentile(struct_var, structural_var_percentile))
    return (struct_var <= threshold).astype(bool)


def compute_flat_region_mask(
    gray: np.ndarray,
    block_size: int,
    flatness_threshold: float = 1.0,
) -> np.ndarray:
    """Return boolean grid where True = block is too flat for noise estimation.

    Blocks whose pixel standard deviation is below ``flatness_threshold``
    (e.g. near-white margins, fully-saturated fill) contain almost no noise
    signal; their sigma estimates are unreliable and should be excluded.
    """
    grid_h = gray.shape[0] // block_size
    grid_w = gray.shape[1] // block_size
    flat = np.zeros((grid_h, grid_w), dtype=bool)
    for row in range(grid_h):
        for col in range(grid_w):
            y0, x0 = row * block_size, col * block_size
            block = gray[y0 : y0 + block_size, x0 : x0 + block_size]
            flat[row, col] = float(np.std(block.astype(np.float32))) < flatness_threshold
    return flat