"""Block-level region type classification for NLI.

Classifies each image block as FLAT_BACKGROUND, TEXT, or PHOTO based on
local edge density and structural variance.  This enables per-region-type
sigma references so photo noise and background noise are never compared
against each other, avoiding false positives driven by naturally different
noise levels in different document zones.

Classification heuristics (percentile-adaptive, not fixed thresholds):
  FLAT_BACKGROUND  low edge density AND low local variance
  TEXT             high edge density (sharp strokes) regardless of variance
  PHOTO            everything else (moderate edges, moderate-to-high variance)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class RegionType(IntEnum):
    FLAT_BACKGROUND = 0
    TEXT = 1
    PHOTO = 2


def classify_blocks(
    gray: np.ndarray,
    block_size: int,
    edge_strength: np.ndarray,
    local_variance: np.ndarray,
) -> np.ndarray:
    """Classify each block as RegionType. Returns int32 grid ``(grid_h, grid_w)``."""
    grid_h = gray.shape[0] // block_size
    grid_w = gray.shape[1] // block_size
    result = np.full((grid_h, grid_w), int(RegionType.PHOTO), dtype=np.int32)

    if grid_h == 0 or grid_w == 0:
        return result

    block_edge_mean = np.zeros((grid_h, grid_w), dtype=np.float32)
    block_var_mean = np.zeros((grid_h, grid_w), dtype=np.float32)

    for row in range(grid_h):
        for col in range(grid_w):
            y0, x0 = row * block_size, col * block_size
            block_edge_mean[row, col] = float(
                np.mean(edge_strength[y0 : y0 + block_size, x0 : x0 + block_size])
            )
            block_var_mean[row, col] = float(
                np.mean(local_variance[y0 : y0 + block_size, x0 : x0 + block_size])
            )

    # Adaptive percentile thresholds
    edge_low = float(np.percentile(block_edge_mean, 20))   # clearly not-edge
    edge_high = float(np.percentile(block_edge_mean, 65))  # clearly text-edge
    var_low = float(np.percentile(block_var_mean, 25))     # clearly flat

    for row in range(grid_h):
        for col in range(grid_w):
            e = block_edge_mean[row, col]
            v = block_var_mean[row, col]
            if e <= edge_low and v <= var_low:
                result[row, col] = int(RegionType.FLAT_BACKGROUND)
            elif e >= edge_high:
                result[row, col] = int(RegionType.TEXT)
            # else: stays PHOTO (default)

    return result


def compute_per_region_reference(
    sigma_grid: np.ndarray,
    region_type_grid: np.ndarray,
    validity_mask: np.ndarray,
    min_blocks: int = 10,
) -> dict[int, tuple[float, float]]:
    """Compute ``(median_sigma, mad_sigma)`` per region type using only valid blocks.

    For types with fewer than ``min_blocks`` valid blocks the overall
    valid-block statistics are used as fallback.

    Returns a dict keyed by ``RegionType`` integer values.
    """
    valid_sigmas_all = sigma_grid[validity_mask] if validity_mask.any() else sigma_grid.ravel()
    if valid_sigmas_all.size == 0:
        valid_sigmas_all = sigma_grid.ravel()

    overall_median = float(np.median(valid_sigmas_all))
    overall_mad = float(np.median(np.abs(valid_sigmas_all - overall_median)) * 1.4826)
    if overall_mad <= 1e-6:
        overall_mad = float(np.std(valid_sigmas_all) + 1e-6)

    result: dict[int, tuple[float, float]] = {}
    for rtype in (RegionType.FLAT_BACKGROUND, RegionType.TEXT, RegionType.PHOTO):
        type_mask = (region_type_grid == int(rtype)) & validity_mask
        type_sigmas = sigma_grid[type_mask]
        if type_sigmas.size >= min_blocks:
            med = float(np.median(type_sigmas))
            mad = float(np.median(np.abs(type_sigmas - med)) * 1.4826)
            if mad <= 1e-6:
                mad = float(np.std(type_sigmas) + 1e-6)
            result[int(rtype)] = (med, mad)
        else:
            result[int(rtype)] = (overall_median, overall_mad)

    return result