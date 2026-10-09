"""Noise-sigma estimation and z-score computation for NLI.

Core responsibilities
---------------------
1. Estimate per-block noise standard deviation via Haar-wavelet MAD.
2. Compute z-scores that indicate how far each block deviates from the
   expected noise level.
3. Combine multiple block scales into one analysis structure.

Phase-1 additions
-----------------
* ``robust_z_grid_filtered``     — z-scores referenced only to *valid* blocks,
  so structural (text/edge) blocks cannot inflate the reference statistics.
* ``robust_z_grid_region_aware`` — each block is compared to the reference for
  its own region type (FLAT_BACKGROUND / TEXT / PHOTO), preventing photo noise
  from looking anomalous against a white-page background reference.
* ``analyze_sigma_maps_v2``      — drop-in replacement for ``analyze_sigma_maps``
  that accepts an optional validity mask and region-type grid.

Phase-2 additions
-----------------
* ``compute_block_mean_intensity``  — mean intensity per block (for NLF).
* ``estimate_noise_level_function`` — fit expected sigma vs. intensity curve.
* ``intensity_aware_z_grid``       — z-scores relative to NLF(intensity) rather
  than a single global median; prevents dark photo regions from being
  flagged as anomalous simply because they differ from bright backgrounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noise_level_inconsistency.models import SigmaScale


# ---------------------------------------------------------------------------
# Block-level sigma estimation (unchanged from original)
# ---------------------------------------------------------------------------

def _haar_detail_coeffs(block: np.ndarray) -> np.ndarray:
    height = block.shape[0] - (block.shape[0] % 2)
    width = block.shape[1] - (block.shape[1] % 2)
    cropped = block[:height, :width]
    if height == 0 or width == 0:
        return np.asarray([], dtype=np.float32)

    a = cropped[0::2, 0::2]
    b = cropped[0::2, 1::2]
    c = cropped[1::2, 0::2]
    d = cropped[1::2, 1::2]
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return np.concatenate([lh.ravel(), hl.ravel(), hh.ravel()]).astype(np.float32, copy=False)


def estimate_block_sigma_mad(block: np.ndarray) -> float:
    """Robust MAD-based noise sigma estimate for a single block."""
    coeffs = np.abs(_haar_detail_coeffs(block))
    if coeffs.size == 0:
        return 0.0
    return float(np.median(coeffs) / 0.6745)


def compute_sigma_grid(residual: np.ndarray, block_size: int) -> np.ndarray:
    """Compute sigma estimate for every non-overlapping block. Returns float32 grid."""
    grid_height = residual.shape[0] // block_size
    grid_width = residual.shape[1] // block_size
    if grid_height == 0 or grid_width == 0:
        return np.zeros((0, 0), dtype=np.float32)

    cropped = residual[: grid_height * block_size, : grid_width * block_size]
    sigma = np.zeros((grid_height, grid_width), dtype=np.float32)
    for row in range(grid_height):
        y0 = row * block_size
        y1 = y0 + block_size
        for col in range(grid_width):
            x0 = col * block_size
            x1 = x0 + block_size
            sigma[row, col] = estimate_block_sigma_mad(cropped[y0:y1, x0:x1])
    return sigma


# ---------------------------------------------------------------------------
# Z-score computation — original (uses all blocks)
# ---------------------------------------------------------------------------

def robust_z_grid(sigma_grid: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-scores using all blocks as reference.  Original implementation."""
    if sigma_grid.size == 0:
        return sigma_grid.copy(), 0.0, 1.0
    median = float(np.median(sigma_grid))
    mad = float(np.median(np.abs(sigma_grid - median)) * 1.4826)
    if mad <= 1e-6:
        mad = float(np.std(sigma_grid) + 1e-6)
    z = (sigma_grid - median) / mad
    return z.astype(np.float32, copy=False), median, mad


# ---------------------------------------------------------------------------
# Z-score computation — Phase 1: validity-filtered reference
# ---------------------------------------------------------------------------

def robust_z_grid_filtered(
    sigma_grid: np.ndarray,
    validity_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Z-scores referenced only to *valid* blocks.

    The reference median and MAD are computed exclusively from blocks that
    passed the structural-variance filter, so text/edge blocks cannot inflate
    the reference statistics.  Z-scores are still produced for *all* blocks
    (including invalid ones) so no information is lost downstream.

    Falls back to ``robust_z_grid`` when no valid blocks exist.
    """
    if validity_mask.size == 0 or not validity_mask.any():
        return robust_z_grid(sigma_grid)

    valid_sigmas = sigma_grid[validity_mask]
    median = float(np.median(valid_sigmas))
    mad = float(np.median(np.abs(valid_sigmas - median)) * 1.4826)
    if mad <= 1e-6:
        mad = float(np.std(valid_sigmas) + 1e-6)
    z = (sigma_grid - median) / mad
    return z.astype(np.float32, copy=False), median, mad


# ---------------------------------------------------------------------------
# Z-score computation — Phase 1: region-aware reference
# ---------------------------------------------------------------------------

def robust_z_grid_region_aware(
    sigma_grid: np.ndarray,
    region_type_grid: np.ndarray,
    validity_mask: np.ndarray,
    per_region_ref: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, float, float]:
    """Z-scores where each block is compared to the reference for its region type.

    For a block at (r, c) with region type t:
        z[r, c] = (sigma[r, c] − median_t) / mad_t

    This prevents photo blocks from appearing anomalous simply because they
    have different noise levels than flat white document backgrounds.

    Returns
    -------
    z_grid : float32 array
    dominant_sigma : overall median from valid blocks
    dominant_mad : overall MAD from valid blocks
    """
    z = np.zeros_like(sigma_grid, dtype=np.float32)
    processed = np.zeros_like(sigma_grid, dtype=bool)

    for rtype_val, (med, mad) in per_region_ref.items():
        mask = region_type_grid == rtype_val
        if mask.any():
            z[mask] = (sigma_grid[mask] - med) / max(mad, 1e-6)
            processed |= mask

    # Fallback for any unmatched blocks (shouldn't happen in practice)
    if not processed.all():
        unmatched = ~processed
        fallback_z, _, _ = robust_z_grid_filtered(sigma_grid, validity_mask)
        z[unmatched] = fallback_z[unmatched]

    # Dominant stats: overall valid-block statistics
    if validity_mask.any():
        valid_sigs = sigma_grid[validity_mask]
    else:
        valid_sigs = sigma_grid.ravel()
    dominant_sigma = float(np.median(valid_sigs))
    dominant_mad = float(np.median(np.abs(valid_sigs - dominant_sigma)) * 1.4826)
    if dominant_mad <= 1e-6:
        dominant_mad = float(np.std(valid_sigs) + 1e-6)

    return z.astype(np.float32, copy=False), dominant_sigma, dominant_mad


# ---------------------------------------------------------------------------
# Z-score computation — Gap 2: local neighbourhood comparison
# ---------------------------------------------------------------------------

def local_neighborhood_z_grid(
    sigma_grid: np.ndarray,
    neighborhood_k: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Compare each block's sigma to the median of its K×K spatial neighbours.

    Unlike global or region-aware references, this is inherently local: a
    splice boundary will stand out from its immediate neighbours even when the
    entire image has uniformly elevated noise.

    The center block is excluded from its own neighbourhood so it does not
    dilute the reference.  Uses ``np.lib.stride_tricks.sliding_window_view``
    (numpy ≥ 1.20) — no scipy required.

    Returns
    -------
    z_grid : float32 array  (same shape as sigma_grid)
    dominant_sigma : overall median of sigma_grid
    dominant_mad   : overall MAD × 1.4826 of sigma_grid
    """
    if sigma_grid.size == 0:
        return sigma_grid.copy().astype(np.float32), 0.0, 1.0

    h, w = sigma_grid.shape
    k = max(3, neighborhood_k | 1)   # ensure odd, minimum 3
    pad = k // 2

    padded = np.pad(sigma_grid.astype(np.float32), pad, mode="reflect")
    # windows shape: (H, W, k, k)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    flat = windows.reshape(h, w, -1)   # (H, W, k*k)

    # Exclude center element so the block doesn't compare to itself
    center_idx = k * k // 2
    neighbor_indices = [i for i in range(k * k) if i != center_idx]
    neighbors = flat[:, :, neighbor_indices]   # (H, W, k*k − 1)

    neighbor_median = np.median(neighbors, axis=-1).astype(np.float32)   # (H, W)
    abs_dev = np.abs(neighbors - neighbor_median[:, :, np.newaxis])
    neighbor_mad = np.median(abs_dev, axis=-1).astype(np.float32) * 1.4826  # (H, W)
    # Fallback: use std when MAD is near zero (very flat neighbourhood)
    flat_mask = neighbor_mad <= 1e-6
    if flat_mask.any():
        neighbor_std = np.std(neighbors, axis=-1).astype(np.float32)
        neighbor_mad = np.where(flat_mask, neighbor_std + 1e-6, neighbor_mad)

    z_grid = ((sigma_grid.astype(np.float32) - neighbor_median) / neighbor_mad)

    dominant_sigma = float(np.median(sigma_grid))
    dominant_mad = float(np.median(np.abs(sigma_grid - dominant_sigma)) * 1.4826)
    if dominant_mad <= 1e-6:
        dominant_mad = float(np.std(sigma_grid) + 1e-6)

    return z_grid.astype(np.float32), dominant_sigma, dominant_mad


# ---------------------------------------------------------------------------
# Grid alignment helpers (Phase 1)
# ---------------------------------------------------------------------------

def _align_validity_to_scale(ref_validity: np.ndarray, ratio: int) -> np.ndarray:
    """Downsample reference-scale validity mask to a coarser block scale.

    A coarser block is valid when ≥ 50 % of its constituent reference blocks
    are valid.
    """
    if ratio == 1:
        return ref_validity
    ref_h, ref_w = ref_validity.shape
    coarse_h, coarse_w = ref_h // ratio, ref_w // ratio
    coarse = np.zeros((coarse_h, coarse_w), dtype=bool)
    for r in range(coarse_h):
        for c in range(coarse_w):
            sub = ref_validity[
                r * ratio : min((r + 1) * ratio, ref_h),
                c * ratio : min((c + 1) * ratio, ref_w),
            ]
            coarse[r, c] = sub.sum() >= max(1, sub.size // 2)
    return coarse


def _align_region_type_to_scale(ref_region_type: np.ndarray, ratio: int) -> np.ndarray:
    """Downsample reference-scale region-type grid to a coarser scale using mode."""
    if ratio == 1:
        return ref_region_type
    ref_h, ref_w = ref_region_type.shape
    coarse_h, coarse_w = ref_h // ratio, ref_w // ratio
    coarse = np.full((coarse_h, coarse_w), 2, dtype=np.int32)  # default PHOTO
    for r in range(coarse_h):
        for c in range(coarse_w):
            sub = ref_region_type[
                r * ratio : min((r + 1) * ratio, ref_h),
                c * ratio : min((c + 1) * ratio, ref_w),
            ].ravel()
            if sub.size > 0:
                unique, counts = np.unique(sub, return_counts=True)
                coarse[r, c] = int(unique[np.argmax(counts)])
    return coarse


# ---------------------------------------------------------------------------
# Grid upsampling helpers (unchanged from original)
# ---------------------------------------------------------------------------

def upsample_grid(grid: np.ndarray, block_size: int, shape: tuple[int, int]) -> np.ndarray:
    if grid.size == 0:
        return np.zeros(shape, dtype=np.float32)
    upsampled = np.repeat(np.repeat(grid, block_size, axis=0), block_size, axis=1)
    return upsampled[: shape[0], : shape[1]].astype(np.float32, copy=False)


def align_grid_to_shape(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    aligned = np.zeros(shape, dtype=np.float32)
    if grid.size == 0:
        return aligned
    height = min(shape[0], grid.shape[0])
    width = min(shape[1], grid.shape[1])
    aligned[:height, :width] = grid[:height, :width]
    if height < shape[0]:
        aligned[height:, :width] = aligned[max(height - 1, 0), :width]
    if width < shape[1]:
        aligned[:, width:] = aligned[:, max(width - 1, 0) : max(width, 1)]
    return aligned


# ---------------------------------------------------------------------------
# SigmaAnalysis container (unchanged)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SigmaAnalysis:
    scales: list[SigmaScale]
    combined_sigma_map: np.ndarray
    combined_abs_z_map: np.ndarray
    reference_abs_z_grid: np.ndarray
    reference_sigma_grid: np.ndarray
    reference_block_size: int
    dominant_sigma: float
    dominant_mad: float


# ---------------------------------------------------------------------------
# Original analysis function (unchanged — kept for backward compat)
# ---------------------------------------------------------------------------

def analyze_sigma_maps(residual: np.ndarray, block_sizes: list[int]) -> SigmaAnalysis:
    """Original multi-scale analysis using all blocks as reference."""
    ordered = sorted(set(int(value) for value in block_sizes if int(value) > 0))
    scales: list[SigmaScale] = []
    for block_size in ordered:
        sigma_grid = compute_sigma_grid(residual, block_size)
        if sigma_grid.size == 0:
            continue
        z_grid, median_sigma, mad_sigma = robust_z_grid(sigma_grid)
        scales.append(
            SigmaScale(
                block_size=block_size,
                sigma_grid=sigma_grid,
                z_grid=z_grid,
                median_sigma=median_sigma,
                mad_sigma=mad_sigma,
            )
        )

    if not scales:
        raise ValueError("Input is too small for the configured block sizes.")

    reference = scales[0]
    reference_shape = reference.sigma_grid.shape
    reference_block_size = reference.block_size
    combined_abs_z_grid = np.zeros(reference_shape, dtype=np.float32)
    combined_sigma_grid = np.zeros(reference_shape, dtype=np.float32)
    counts = np.zeros(reference_shape, dtype=np.float32)

    for scale in scales:
        ratio = scale.block_size // reference_block_size
        expanded_abs_z = np.repeat(np.repeat(np.abs(scale.z_grid), ratio, axis=0), ratio, axis=1)
        expanded_sigma = np.repeat(np.repeat(scale.sigma_grid, ratio, axis=0), ratio, axis=1)
        expanded_abs_z = align_grid_to_shape(expanded_abs_z, reference_shape)
        expanded_sigma = align_grid_to_shape(expanded_sigma, reference_shape)
        combined_abs_z_grid = np.maximum(combined_abs_z_grid, expanded_abs_z)
        combined_sigma_grid += expanded_sigma
        counts += 1.0

    combined_sigma_grid = combined_sigma_grid / np.maximum(counts, 1.0)
    combined_sigma_map = upsample_grid(combined_sigma_grid, reference_block_size, residual.shape)
    combined_abs_z_map = upsample_grid(combined_abs_z_grid, reference_block_size, residual.shape)

    return SigmaAnalysis(
        scales=scales,
        combined_sigma_map=combined_sigma_map,
        combined_abs_z_map=combined_abs_z_map,
        reference_abs_z_grid=combined_abs_z_grid,
        reference_sigma_grid=combined_sigma_grid,
        reference_block_size=reference_block_size,
        dominant_sigma=reference.median_sigma,
        dominant_mad=reference.mad_sigma,
    )


# ---------------------------------------------------------------------------
# Phase-2: Intensity-aware noise level function (NLF)
# ---------------------------------------------------------------------------

def compute_block_mean_intensity(gray: np.ndarray, block_size: int) -> np.ndarray:
    """Mean intensity per block. Returns float32 grid (grid_h, grid_w)."""
    grid_h = gray.shape[0] // block_size
    grid_w = gray.shape[1] // block_size
    intensity = np.zeros((grid_h, grid_w), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            y0, x0 = r * block_size, c * block_size
            intensity[r, c] = float(np.mean(gray[y0 : y0 + block_size, x0 : x0 + block_size]))
    return intensity


def estimate_noise_level_function(
    sigma_grid: np.ndarray,
    intensity_grid: np.ndarray,
    validity_mask: np.ndarray,
    n_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a piecewise-constant noise level function: σ_expected = NLF(intensity).

    Bins valid blocks by mean intensity and estimates the median sigma in
    each bin.  Returns ``(bin_centers, bin_sigmas)`` as float64 arrays.

    This captures the Poisson-like noise dependence on intensity: darker
    regions of a photo naturally have different noise than bright regions.
    Without NLF modelling the detector mistakes this for tampering.
    """
    valid_sigs = sigma_grid[validity_mask].ravel()
    valid_int = intensity_grid[validity_mask].ravel()

    if valid_sigs.size == 0:
        # Fallback: flat NLF at overall median sigma
        bin_centers = np.linspace(0.0, 255.0, n_bins, dtype=np.float64)
        fallback = float(np.median(sigma_grid))
        return bin_centers, np.full(n_bins, fallback, dtype=np.float64)

    edges = np.linspace(0.0, 255.0 + 1e-6, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) * 0.5
    bin_sigmas = np.zeros(n_bins, dtype=np.float64)
    overall_median = float(np.median(valid_sigs))

    for i in range(n_bins):
        mask = (valid_int >= edges[i]) & (valid_int < edges[i + 1])
        if mask.sum() >= 3:
            bin_sigmas[i] = float(np.median(valid_sigs[mask]))
        else:
            bin_sigmas[i] = overall_median

    return bin_centers, bin_sigmas


def intensity_aware_z_grid(
    sigma_grid: np.ndarray,
    intensity_grid: np.ndarray,
    validity_mask: np.ndarray,
    n_bins: int = 8,
) -> tuple[np.ndarray, float, float]:
    """Z-scores relative to NLF-expected sigma at each block's intensity level.

    For each block:
        z = (σ_block − NLF(intensity_block)) / local_MAD

    This prevents blocks in dark image regions from being flagged as anomalous
    simply because their sigma differs from the sigma in bright regions.

    Returns
    -------
    z_grid : float32 array
    dominant_sigma : NLF median (median of bin_sigmas)
    dominant_mad : MAD of sigma residuals (σ_block − NLF) for valid blocks
    """
    bin_centers, bin_sigmas = estimate_noise_level_function(
        sigma_grid, intensity_grid, validity_mask, n_bins
    )

    # Vectorised nearest-bin lookup
    intensity_flat = intensity_grid.ravel().astype(np.float64)
    distances = np.abs(intensity_flat[:, None] - bin_centers[None, :])
    nearest_bin = np.argmin(distances, axis=1).reshape(sigma_grid.shape)
    expected_sigma = bin_sigmas[nearest_bin].astype(np.float32)

    residuals = sigma_grid - expected_sigma

    # MAD of residuals on valid blocks
    valid_residuals = residuals[validity_mask] if validity_mask.any() else residuals.ravel()
    if valid_residuals.size == 0:
        valid_residuals = residuals.ravel()
    mad = float(np.median(np.abs(valid_residuals)) * 1.4826)
    if mad <= 1e-6:
        mad = float(np.std(valid_residuals) + 1e-6)

    z = residuals / mad
    dominant_sigma = float(np.median(bin_sigmas))
    return z.astype(np.float32, copy=False), dominant_sigma, mad


# ---------------------------------------------------------------------------
# Phase-1+2: Enhanced multi-scale analysis
# ---------------------------------------------------------------------------

def analyze_sigma_maps_v2(
    residual: np.ndarray,
    block_sizes: list[int],
    validity_mask: np.ndarray | None = None,
    region_type_grid: np.ndarray | None = None,
    intensity_aware: bool = False,
    nlf_bins: int = 8,
    gray: np.ndarray | None = None,
    consensus_z_threshold: float = 1.5,
    local_neighborhood_z: bool = False,
    neighborhood_k: int = 5,
) -> SigmaAnalysis:
    """Enhanced multi-scale analysis with validity filtering and region awareness.

    Enhancements over ``analyze_sigma_maps``:

    * If *validity_mask* is provided, z-scores are referenced only to valid
      blocks (suppresses structure-driven false z-scores).
    * If *region_type_grid* is also provided, each block is compared to its
      per-region-type reference (photo vs. background vs. text).
    * If *intensity_aware* is True and *gray* is provided, z-scores are
      referenced to NLF(intensity) rather than a global median.
    * Validity and region-type grids are downsampled consistently when
      processing larger block sizes.
    """
    from noise_level_inconsistency.region_classifier import compute_per_region_reference

    ordered = sorted(set(int(value) for value in block_sizes if int(value) > 0))
    if not ordered:
        raise ValueError("block_sizes must not be empty.")

    ref_block_size = ordered[0]
    scales: list[SigmaScale] = []

    for block_size in ordered:
        sigma_grid = compute_sigma_grid(residual, block_size)
        if sigma_grid.size == 0:
            continue

        ratio = block_size // ref_block_size

        # Align validity mask to this scale
        scale_validity: np.ndarray | None = None
        if validity_mask is not None:
            scale_validity = _align_validity_to_scale(validity_mask, ratio)
            # Safety: ensure shape matches sigma grid
            if scale_validity.shape != sigma_grid.shape:
                scale_validity = None

        # Align region-type grid to this scale
        scale_regions: np.ndarray | None = None
        if region_type_grid is not None:
            scale_regions = _align_region_type_to_scale(region_type_grid, ratio)
            if scale_regions.shape != sigma_grid.shape:
                scale_regions = None

        # Choose primary z-score strategy (priority: intensity > region > filtered > global)
        if intensity_aware and gray is not None:
            intensity_grid = compute_block_mean_intensity(gray, block_size)
            sv = scale_validity if scale_validity is not None else np.ones(sigma_grid.shape, dtype=bool)
            z_grid, median_sigma, mad_sigma = intensity_aware_z_grid(
                sigma_grid, intensity_grid, sv, nlf_bins
            )
        elif scale_validity is not None and scale_regions is not None:
            per_region_ref = compute_per_region_reference(sigma_grid, scale_regions, scale_validity)
            z_grid, median_sigma, mad_sigma = robust_z_grid_region_aware(
                sigma_grid, scale_regions, scale_validity, per_region_ref
            )
        elif scale_validity is not None:
            z_grid, median_sigma, mad_sigma = robust_z_grid_filtered(sigma_grid, scale_validity)
        else:
            z_grid, median_sigma, mad_sigma = robust_z_grid(sigma_grid)

        # Gap 2: supplement with local neighbourhood z — keep whichever value has
        # the higher absolute magnitude.  np.maximum on signed values would turn a
        # valid negative primary z (e.g. -3.3 for a low-noise patch in a high-noise
        # host) into 0 when local_z is 0, destroying the primary signal before the
        # abs() step in the combination loop.
        if local_neighborhood_z:
            local_z, _, _ = local_neighborhood_z_grid(sigma_grid, neighborhood_k)
            z_grid = np.where(np.abs(z_grid) >= np.abs(local_z), z_grid, local_z)

        scales.append(
            SigmaScale(
                block_size=block_size,
                sigma_grid=sigma_grid,
                z_grid=z_grid,
                median_sigma=median_sigma,
                mad_sigma=mad_sigma,
            )
        )

    if not scales:
        raise ValueError("Input is too small for the configured block sizes.")

    # Combine scales — Gap 1: consensus-weighted max instead of plain max.
    # Each block is weighted by the fraction of scales that flagged it as
    # anomalous (z ≥ consensus_z_threshold).  A JPEG artifact visible at only
    # one scale gets weight ≈ 0.3+0.7*(1/3) ≈ 0.53; a real splice visible at
    # all three gets weight = 1.0.
    reference = scales[0]
    reference_shape = reference.sigma_grid.shape
    reference_block_size_used = reference.block_size
    max_abs_z_grid = np.zeros(reference_shape, dtype=np.float32)
    agreement_count_grid = np.zeros(reference_shape, dtype=np.float32)
    combined_sigma_grid = np.zeros(reference_shape, dtype=np.float32)
    counts = np.zeros(reference_shape, dtype=np.float32)

    for scale in scales:
        ratio = scale.block_size // reference_block_size_used
        expanded_abs_z = np.repeat(np.repeat(np.abs(scale.z_grid), ratio, axis=0), ratio, axis=1)
        expanded_sigma = np.repeat(np.repeat(scale.sigma_grid, ratio, axis=0), ratio, axis=1)
        expanded_abs_z = align_grid_to_shape(expanded_abs_z, reference_shape)
        expanded_sigma = align_grid_to_shape(expanded_sigma, reference_shape)
        max_abs_z_grid = np.maximum(max_abs_z_grid, expanded_abs_z)
        agreement_count_grid += (expanded_abs_z >= consensus_z_threshold).astype(np.float32)
        combined_sigma_grid += expanded_sigma
        counts += 1.0

    n_scales = float(len(scales))
    consensus_weight = 0.3 + 0.7 * (agreement_count_grid / max(n_scales, 1.0))
    combined_abs_z_grid = (max_abs_z_grid * consensus_weight).astype(np.float32)

    combined_sigma_grid = combined_sigma_grid / np.maximum(counts, 1.0)
    combined_sigma_map = upsample_grid(combined_sigma_grid, reference_block_size_used, residual.shape)
    combined_abs_z_map = upsample_grid(combined_abs_z_grid, reference_block_size_used, residual.shape)

    return SigmaAnalysis(
        scales=scales,
        combined_sigma_map=combined_sigma_map,
        combined_abs_z_map=combined_abs_z_map,
        reference_abs_z_grid=combined_abs_z_grid,
        reference_sigma_grid=combined_sigma_grid,
        reference_block_size=reference_block_size_used,
        dominant_sigma=reference.median_sigma,
        dominant_mad=reference.mad_sigma,
    )
