"""Anomaly detection, region extraction, and scoring for NLI.

Phase-1 changes
---------------
* ``detect_candidate_regions`` now accepts an optional *validity_mask* so
  that blocks dominated by document structure (text, edges) cannot be flagged
  as anomalous even if their z-score is high.

Phase-2 change
--------------
* ``smooth_z_grid`` applies a small box-filter to the combined z-score grid
  before thresholding, eliminating isolated single-block noise hits and
  reducing checkerboard artefacts from the multi-scale ``np.maximum`` combine.

Phase-3 changes
---------------
* ``_region_score_v2`` replaces the old linear-clamp formula with a
  sigmoid (1 − e^−kx) mapping over excess z-score.  The old formula
  saturated at 1.0 for virtually every input; the sigmoid spreads scores
  meaningfully across the range [0, 1].
* Per-region *reason_codes* are attached to each ``CandidateRegionStats``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from noise_level_inconsistency.config import MethodConfig
from noise_level_inconsistency.models import CandidateRegionStats
from noise_level_inconsistency.sigma import SigmaAnalysis


# ---------------------------------------------------------------------------
# Connected-component analysis (unchanged)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Component:
    cells: list[tuple[int, int]]

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rows = [row for row, _ in self.cells]
        cols = [col for _, col in self.cells]
        return min(rows), min(cols), max(rows), max(cols)


def connected_components(mask: np.ndarray) -> list[Component]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[Component] = []
    height, width = mask.shape

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if 0 <= next_row < height and 0 <= next_col < width:
                            if mask[next_row, next_col] and not visited[next_row, next_col]:
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
            components.append(Component(cells=cells))
    return components


def _bbox_gap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> tuple[int, int]:
    first_y0, first_x0, first_y1, first_x1 = first
    second_y0, second_x0, second_y1, second_x1 = second
    vertical_gap = max(0, max(second_y0 - first_y1 - 1, first_y0 - second_y1 - 1))
    horizontal_gap = max(0, max(second_x0 - first_x1 - 1, first_x0 - second_x1 - 1))
    return vertical_gap, horizontal_gap


def merge_components(
    components: Iterable[Component],
    gap_blocks: int,
    max_bbox_fraction: float = 1.0,
    grid_shape: tuple[int, int] | None = None,
) -> list[Component]:
    """Merge spatially nearby components.

    max_bbox_fraction: if the merged bbox would cover more than this fraction
    of grid_shape, skip the merge (keeps scattered anomalies as separate
    components instead of one giant blob).  Default 1.0 = no limit.
    """
    grid_area = (grid_shape[0] * grid_shape[1]) if grid_shape is not None else 0
    check_fraction = (
        grid_area > 0
        and max_bbox_fraction < 1.0
    )

    merged = list(components)
    changed = True
    while changed:
        changed = False
        next_components: list[Component] = []
        while merged:
            current = merged.pop(0)
            current_cells = set(current.cells)
            current_bbox = current.bbox
            remainder: list[Component] = []
            for candidate in merged:
                vertical_gap, horizontal_gap = _bbox_gap(current_bbox, candidate.bbox)
                if vertical_gap <= gap_blocks and horizontal_gap <= gap_blocks:
                    # Preview what the merged bbox would be
                    merged_cells = current_cells | set(candidate.cells)
                    rows = [row for row, _ in merged_cells]
                    cols = [col for _, col in merged_cells]
                    candidate_bbox = min(rows), min(cols), max(rows), max(cols)

                    if check_fraction:
                        bbox_h = candidate_bbox[2] - candidate_bbox[0] + 1
                        bbox_w = candidate_bbox[3] - candidate_bbox[1] + 1
                        if (bbox_h * bbox_w) / grid_area > max_bbox_fraction:
                            remainder.append(candidate)
                            continue  # refuse this merge

                    current_cells = merged_cells
                    current_bbox = candidate_bbox
                    changed = True
                else:
                    remainder.append(candidate)
            merged = remainder
            next_components.append(Component(cells=sorted(current_cells)))
        merged = next_components
    return merged


def _component_mask(shape: tuple[int, int], cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for row, col in cells:
        mask[row, col] = True
    return mask


def _pixel_bbox(component: Component, block_size: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    row0, col0, row1, col1 = component.bbox
    x0 = col0 * block_size
    y0 = row0 * block_size
    x1 = min(image_shape[1], (col1 + 1) * block_size)
    y1 = min(image_shape[0], (row1 + 1) * block_size)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Phase-2: Z-grid smoothing
# ---------------------------------------------------------------------------

def smooth_z_grid(z_grid: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a box-filter mean to the z-score grid to reduce artefacts.

    The multi-scale ``np.maximum`` combination can leave isolated high-z
    blocks that are just statistical noise.  A small mean filter (default
    3×3) averages out single-block spikes without significantly blurring
    real tampering regions.

    ``radius=0`` is a no-op; ``radius=1`` gives a 3×3 mean filter.
    """
    if radius <= 0 or z_grid.size == 0:
        return z_grid

    ksize = 2 * radius + 1
    padded = np.pad(z_grid.astype(np.float64), radius, mode="reflect")
    pcs = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    pcs[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h, w = z_grid.shape
    result = (
        pcs[ksize : ksize + h, ksize : ksize + w]
        - pcs[0:h, ksize : ksize + w]
        - pcs[ksize : ksize + h, 0:w]
        + pcs[0:h, 0:w]
    ) / float(ksize * ksize)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase-3: Calibrated region scoring
# ---------------------------------------------------------------------------

def _region_score_v2(
    mean_abs_z: float,
    area_blocks: int,
    total_valid_blocks: int,
    threshold: float,
    mismatch: float,
) -> tuple[float, list[str]]:
    """Calibrated region score using a sigmoid over excess z-score.

    Old formula: ``min(1.0, mean_z / (threshold × 1.75))``
    → saturates at z = 4.375, which most document blocks exceed, giving
    score = 1.0 for everything.

    New formula:
        severity    = 1 − exp(−0.2 × (mean_z − threshold))
        size_factor = log₂(area + 1) / log₂(total_valid + 1)  [log scale]
        mismatch_f  = min(1, mismatch / (threshold × 2))
        score       = 0.55 × severity + 0.25 × size_factor + 0.20 × mismatch_f

    Score at key z-values (threshold = 2.5):
        z = 3.0  → severity ≈ 0.095  → score ≈ 0.05 + size + mismatch terms
        z = 5.0  → severity ≈ 0.393  → score ≈ 0.22 + …
        z = 10.0 → severity ≈ 0.777  → score ≈ 0.43 + …
        z = 20.0 → severity ≈ 0.970  → score ≈ 0.53 + …

    Returns (score, reason_codes).
    """
    reason_codes: list[str] = []

    if mean_abs_z <= threshold:
        return 0.0, reason_codes

    # Sigmoid over excess z
    excess = mean_abs_z - threshold
    severity = float(1.0 - np.exp(-0.2 * excess))
    reason_codes.append(f"mean_abs_z={mean_abs_z:.2f}")

    # Log-scale size factor (prevents small isolated hits from scoring high)
    if total_valid_blocks > 1:
        size_factor = float(
            min(1.0, np.log2(area_blocks + 1) / max(np.log2(total_valid_blocks + 1), 1.0))
        )
    else:
        size_factor = float(min(1.0, area_blocks / 20.0))

    if size_factor < 0.05:
        reason_codes.append("small_region")

    # Mismatch factor: how different is this region's sigma from the reference?
    mismatch_factor = float(min(1.0, mismatch / max(threshold * 2.0, 1e-6)))
    if mismatch_factor > 0.5:
        reason_codes.append("noise_curve_mismatch")

    score = 0.55 * severity + 0.25 * size_factor + 0.20 * mismatch_factor
    return float(max(0.0, min(1.0, score))), reason_codes


# ---------------------------------------------------------------------------
# Legacy scoring (kept for reference; no longer called by default)
# ---------------------------------------------------------------------------

def _region_score(
    mean_abs_z: float,
    area_blocks: int,
    threshold: float,
    min_component_blocks: int,
    mismatch: float,
    advanced_mode: bool,
) -> float:
    severity = min(1.0, mean_abs_z / max(threshold * 1.75, 1e-6))
    size_factor = min(1.0, area_blocks / max(min_component_blocks * 4, 1))
    score = 0.7 * severity + 0.3 * size_factor
    if advanced_mode:
        score = 0.55 * score + 0.45 * min(1.0, mismatch / max(threshold, 1e-6))
    return float(min(1.0, score))


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

def _align_validity_for_detection(
    validity_mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize validity mask to match the reference grid shape via nearest-neighbour."""
    if validity_mask.shape == target_shape:
        return validity_mask

    src_h, src_w = validity_mask.shape
    tgt_h, tgt_w = target_shape

    if src_h == 0 or src_w == 0:
        return np.zeros(target_shape, dtype=bool)

    # Nearest-neighbour rescale via integer index lookup
    row_idx = np.clip(
        (np.arange(tgt_h) * src_h / tgt_h).astype(int), 0, src_h - 1
    )
    col_idx = np.clip(
        (np.arange(tgt_w) * src_w / tgt_w).astype(int), 0, src_w - 1
    )
    return validity_mask[np.ix_(row_idx, col_idx)]


def detect_candidate_regions(
    analysis: SigmaAnalysis,
    config: MethodConfig,
    validity_mask: np.ndarray | None = None,
    channel_agreement_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[CandidateRegionStats]]:
    """Detect anomalous regions in the z-score grid.

    Phase-1: if *validity_mask* is supplied, only valid blocks can be flagged
    as anomalous.  Blocks dominated by text/edges/structure are excluded from
    detection even if their z-score is high, preventing structure-driven FPs.

    Phase-2: the combined z-grid is smoothed before thresholding (controlled
    by ``config.smooth_z_grid_radius``).

    Phase-3: uses the calibrated sigmoid scoring formula.
    """
    # Phase-2: smooth z-grid to reduce isolated noise hits
    z_grid = analysis.reference_abs_z_grid
    smooth_radius = getattr(config, "smooth_z_grid_radius", 1)
    if smooth_radius > 0:
        z_grid = smooth_z_grid(z_grid, smooth_radius)

    # Phase-1: threshold, optionally masked to valid blocks only
    candidate_grid = z_grid >= config.anomaly_z_threshold
    if validity_mask is not None and validity_mask.size > 0:
        aligned_validity = _align_validity_for_detection(validity_mask, candidate_grid.shape)
        candidate_grid = candidate_grid & aligned_validity

    # Gap 4: per-channel agreement filter — suppress blocks not agreed upon by ≥N channels
    if channel_agreement_mask is not None and channel_agreement_mask.size > 0:
        aligned_ch = _align_validity_for_detection(
            channel_agreement_mask.astype(bool), candidate_grid.shape
        )
        candidate_grid = candidate_grid & aligned_ch

    components = connected_components(candidate_grid)
    components = merge_components(
        components,
        config.merge_gap_blocks,
        max_bbox_fraction=getattr(config, "max_region_bbox_fraction", 1.0),
        grid_shape=z_grid.shape,
    )

    # Total valid blocks for size_factor denominator
    total_valid_blocks = int(validity_mask.sum()) if validity_mask is not None else int(z_grid.size)

    anomaly_mask = np.zeros_like(analysis.combined_abs_z_map, dtype=bool)
    regions: list[CandidateRegionStats] = []

    for component in components:
        if component.area < config.min_component_blocks:
            continue

        grid_mask = _component_mask(z_grid.shape, component.cells)
        sigma_values = analysis.reference_sigma_grid[grid_mask]
        abs_z_values = z_grid[grid_mask]

        mismatch = float(
            abs(float(np.mean(sigma_values)) - analysis.dominant_sigma)
            / max(analysis.dominant_mad, 1e-6)
        )

        # Phase-3: calibrated scoring
        score, reason_codes = _region_score_v2(
            mean_abs_z=float(np.mean(abs_z_values)),
            area_blocks=component.area,
            total_valid_blocks=max(total_valid_blocks, 1),
            threshold=config.anomaly_z_threshold,
            mismatch=mismatch,
        )

        if score < 0.05:
            continue

        x0, y0, x1, y1 = _pixel_bbox(
            component,
            analysis.reference_block_size,
            analysis.combined_abs_z_map.shape,
        )
        expand = max(0, config.region_expand_blocks * analysis.reference_block_size)
        x0 = max(0, x0 - expand)
        y0 = max(0, y0 - expand)
        x1 = min(analysis.combined_abs_z_map.shape[1], x1 + expand)
        y1 = min(analysis.combined_abs_z_map.shape[0], y1 + expand)

        # Fill only the actual anomalous blocks (not the bounding-box rectangle).
        # This prevents a component with scattered cells from painting the entire
        # bounding-box area red when its cells are spread across the image.
        bs = analysis.reference_block_size
        img_h, img_w = analysis.combined_abs_z_map.shape[:2]
        for row, col in component.cells:
            py0 = row * bs
            py1 = min(img_h, (row + 1) * bs)
            px0 = col * bs
            px1 = min(img_w, (col + 1) * bs)
            anomaly_mask[py0:py1, px0:px1] = True

        regions.append(
            CandidateRegionStats(
                bbox=(x0, y0, x1, y1),
                score=score,
                area_blocks=component.area,
                area_pixels=int(anomaly_mask[y0:y1, x0:x1].sum()),
                mean_sigma=float(np.mean(sigma_values)),
                max_sigma=float(np.max(sigma_values)),
                mean_abs_z=float(np.mean(abs_z_values)),
                max_abs_z=float(np.max(abs_z_values)),
                mismatch=mismatch,
                reason_codes=reason_codes,
                region_type="unknown",
            )
        )

    regions.sort(key=lambda item: item.score, reverse=True)
    return anomaly_mask, regions
