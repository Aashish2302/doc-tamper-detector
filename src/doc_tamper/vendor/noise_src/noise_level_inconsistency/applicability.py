"""Applicability gating for NLI.

Before running the full NLI pipeline, this module assesses whether the
input image is suitable for noise-level inconsistency analysis and assigns
a reliability score.  Low-reliability inputs are flagged so analysts know
the NLI evidence is weak or unreliable.

Gate checks (each produces a reason_code when triggered and reduces reliability):

  too_small                Image is too small for multi-scale block analysis.
  insufficient_valid_blocks Most blocks are structure-dominated (edge/text);
                            not enough clean-noise canvas for estimation.
  too_blurry               Laplacian variance is too low → image lacks
                            high-frequency noise signal (heavy blur / heavy
                            JPEG compression removed noise floor).
  near_zero_noise          Median residual is near zero → likely a rasterised
                            vector PDF with no acquisition noise at all.
  mostly_flat_background   >85 % of blocks are near-flat; hard to distinguish
                            noise inconsistency from natural uniformity.

reliability ∈ [0, 1].  is_applicable = True if reliability ≥ 0.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_level_inconsistency.residuals import laplacian_residual


@dataclass(slots=True)
class ApplicabilityResult:
    is_applicable: bool
    reliability: float
    reason_codes: list[str]
    valid_block_fraction: float


def assess_applicability(
    gray: np.ndarray,
    validity_mask: np.ndarray,
    min_valid_block_fraction: float = 0.15,
    blur_variance_threshold: float = 50.0,
    min_noise_sigma: float = 0.5,
    min_dimension: int = 64,
) -> ApplicabilityResult:
    """Return an :class:`ApplicabilityResult` describing NLI suitability.

    Parameters
    ----------
    gray:
        Float32 grayscale image array.
    validity_mask:
        Boolean grid (grid_h × grid_w) produced by patch_validity; True = valid.
    min_valid_block_fraction:
        Minimum fraction of valid blocks required for reliable analysis.
    blur_variance_threshold:
        Minimum Laplacian variance; images below this are considered too blurry.
    min_noise_sigma:
        Minimum median Laplacian value; below this the image likely has no
        acquisition noise (rasterised vector content).
    min_dimension:
        Minimum image dimension (height or width) in pixels.
    """
    reason_codes: list[str] = []
    reliability = 1.0

    # Check 1: image too small
    if gray.shape[0] < min_dimension or gray.shape[1] < min_dimension:
        reason_codes.append("too_small")
        reliability -= 0.40

    # Check 2: too few valid blocks (near-all blocks are structural)
    total_blocks = max(validity_mask.size, 1)
    valid_count = int(validity_mask.sum())
    valid_fraction = valid_count / total_blocks
    if valid_fraction < min_valid_block_fraction:
        reason_codes.append("insufficient_valid_blocks")
        reliability -= 0.35

    # Check 3: too blurry — low Laplacian variance means no high-frequency content
    lap = laplacian_residual(gray)
    lap_variance = float(np.var(lap))
    if lap_variance < blur_variance_threshold:
        reason_codes.append("too_blurry")
        reliability -= 0.25

    # Check 4: near-zero noise floor (rasterised vector PDF)
    median_lap = float(np.median(lap))
    if median_lap < min_noise_sigma:
        reason_codes.append("near_zero_noise")
        reliability -= 0.30

    # Check 5: mostly flat background
    flat_fraction = 1.0 - valid_fraction
    if flat_fraction > 0.85 and "insufficient_valid_blocks" not in reason_codes:
        reason_codes.append("mostly_flat_background")
        reliability -= 0.15

    reliability = float(max(0.0, min(1.0, reliability)))
    is_applicable = reliability >= 0.3

    return ApplicabilityResult(
        is_applicable=is_applicable,
        reliability=reliability,
        reason_codes=reason_codes,
        valid_block_fraction=valid_fraction,
    )
