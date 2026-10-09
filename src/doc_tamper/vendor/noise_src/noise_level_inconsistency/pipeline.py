"""Main inference pipeline for NLI.

All three improvement phases are wired here:

Phase 1 — Honest & stable
  * Patch validity mask excludes structure-dominated blocks from analysis.
  * Region-type classifier separates photo/text/background noise references.
  * Applicability gate rejects unsuitable inputs early with explicit notes.

Phase 2 — Stronger signal
  * Denoiser residual suppresses edge response (default residual mode).
  * Intensity-aware NLF prevents intensity-driven false positives.
  * Z-grid smoothing removes isolated single-block noise hits.

Phase 3 — Calibrated scoring
  * Sigmoid scoring formula avoids saturation at 1.0.
  * Fallback score is near 0 when no anomalous regions are found (honest).
  * Confidence is modulated by the applicability reliability score.
  * ``nli_applicable`` and ``reliability`` emitted in both notes and the
    top-level result fields.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from noise_level_inconsistency.config import AppConfig
from noise_level_inconsistency.contracts import ArtifactBundle, Decision, MethodResult, RegionResult
from noise_level_inconsistency.detection import detect_candidate_regions
from noise_level_inconsistency.io import UnsupportedInputError, discover_inputs, load_prepared_input
from noise_level_inconsistency.models import CandidateRegionStats, PreparedInput
from noise_level_inconsistency.residuals import extract_residual
from noise_level_inconsistency.sigma import analyze_sigma_maps, analyze_sigma_maps_v2
from noise_level_inconsistency.utils import ensure_dir, get_logger, portable_path, write_json
from noise_level_inconsistency.visualization import (
    build_overlay_image,
    save_debug_panel,
    save_image,
    save_overlay,
    to_grayscale_image,
    to_heatmap_image,
)


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class InferenceArtifacts:
    result: MethodResult
    noise_map_path: Path
    zscore_heatmap_path: Path
    overlay_path: Path
    mask_path: Path
    region_csv_path: Path
    debug_panel_path: Path | None
    residual_map_path: Path | None


# ---------------------------------------------------------------------------
# Decision and confidence helpers (Phase 3)
# ---------------------------------------------------------------------------

def _decision_from_score(score: float, config: AppConfig) -> str:
    if score >= config.method.tampered_score_threshold:
        return Decision.TAMPERED.value
    if score >= config.method.suspicious_score_threshold:
        return Decision.SUSPICIOUS.value
    return Decision.AUTHENTIC.value


def _confidence_from_score(
    score: float,
    region_count: int,
    reliability: float,
    valid_block_fraction: float,
) -> float:
    """Confidence is modulated by reliability and coverage, not just score.

    Phase-3 fix: old formula was ``min(1, score + 0.1)`` which gave confidence
    near 1.0 whenever score was high — independent of whether the analysis was
    actually trustworthy.  New formula gates confidence on:
      * ``reliability``         — from the applicability gate
      * ``coverage``            — how much of the image had valid noise blocks
      * small bonus for multiple independent suspicious regions
    """
    # Coverage: full confidence when ≥ 50 % of blocks are valid
    coverage = float(min(1.0, valid_block_fraction / 0.5)) if valid_block_fraction > 0 else 0.5
    base = score * reliability * coverage

    if region_count >= 3:
        base = min(1.0, base + 0.05)
    elif region_count >= 2:
        base = min(1.0, base + 0.02)

    return float(max(0.01, min(1.0, base)))


# ---------------------------------------------------------------------------
# Phase-3: honest fallback score
# ---------------------------------------------------------------------------

def _compute_fallback_score(analysis, anomaly_z_threshold: float) -> float:
    """Score when no anomalous regions were detected.

    Old behaviour: return up to 0.2, which inflated scores for clean images.
    New behaviour: return near 0; only give a tiny signal when the 95th-
    percentile z-score is very close to the threshold (borderline cases).
    """
    p95_z = float(np.percentile(analysis.combined_abs_z_map, 95))
    # Ramp from 0.0 at 70 % of threshold to 0.08 at threshold
    fraction = p95_z / max(anomaly_z_threshold, 1e-6)
    if fraction < 0.70:
        return 0.0
    return float(min(0.08, (fraction - 0.70) * 0.267))   # linear up to 0.08


# ---------------------------------------------------------------------------
# CSV output (updated for new CandidateRegionStats fields)
# ---------------------------------------------------------------------------

def _write_region_csv(path: Path, regions: list[CandidateRegionStats]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
                "score",
                "area_blocks",
                "area_pixels",
                "mean_sigma",
                "max_sigma",
                "mean_abs_z",
                "max_abs_z",
                "mismatch",
                "region_type",
                "reason_codes",
            ]
        )
        for region in regions:
            writer.writerow(
                [
                    region.bbox[0],
                    region.bbox[1],
                    region.bbox[2],
                    region.bbox[3],
                    f"{region.score:.6f}",
                    region.area_blocks,
                    region.area_pixels,
                    f"{region.mean_sigma:.6f}",
                    f"{region.max_sigma:.6f}",
                    f"{region.mean_abs_z:.6f}",
                    f"{region.max_abs_z:.6f}",
                    f"{region.mismatch:.6f}",
                    region.region_type,
                    "|".join(region.reason_codes),
                ]
            )
    return path


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(
    prepared: PreparedInput,
    score: float,
    regions: list[CandidateRegionStats],
    artifact_bundle: ArtifactBundle,
    runtime_ms: int,
    notes: list[str],
    config: AppConfig,
    unsupported: bool = False,
    reliability: float = 1.0,
    nli_applicable: bool = True,
) -> MethodResult:
    decision = Decision.UNSUPPORTED.value if unsupported else _decision_from_score(score, config)

    valid_block_fraction = 1.0
    for note in notes:
        if note.startswith("valid_block_fraction="):
            try:
                valid_block_fraction = float(note.split("=", 1)[1])
            except ValueError:
                pass

    confidence = (
        1.0
        if unsupported
        else _confidence_from_score(score, len(regions), reliability, valid_block_fraction)
    )

    region_results = [
        RegionResult(
            bbox=list(region.bbox),
            score=region.score,
            label="noise_inconsistency",
            reason_codes=region.reason_codes,
        )
        for region in regions
    ]

    return MethodResult(
        method="noise_level_inconsistency",
        doc_id=prepared.doc_id,
        input_path=prepared.relative_input_path,
        score=score,
        decision=decision,
        confidence=confidence,
        regions=region_results,
        artifacts=artifact_bundle,
        runtime_ms=runtime_ms,
        notes=notes,
        nli_applicable=nli_applicable,
        reliability=reliability,
    )


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer_prepared_input(
    prepared: PreparedInput,
    config: AppConfig,
    output_root: str | Path | None = None,
    save_debug_panel_only: bool = False,
) -> InferenceArtifacts:
    start = perf_counter()
    output_dir = ensure_dir(Path(output_root or config.output.root_dir) / prepared.doc_id)
    notes = list(prepared.notes)
    notes.append(f"residual_mode={config.method.residual_mode}")
    notes.append(f"block_sizes={','.join(str(size) for size in config.method.block_sizes)}")
    notes.append(f"advanced_mode={str(config.method.advanced_mode).lower()}")

    ref_block_size = config.method.block_sizes[0] if config.method.block_sizes else 8

    # -----------------------------------------------------------------------
    # Phase 1: Patch validity mask
    # -----------------------------------------------------------------------
    validity_mask: np.ndarray | None = None
    region_type_grid: np.ndarray | None = None
    reliability = 1.0
    nli_applicable = True
    valid_block_fraction = 1.0

    if config.method.patch_validity_enabled:
        from noise_level_inconsistency.patch_validity import (
            compute_block_validity_mask,
            compute_edge_strength,
            compute_flat_region_mask,
            compute_local_variance,
        )

        edge_strength = compute_edge_strength(prepared.gray)
        local_variance = compute_local_variance(prepared.gray)

        validity_mask = compute_block_validity_mask(
            prepared.gray,
            ref_block_size,
            structural_blur_radius=config.method.structural_blur_radius,
            structural_var_percentile=config.method.structural_var_percentile,
        )
        flat_mask = compute_flat_region_mask(
            prepared.gray,
            ref_block_size,
            flatness_threshold=config.method.flatness_threshold,
        )
        # Exclude both structural-heavy AND too-flat blocks
        validity_mask = validity_mask & ~flat_mask

        # Phase 1: Region-aware sigma references
        if config.method.region_aware_sigma:
            from noise_level_inconsistency.region_classifier import classify_blocks

            region_type_grid = classify_blocks(
                prepared.gray, ref_block_size, edge_strength, local_variance
            )

        # Phase 1: Applicability gate
        if config.method.applicability_enabled:
            from noise_level_inconsistency.applicability import assess_applicability

            applicability = assess_applicability(
                prepared.gray,
                validity_mask,
                min_valid_block_fraction=config.method.min_valid_block_fraction,
                blur_variance_threshold=config.method.blur_variance_threshold,
                min_noise_sigma=config.method.min_noise_sigma,
                min_dimension=config.preprocess.min_dimension,
            )
            reliability = applicability.reliability
            nli_applicable = applicability.is_applicable
            valid_block_fraction = applicability.valid_block_fraction

            notes.append(f"nli_applicable={nli_applicable}")
            notes.append(f"nli_reliability={reliability:.2f}")
            notes.append(f"valid_block_fraction={valid_block_fraction:.2f}")
            for code in applicability.reason_codes:
                notes.append(f"applicability_reason={code}")

            if not nli_applicable:
                LOGGER.info(
                    "NLI not applicable for %s (reliability=%.2f, reasons=%s)",
                    prepared.doc_id,
                    reliability,
                    applicability.reason_codes,
                )
                # Build a low-score unsupported-like result without full analysis
                runtime_ms = int((perf_counter() - start) * 1000)
                blank_bundle = ArtifactBundle(debug_dir=portable_path(output_dir, Path.cwd()))
                result = _build_result(
                    prepared,
                    score=0.0,
                    regions=[],
                    artifact_bundle=blank_bundle,
                    runtime_ms=runtime_ms,
                    notes=notes,
                    config=config,
                    unsupported=False,   # authentic (not enough signal to flag)
                    reliability=reliability,
                    nli_applicable=False,
                )
                write_json(output_dir / "result.json", result.to_dict())
                # Still save minimal artefacts so the output directory is coherent
                noise_map_path = save_image(
                    to_grayscale_image(np.zeros(prepared.gray.shape, dtype=np.float32)),
                    output_dir / "noise_map.png",
                )
                blank_heatmap = save_image(
                    to_heatmap_image(np.zeros(prepared.gray.shape, dtype=np.float32)),
                    output_dir / "zscore_heatmap.png",
                )
                blank_mask = save_image(
                    to_grayscale_image(np.zeros(prepared.gray.shape, dtype=np.float32)),
                    output_dir / "anomaly_mask.png",
                )
                overlay_path = save_image(
                    build_overlay_image(prepared.rgb_image, np.zeros(prepared.gray.shape, dtype=bool), []),
                    output_dir / "overlay.png",
                )
                region_csv_path = _write_region_csv(output_dir / "region_stats.csv", [])
                return InferenceArtifacts(
                    result=result,
                    noise_map_path=noise_map_path,
                    zscore_heatmap_path=blank_heatmap,
                    overlay_path=overlay_path,
                    mask_path=blank_mask,
                    region_csv_path=region_csv_path,
                    debug_panel_path=None,
                    residual_map_path=None,
                )
    else:
        # validity disabled — record full fraction
        notes.append("patch_validity_enabled=false")

    # -----------------------------------------------------------------------
    # Phase 2: Extract residual (denoiser by default)
    # -----------------------------------------------------------------------
    residual = extract_residual(
        prepared.gray,
        config.method.residual_mode,
        radius=config.method.residual_denoise_radius,
    )

    # -----------------------------------------------------------------------
    # Gap 4: Per-channel RGB agreement mask
    # Build a boolean grid where True = ≥ min_channel_agreement channels
    # independently flag the block as anomalous.  This suppresses single-
    # channel JPEG artefacts that wouldn't show in the other two channels.
    # -----------------------------------------------------------------------
    channel_agreement_mask: np.ndarray | None = None
    if getattr(config.method, "multichannel_analysis", True):
        rgb_arr = np.asarray(prepared.rgb_image, dtype=np.float32)   # (H, W, 3)
        ref_bs = int(min(config.method.block_sizes))
        ch_flags: list[np.ndarray] = []
        for ch_idx in range(3):
            ch_gray = rgb_arr[:, :, ch_idx]
            ch_residual = extract_residual(
                ch_gray,
                config.method.residual_mode,
                radius=config.method.residual_denoise_radius,
            )
            ch_analysis = analyze_sigma_maps(ch_residual, [ref_bs])
            ch_z = ch_analysis.reference_abs_z_grid
            ch_flags.append((ch_z >= config.method.anomaly_z_threshold).astype(np.uint8))
        agreement = np.sum(np.stack(ch_flags, axis=0), axis=0)   # (grid_H, grid_W)
        min_agree = int(getattr(config.method, "min_channel_agreement", 2))
        channel_agreement_mask = agreement >= min_agree
        notes.append(f"multichannel_agreement={int(channel_agreement_mask.sum())}/{channel_agreement_mask.size}")

    # -----------------------------------------------------------------------
    # Phase 1+2: Sigma analysis (enhanced or legacy)
    # -----------------------------------------------------------------------
    if validity_mask is not None or region_type_grid is not None:
        analysis = analyze_sigma_maps_v2(
            residual,
            config.method.block_sizes,
            validity_mask=validity_mask,
            region_type_grid=region_type_grid,
            intensity_aware=config.method.intensity_aware_nlf,
            nlf_bins=config.method.nlf_intensity_bins,
            gray=prepared.gray if config.method.intensity_aware_nlf else None,
            consensus_z_threshold=getattr(config.method, "consensus_z_threshold", 1.5),
            local_neighborhood_z=getattr(config.method, "local_neighborhood_z", True),
            neighborhood_k=getattr(config.method, "local_neighborhood_k", 5),
        )
    else:
        analysis = analyze_sigma_maps_v2(
            residual,
            config.method.block_sizes,
            consensus_z_threshold=getattr(config.method, "consensus_z_threshold", 1.5),
            local_neighborhood_z=getattr(config.method, "local_neighborhood_z", True),
            neighborhood_k=getattr(config.method, "local_neighborhood_k", 5),
        )

    # -----------------------------------------------------------------------
    # Phase 1+2+3: Detection
    # -----------------------------------------------------------------------
    anomaly_mask, regions = detect_candidate_regions(
        analysis,
        config.method,
        validity_mask,
        channel_agreement_mask=channel_agreement_mask,
    )

    # -----------------------------------------------------------------------
    # Phase 3: Honest score computation
    # -----------------------------------------------------------------------
    if regions:
        score = float(max(region.score for region in regions))
    else:
        score = _compute_fallback_score(analysis, config.method.anomaly_z_threshold)
    score = float(min(1.0, max(0.0, score)))

    runtime_ms = int((perf_counter() - start) * 1000)

    # -----------------------------------------------------------------------
    # Visualisations
    # -----------------------------------------------------------------------
    noise_map_path = save_image(
        to_grayscale_image(analysis.combined_sigma_map), output_dir / "noise_map.png"
    )
    zscore_heatmap_path = save_image(
        to_heatmap_image(analysis.combined_abs_z_map), output_dir / "zscore_heatmap.png"
    )
    mask_path = save_image(
        to_grayscale_image(anomaly_mask.astype(np.float32)), output_dir / "anomaly_mask.png"
    )
    overlay_image = build_overlay_image(
        prepared.rgb_image,
        anomaly_mask,
        [region.bbox for region in regions],
    )
    overlay_path = save_image(overlay_image, output_dir / "overlay.png")
    region_csv_path = _write_region_csv(output_dir / "region_stats.csv", regions)

    debug_panel_path = None
    residual_map_path = None
    if config.output.save_residual_map or save_debug_panel_only:
        residual_map_path = save_image(
            to_grayscale_image(residual), output_dir / "residual_map.png"
        )
    if config.output.save_debug_panel or save_debug_panel_only:
        debug_panel_path = save_debug_panel(
            [
                ("rgb", prepared.rgb_image),
                ("noise_map", to_grayscale_image(analysis.combined_sigma_map).convert("RGB")),
                ("zscore", to_heatmap_image(analysis.combined_abs_z_map)),
                ("overlay", overlay_image),
            ],
            output_dir / "debug_panel.png",
        )

    artifact_bundle = ArtifactBundle(
        heatmap=portable_path(zscore_heatmap_path, Path.cwd()),
        mask=portable_path(mask_path, Path.cwd()),
        overlay=portable_path(overlay_path, Path.cwd()),
        debug_dir=portable_path(output_dir, Path.cwd()),
    )
    result = _build_result(
        prepared,
        score,
        regions,
        artifact_bundle,
        runtime_ms,
        notes,
        config,
        reliability=reliability,
        nli_applicable=nli_applicable,
    )
    write_json(output_dir / "result.json", result.to_dict())

    return InferenceArtifacts(
        result=result,
        noise_map_path=noise_map_path,
        zscore_heatmap_path=zscore_heatmap_path,
        overlay_path=overlay_path,
        mask_path=mask_path,
        region_csv_path=region_csv_path,
        debug_panel_path=debug_panel_path,
        residual_map_path=residual_map_path,
    )


# ---------------------------------------------------------------------------
# Public entry points (unchanged signatures)
# ---------------------------------------------------------------------------

def infer_single(
    input_path: str | Path,
    config: AppConfig,
    output_root: str | Path | None = None,
    save_debug_panel_only: bool = False,
) -> dict:
    source = Path(input_path)
    prepared = load_prepared_input(source, source.parent, config)
    artifacts = infer_prepared_input(
        prepared, config, output_root=output_root, save_debug_panel_only=save_debug_panel_only
    )
    return artifacts.result.to_dict()


def infer_batch(
    input_path: str | Path,
    config: AppConfig,
    output_root: str | Path | None = None,
    save_debug_panel_only: bool = False,
) -> dict:
    inputs, base_dir = discover_inputs(input_path, config)
    destination = ensure_dir(output_root or config.output.root_dir)
    results: list[dict] = []
    errors: list[dict] = []

    for source in inputs:
        try:
            prepared = load_prepared_input(source, base_dir, config)
            result = infer_prepared_input(
                prepared,
                config,
                output_root=destination,
                save_debug_panel_only=save_debug_panel_only,
            ).result.to_dict()
            results.append(result)
        except (UnsupportedInputError, ValueError) as exc:
            LOGGER.warning("Skipping %s: %s", source, exc)
            errors.append({"input_path": portable_path(source, base_dir), "error": str(exc)})

    batch_payload = {"results": results, "errors": errors}
    write_json(Path(destination) / "batch_results.json", batch_payload)
    export_doc_scores(results, Path(destination) / "doc_scores.csv")
    return batch_payload


def export_doc_scores(results: list[dict], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["doc_id", "input_path", "score", "decision", "confidence", "nli_applicable", "reliability"]
        )
        for result in results:
            writer.writerow(
                [
                    result["doc_id"],
                    result["input_path"],
                    f"{result['score']:.6f}",
                    result["decision"],
                    f"{result['confidence']:.6f}",
                    result.get("nli_applicable", True),
                    f"{result.get('reliability', 1.0):.4f}",
                ]
            )
    return target
