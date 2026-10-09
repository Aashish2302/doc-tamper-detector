from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class InputConfig:
    recursive: bool = True
    supported_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    )


@dataclass(slots=True)
class PreprocessConfig:
    default_input_dpi: int = 200
    target_dpi: int = 200
    min_dimension: int = 64


@dataclass(slots=True)
class MethodConfig:
    # ---- core residual / block params ----
    residual_mode: str = "denoiser"        # laplacian | wavelet | denoiser | srm
    residual_denoise_radius: int = 2       # radius for denoiser median filter
    block_sizes: list[int] = field(default_factory=lambda: [8, 16, 32])
    anomaly_z_threshold: float = 2.5
    merge_gap_blocks: int = 1
    min_component_blocks: int = 3
    suspicious_score_threshold: float = 0.3
    tampered_score_threshold: float = 0.65
    advanced_mode: bool = False            # kept for backward compat; superseded by Phase 1–3
    region_expand_blocks: int = 1

    # ---- Phase 1: patch validity ----
    patch_validity_enabled: bool = True
    structural_blur_radius: int = 5        # box-blur radius for structural variance
    structural_var_percentile: float = 85.0  # top % of structured blocks to exclude
    flatness_threshold: float = 1.0        # std below which a block is too flat

    # ---- Phase 1: region-aware sigma ----
    region_aware_sigma: bool = True
    min_blocks_per_region_type: int = 10   # min valid blocks to build per-type reference

    # ---- Phase 1: applicability gate ----
    applicability_enabled: bool = True
    min_valid_block_fraction: float = 0.15
    blur_variance_threshold: float = 50.0
    min_noise_sigma: float = 0.5

    # ---- Phase 2: z-grid smoothing ----
    smooth_z_grid_radius: int = 1          # 0 = disabled; 1 = 3×3 mean filter on z-grid

    # ---- Phase 2: intensity-aware NLF ----
    intensity_aware_nlf: bool = True       # compare block sigma to NLF(intensity)
    nlf_intensity_bins: int = 8

    # ---- Gap 1: multi-scale consensus weighting ----
    consensus_z_threshold: float = 1.5    # z floor for counting a scale as "agreeing"

    # ---- Gap 2: local neighbourhood z-score ----
    local_neighborhood_z: bool = True     # compare each block to its K×K spatial neighbours
    local_neighborhood_k: int = 5         # neighbourhood size (5 → 5×5 window)

    # ---- Gap 3: max merged component bbox fraction ----
    max_region_bbox_fraction: float = 0.50  # skip merges that would exceed this fraction of grid

    # ---- Gap 4: per-channel RGB agreement ----
    multichannel_analysis: bool = True    # run sigma on R/G/B separately
    min_channel_agreement: int = 2        # min channels that must agree for a block to fire


@dataclass(slots=True)
class OutputConfig:
    root_dir: str = "methods/noise_level_inconsistency/outputs/results"
    save_debug_panel: bool = False
    save_residual_map: bool = True


@dataclass(slots=True)
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> AppConfig:
    defaults = asdict(AppConfig())
    if config_path is None:
        merged = defaults
    else:
        loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        merged = _deep_merge(defaults, loaded)

    return AppConfig(
        input=InputConfig(**merged["input"]),
        preprocess=PreprocessConfig(**merged["preprocess"]),
        method=MethodConfig(**merged["method"]),
        output=OutputConfig(**merged["output"]),
    )
