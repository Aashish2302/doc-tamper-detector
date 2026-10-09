from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class PreparedInput:
    doc_id: str
    source_path: Path
    relative_input_path: str
    rgb_image: Image.Image
    gray: np.ndarray
    dpi: int
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SigmaScale:
    block_size: int
    sigma_grid: np.ndarray
    z_grid: np.ndarray
    median_sigma: float
    mad_sigma: float


@dataclass(slots=True)
class CandidateRegionStats:
    bbox: tuple[int, int, int, int]
    score: float
    area_blocks: int
    area_pixels: int
    mean_sigma: float
    max_sigma: float
    mean_abs_z: float
    max_abs_z: float
    mismatch: float
    reason_codes: list[str] = field(default_factory=list)
    region_type: str = "unknown"
