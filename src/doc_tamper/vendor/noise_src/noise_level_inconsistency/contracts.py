from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    TAMPERED = "tampered"
    UNSUPPORTED = "unsupported"


@dataclass(slots=True)
class RegionResult:
    bbox: list[int]
    score: float
    label: str = "noise_inconsistency"
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bbox": [int(value) for value in self.bbox],
            "score": round(float(self.score), 6),
            "label": self.label,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(slots=True)
class ArtifactBundle:
    heatmap: str | None = None
    mask: str | None = None
    overlay: str | None = None
    debug_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "heatmap": self.heatmap,
            "mask": self.mask,
            "overlay": self.overlay,
            "debug_dir": self.debug_dir,
        }


@dataclass(slots=True)
class MethodResult:
    method: str
    doc_id: str
    input_path: str
    score: float
    decision: str
    confidence: float
    regions: list[RegionResult] = field(default_factory=list)
    artifacts: ArtifactBundle = field(default_factory=ArtifactBundle)
    runtime_ms: int = 0
    notes: list[str] = field(default_factory=list)
    nli_applicable: bool = True
    reliability: float = 1.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "doc_id": self.doc_id,
            "input_path": self.input_path,
            "score": round(float(self.score), 6),
            "decision": self.decision,
            "confidence": round(float(self.confidence), 6),
            "regions": [region.to_dict() for region in self.regions],
            "artifacts": self.artifacts.to_dict(),
            "runtime_ms": int(self.runtime_ms),
            "notes": list(self.notes),
            "nli_applicable": bool(self.nli_applicable),
            "reliability": round(float(self.reliability), 4),
        }
