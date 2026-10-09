from __future__ import annotations

import json
import logging
import re
from pathlib import Path


LOGGER_NAME = "noise_level_inconsistency"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or LOGGER_NAME)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict | list) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def portable_path(path: str | Path, start: str | Path | None = None) -> str:
    candidate = Path(path)
    if start is not None:
        try:
            candidate = candidate.resolve().relative_to(Path(start).resolve())
        except ValueError:
            candidate = Path(path)
    return candidate.as_posix()


def sanitize_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "item"


def build_doc_id(path: str | Path, base_dir: str | Path) -> str:
    candidate = Path(path)
    base = Path(base_dir)
    try:
        relative = candidate.resolve().relative_to(base.resolve())
    except ValueError:
        relative = Path(candidate.name)
    parts = [sanitize_segment(part) for part in relative.with_suffix("").parts]
    return "__".join(parts)
