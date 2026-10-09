from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from noise_level_inconsistency.config import AppConfig
from noise_level_inconsistency.models import PreparedInput
from noise_level_inconsistency.preprocess import normalize_input_image
from noise_level_inconsistency.utils import build_doc_id, portable_path


class UnsupportedInputError(Exception):
    """Raised when an input cannot be processed by this method."""


def discover_inputs(input_path: str | Path, config: AppConfig) -> tuple[list[Path], Path]:
    root = Path(input_path)
    if not root.exists():
        raise UnsupportedInputError(f"Input path does not exist: {root}")

    supported = {extension.lower() for extension in config.input.supported_extensions}
    if root.is_file():
        if root.suffix.lower() not in supported:
            raise UnsupportedInputError(f"Unsupported image extension: {root.suffix}")
        return [root], root.parent

    pattern = "**/*" if config.input.recursive else "*"
    inputs = sorted(
        path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in supported
    )
    if not inputs:
        raise UnsupportedInputError(f"No supported image inputs found under: {root}")
    return inputs, root


def load_prepared_input(path: str | Path, base_dir: str | Path, config: AppConfig) -> PreparedInput:
    source = Path(path)
    try:
        with Image.open(source) as image:
            rgb_image, gray, dpi, notes = normalize_input_image(image, config.preprocess)
    except UnidentifiedImageError as exc:
        raise UnsupportedInputError(f"Unreadable image: {source}") from exc

    return PreparedInput(
        doc_id=build_doc_id(source, base_dir),
        source_path=source,
        relative_input_path=portable_path(source, base_dir),
        rgb_image=rgb_image,
        gray=gray,
        dpi=dpi,
        notes=notes,
    )
