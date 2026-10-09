from __future__ import annotations

import argparse
import json
from pathlib import Path

from noise_level_inconsistency.config import load_config
from noise_level_inconsistency.io import UnsupportedInputError
from noise_level_inconsistency.pipeline import infer_batch, infer_single
from noise_level_inconsistency.utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nli-doc-forensics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run NLI on a single local page image.")
    infer_parser.add_argument("--input", required=True, help="Single image or rasterized PDF page image.")
    infer_parser.add_argument("--config", required=True, help="YAML config path.")
    infer_parser.add_argument("--output-dir", default=None, help="Optional output root override.")

    batch_parser = subparsers.add_parser("infer-batch", help="Run NLI over a folder of local images.")
    batch_parser.add_argument("--input", required=True, help="Folder containing images.")
    batch_parser.add_argument("--config", required=True, help="YAML config path.")
    batch_parser.add_argument("--output-dir", default=None, help="Optional output root override.")

    debug_parser = subparsers.add_parser("save-debug", help="Run NLI and force extended debug outputs.")
    debug_parser.add_argument("--input", required=True, help="Single image or rasterized PDF page image.")
    debug_parser.add_argument("--config", required=True, help="YAML config path.")
    debug_parser.add_argument("--output-dir", default=None, help="Optional output root override.")

    return parser


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))

    try:
        if args.command == "infer":
            _emit(infer_single(args.input, config, output_root=args.output_dir))
            return 0
        if args.command == "infer-batch":
            _emit(infer_batch(args.input, config, output_root=args.output_dir))
            return 0
        if args.command == "save-debug":
            _emit(
                infer_single(
                    args.input,
                    config,
                    output_root=args.output_dir,
                    save_debug_panel_only=True,
                )
            )
            return 0
    except (UnsupportedInputError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    parser.exit(status=1, message="Unknown command.\n")


if __name__ == "__main__":
    raise SystemExit(main())
