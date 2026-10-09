from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from noise_level_inconsistency.utils import ensure_dir, write_json


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_doc_scores_from_results(results_dir: Path, output_csv: Path) -> Path:
    result_files = sorted(results_dir.glob("*/result.json"))
    ensure_dir(output_csv.parent)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_id", "input_path", "score", "decision", "confidence"])
        for result_file in result_files:
            payload = _load_json(result_file)
            writer.writerow(
                [
                    payload["doc_id"],
                    payload["input_path"],
                    f"{payload['score']:.6f}",
                    payload["decision"],
                    f"{payload['confidence']:.6f}",
                ]
            )
    return output_csv


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def evaluate_region_metrics(results_dir: Path, mask_dir: Path) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_doc: list[dict] = []

    for result_file in sorted(results_dir.glob("*/result.json")):
        payload = _load_json(result_file)
        predicted_mask_path = results_dir / payload["doc_id"] / "anomaly_mask.png"
        ground_truth_mask = mask_dir / f"{payload['doc_id']}.png"
        if not predicted_mask_path.exists() or not ground_truth_mask.exists():
            continue
        predicted = _load_binary_mask(predicted_mask_path)
        truth = _load_binary_mask(ground_truth_mask)
        common_height = min(predicted.shape[0], truth.shape[0])
        common_width = min(predicted.shape[1], truth.shape[1])
        predicted = predicted[:common_height, :common_width]
        truth = truth[:common_height, :common_width]
        tp = int(np.logical_and(predicted, truth).sum())
        fp = int(np.logical_and(predicted, np.logical_not(truth)).sum())
        fn = int(np.logical_and(np.logical_not(predicted), truth).sum())
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_doc.append(
            {
                "doc_id": payload["doc_id"],
                "precision": round(precision, 6),
                "recall": round(recall, 6),
            }
        )

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "per_doc": per_doc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="nli-evaluate")
    parser.add_argument("--results-dir", required=True, help="Directory containing per-doc result folders.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation exports.")
    parser.add_argument("--mask-dir", default=None, help="Optional directory containing ground-truth masks.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(args.output_dir)
    export_doc_scores_from_results(results_dir, output_dir / "doc_scores.csv")

    payload = {"doc_scores_csv": str((output_dir / 'doc_scores.csv').as_posix())}
    if args.mask_dir:
        payload["region_metrics"] = evaluate_region_metrics(results_dir, Path(args.mask_dir))
    write_json(output_dir / "evaluation_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
