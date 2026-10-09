#!/usr/bin/env python3
"""CLI: python examples/predict_cli.py <image> [out_dir] — writes mask.png and overlay.png, prints the verdict."""
import sys, os, json, cv2
from doc_tamper import TamperDetector

img = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "."
det = TamperDetector()
r = det.predict(img)
os.makedirs(out, exist_ok=True)
cv2.imwrite(os.path.join(out, "mask.png"), r["mask"])
cv2.imwrite(os.path.join(out, "overlay.png"), det.overlay(cv2.imread(img), r["mask"]))
print(json.dumps({k: r[k] for k in ("tampered", "tampered_pixels", "tampered_fraction", "max_score")}, indent=2))
print(f"wrote {out}/mask.png and {out}/overlay.png")
