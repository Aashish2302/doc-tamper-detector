"""Minimal smoke test: build the model and run a forward pass on a random image.
Needs weights/ present (scripts/download_weights.sh). Run: PYTHONPATH=src python tests/smoke_test.py"""
import numpy as np
from doc_tamper import TamperDetector
det = TamperDetector(device="cpu", overlap=256)
r = det.predict((np.random.rand(600, 900, 3) * 255).astype("uint8"))
assert set(["tampered", "max_score", "mask"]).issubset(r)
assert r["mask"].shape == (600, 900)
print("SMOKE OK:", {k: r[k] for k in ("tampered", "max_score", "tampered_pixels")})
