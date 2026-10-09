"""FastAPI inference server. POST an image to /predict, get a tamper verdict + mask PNG (base64)."""
import base64, io, os
import cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from .inference import TamperDetector

app = FastAPI(title="Document Tamper Detector", version="1.0.0")
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = TamperDetector(overlap=int(os.environ.get("DT_OVERLAP", "384")))
    return _detector


@app.get("/health")
def health():
    return {"status": "ok", "device": str(get_detector().device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = Query(0.5), return_overlay: bool = Query(True)):
    buf = np.frombuffer(await file.read(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"error": "could not decode image"})
    det = get_detector()
    r = det.predict(bgr, threshold=threshold)
    resp = {k: r[k] for k in ("tampered", "tampered_pixels", "tampered_fraction", "max_score", "threshold", "height", "width")}
    _, mpng = cv2.imencode(".png", r["mask"]); resp["mask_png_base64"] = base64.b64encode(mpng).decode()
    if return_overlay:
        _, opng = cv2.imencode(".png", det.overlay(bgr, r["mask"])); resp["overlay_png_base64"] = base64.b64encode(opng).decode()
    return resp
