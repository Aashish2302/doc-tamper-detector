"""5-channel content mask (Branch B prior) for the tamper detector.

Channels, in order: text, logo, signature, stamp, watermark. Computed at long-side 1600 by a mix of
learned detectors (OCR text, YOLO document-layout logo, YOLOS signature) and classical CV (stamp, watermark).
The detector weights download automatically from the Hugging Face Hub the first time this runs.
"""
from __future__ import annotations
import os, sys, types, importlib.machinery as _mach
import cv2
import numpy as np

# detectors_v2 (classical stamp/watermark + layout->logo + dilate) is vendored
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "detectors"))
from detectors_v2 import watermark_mask_v2, logo_mask_from_layout, stamp_mask_v2, dilate  # noqa: E402

LONG_SIDE = 1600
CHANNELS = ["text", "logo", "signature", "stamp", "watermark"]


class ContentMaskExtractor:
    """Lazily loads the three learned detectors and produces a (5, H, W) uint8 content mask."""

    def __init__(self, device: str = "cpu", ocr_langs=("en",)):
        self.device = device
        self._ocr_langs = list(ocr_langs)
        self._reader = self._sig_proc = self._sig_model = self._layout = None

    # ---- lazy detector loaders -------------------------------------------------
    def _ensure(self):
        if self._reader is not None:
            return
        # some transformers paths pull bitsandbytes transitively; stub it (never used here)
        for n in ("bitsandbytes", "bitsandbytes.nn"):
            if n not in sys.modules:
                m = types.ModuleType(n); m.__spec__ = _mach.ModuleSpec(n, loader=None); sys.modules[n] = m
        sys.modules["bitsandbytes"].nn = sys.modules["bitsandbytes.nn"]
        import easyocr, torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        os.environ.setdefault("YOLO_VERBOSE", "false")
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        gpu = self.device.startswith("cuda")
        self._torch = torch
        self._reader = easyocr.Reader(self._ocr_langs, gpu=gpu)
        self._sig_proc = AutoImageProcessor.from_pretrained("mdefrance/yolos-tiny-signature-detection")
        self._sig_model = AutoModelForObjectDetection.from_pretrained(
            "mdefrance/yolos-tiny-signature-detection").eval()
        self._layout = YOLO(hf_hub_download("Armaggheddon/yolo26-document-layout", "yolo26n_doc_layout.pt"))

    # ---- per-channel detectors -------------------------------------------------
    def _text(self, rgb):
        m = np.zeros(rgb.shape[:2], np.uint8)
        for item in self._reader.readtext(rgb):
            cv2.fillPoly(m, [np.array(item[0], dtype=np.int32)], 255)
        return dilate(m, 25)

    def _signature(self, rgb):
        from PIL import Image
        pil = Image.fromarray(rgb)
        with self._torch.no_grad():
            o = self._sig_model(**self._sig_proc(images=pil, return_tensors="pt"))
        res = self._sig_proc.post_process_object_detection(
            o, threshold=0.3, target_sizes=self._torch.tensor([pil.size[::-1]]))[0]
        m = np.zeros(rgb.shape[:2], np.uint8)
        for b in res["boxes"]:
            x0, y0, x1, y1 = [int(v) for v in b.tolist()]
            cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)
        return dilate(m, 15)

    def _layout_boxes(self, bgr):
        res = self._layout.predict(bgr[:, :, ::-1], verbose=False, conf=0.25)[0]
        out = []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            out.append((self._layout.names[int(b.cls.item())], float(b.conf.item()), x0, y0, x1, y1))
        return out

    # ---- public API ------------------------------------------------------------
    def extract(self, bgr: np.ndarray) -> np.ndarray:
        """bgr: (H0,W0,3) uint8 (OpenCV order). Returns (5,H0,W0) uint8, 0/255 per channel."""
        self._ensure()
        h0, w0 = bgr.shape[:2]
        s = LONG_SIDE / max(h0, w0)
        work = cv2.resize(bgr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else bgr
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        H, W = work.shape[:2]
        m_text = self._text(rgb)
        m_sig = self._signature(rgb)
        m_stamp = stamp_mask_v2(rgb)[0]
        m_logo = logo_mask_from_layout(self._layout_boxes(work), (H, W))[0]
        excl = m_text.copy()
        for x in (m_sig, m_stamp, m_logo):
            excl = cv2.bitwise_or(excl, x)
        m_water = watermark_mask_v2(rgb, excl)[0]
        stack = np.stack([m_text, m_logo, m_sig, m_stamp, m_water], 0).astype(np.uint8)
        if (H, W) != (h0, w0):  # back to original resolution
            stack = np.stack([cv2.resize(c, (w0, h0), interpolation=cv2.INTER_NEAREST) for c in stack], 0)
        return stack
