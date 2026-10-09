# Document Tamper Detector — RGB + content-mask branch. GPU (CUDA 12.1) image.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf PIP_NO_CACHE_DIR=1 YOLO_VERBOSE=false
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git libgl1 libglib2.0-0 curl ca-certificates \
 && ln -sf /usr/bin/python3.10 /usr/bin/python && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY examples/ examples/
ENV PYTHONPATH=/app/src DT_WEIGHTS=/app/weights/model.pt DT_DINOV3=/app/weights/dinov3_vitb16.pth

# Pre-download the three detector models into the image so the container needs no
# internet at run time (comment out to fetch lazily on first request instead).
RUN python - <<'PY'
import os, types, importlib.machinery as m
for n in ("bitsandbytes","bitsandbytes.nn"):
    x=types.ModuleType(n); x.__spec__=m.ModuleSpec(n,loader=None); import sys; sys.modules[n]=x
import easyocr; easyocr.Reader(["en"], gpu=False)
from transformers import AutoImageProcessor, AutoModelForObjectDetection
AutoImageProcessor.from_pretrained("mdefrance/yolos-tiny-signature-detection")
AutoModelForObjectDetection.from_pretrained("mdefrance/yolos-tiny-signature-detection")
from huggingface_hub import hf_hub_download; hf_hub_download("Armaggheddon/yolo26-document-layout","yolo26n_doc_layout.pt")
print("detector models cached")
PY

# Model + DINOv3 weights are NOT baked in (large). Mount or download at deploy time:
#   docker run -v /host/weights:/app/weights ...
# or bake them by running scripts/download_weights.sh before `docker build` and adding `COPY weights/ weights/`.

EXPOSE 8080
CMD ["uvicorn", "doc_tamper.api:app", "--host", "0.0.0.0", "--port", "8080"]
