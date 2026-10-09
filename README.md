# Document Tamper Detector — RGB + Content-Mask Branch

Per-pixel **tamper localization** for document images (marksheets, grade sheets, results, NOC / bonafide
letters and similar Indian academic documents). Given a page, it returns a probability map and a binary
**mask** highlighting the regions that look forged, plus a simple `tampered: true/false` verdict.

This is the **A3 / "RGB + all-five content-mask"** configuration — the best-performing content-fed model from
our evaluation (DOC57). Architecture: a frozen **DINOv3 ViT-B/16** vision backbone (LoRA-adapted) plus a
small **content-mask branch** (text / logo / signature / stamp / watermark priors), fused and decoded to a
tamper mask.

> ⚠️ **Scope & honesty note.** The model was trained on **academic documents**. On that domain it localizes
> logo/seal swaps, signature swaps and larger text edits well. It does **not** currently transfer to ID cards,
> PAN cards, bills or certificates, nor to single-digit edits (see our DOC58/DOC59 analysis). Deploy it for the
> academic-document use case it was built for.

---

## What's in here

```
src/doc_tamper/
  inference.py      # TamperDetector: load model + content extractor, predict()
  content_mask.py   # 5-channel content-mask pipeline (OCR text, YOLO logo, YOLOS signature, CV stamp/watermark)
  api.py            # FastAPI server (/predict, /health)
  vendor/           # all model code, vendored so the repo is self-contained
    dualbranch/     #   the dual-branch model + fusion/semantic helpers
    noise_src/      #   the DINOv3+CMX base model and noise branch
    dinov3_repo/    #   the DINOv3 ViT package
    detectors/      #   classical stamp/watermark + layout->logo helpers
scripts/download_weights.sh   # fetch model + backbone weights from the GitHub Release
examples/predict_cli.py       # one-shot CLI
Dockerfile, requirements.txt  # GPU container
configs/model.yaml            # the deployed config, documented
```

Weights are **not** in the git tree (they are large). They are published as a **GitHub Release asset** and
fetched by `scripts/download_weights.sh`:

| file | size | what |
|---|---|---|
| `model.pt` | ~460 MB | the trained tamper-localizer (A3) |
| `dinov3_vitb16.pth` | ~343 MB | the DINOv3 backbone it builds on |

---

## Quick start (local, GPU recommended)

```bash
git clone <this-repo> && cd doc-tamper-detector
scripts/download_weights.sh                      # pulls model.pt + dinov3_vitb16.pth into ./weights
pip install -r requirements.txt

# one image -> mask.png + overlay.png + JSON verdict
PYTHONPATH=src python examples/predict_cli.py path/to/page.png out/
```

Or from Python:

```python
import sys; sys.path.insert(0, "src")
from doc_tamper import TamperDetector
det = TamperDetector()                    # overlap=384 (accurate). Use overlap=128 for ~5x speed.
r = det.predict("page.png", threshold=0.5)
print(r["tampered"], r["max_score"])      # verdict + confidence
# r["mask"] is a HxW uint8 mask; r["prob_map"] is the HxW probability map
```

First run downloads three small detector models (~50 MB total) from the Hugging Face Hub.

---

## Run as a service (Docker)

```bash
scripts/download_weights.sh                       # weights into ./weights
docker build -t doc-tamper .
docker run --gpus all -p 8080:8080 -v "$PWD/weights:/app/weights" doc-tamper
```

The image pre-caches the detector models at build time, so the running container needs no outbound internet.
Query it:

```bash
curl -s -X POST "http://localhost:8080/predict?threshold=0.5" -F "file=@page.png" | jq '.tampered, .max_score'
# response: {tampered, tampered_pixels, tampered_fraction, max_score, mask_png_base64, overlay_png_base64, ...}
```

`GET /health` returns `{"status":"ok","device":"cuda"}`.

---

## Deploy on AWS

The model is ~94 M parameters (~460 MB weights, +343 MB backbone). Peak GPU memory ~6 GiB. Throughput on an
NVIDIA L4: roughly **1 s/page** at overlap 384, **~0.2 s/page** at overlap 128.

### Recommended: single EC2 GPU instance

1. **Launch** an EC2 instance with an NVIDIA GPU and the latest **Deep Learning AMI** (Docker + NVIDIA runtime
   preinstalled). Good fit: **`g6.xlarge`** (1× L4, 24 GB) — the cheapest current-gen GPU that fits comfortably.
2. **Get the code + weights**
   ```bash
   git clone <this-repo> && cd doc-tamper-detector
   REPO=<owner>/<repo> scripts/download_weights.sh      # or copy weights/ up via scp
   ```
3. **Build & run**
   ```bash
   docker build -t doc-tamper .
   docker run -d --restart unless-stopped --gpus all -p 8080:8080 \
     -v "$PWD/weights:/app/weights" --name tamper doc-tamper
   ```
4. **Expose it**: put it behind an Application Load Balancer / API Gateway, or an Nginx reverse proxy with TLS.
   Lock the security group to your callers. Health check path: `/health`.

### Container registry (ECR) + ECS/EKS

```bash
aws ecr create-repository --repository-name doc-tamper
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker tag doc-tamper <acct>.dkr.ecr.<region>.amazonaws.com/doc-tamper:1.0
docker push <acct>.dkr.ecr.<region>.amazonaws.com/doc-tamper:1.0
```
Run as an **ECS task** (GPU-enabled capacity provider, e.g. a `g6` instance) or an **EKS** GPU node group.
Store `model.pt` / `dinov3_vitb16.pth` in **S3** and have the task pull them to a volume on start
(`aws s3 cp s3://<bucket>/weights/ /app/weights/ --recursive`) instead of baking them into the image.

### SageMaker (real-time endpoint)

Bring-your-own-container: push the image to ECR, upload the two weight files as a `model.tar.gz` to S3, and
create a SageMaker Model → EndpointConfig (instance `ml.g6.xlarge`) → Endpoint. Wrap `api.py` with the
SageMaker `/invocations` and `/ping` contract (map `/ping`→`/health`, `/invocations`→`/predict`).

### Cost / scaling notes

- **GPU strongly preferred.** CPU works but is ~30–60 s/page.
- For batch/offline jobs, use **overlap 128** (≈5× faster, marginally lower IoU) and raise `batch`.
- One L4 handles ~1 req/s single-stream; scale horizontally behind a load balancer for more.
- Keep fp32 (the deployment default). bf16 changed outputs on some pages in testing.

---

## API reference

`POST /predict?threshold=0.5&return_overlay=true` with multipart `file=@image`. Returns JSON:

| field | meaning |
|---|---|
| `tampered` | `true` if any pixel exceeds the threshold |
| `tampered_pixels`, `tampered_fraction` | size of the flagged region |
| `max_score` | highest tamper probability on the page (0–1) |
| `mask_png_base64` | the binary tamper mask, PNG |
| `overlay_png_base64` | the page with the mask drawn in red (if `return_overlay`) |
| `height`, `width` | input dimensions |

`TamperDetector(weights=, dinov3_ckpt=, device=, tile=512, overlap=384, batch=8)` — override paths with the
`DT_WEIGHTS` / `DT_DINOV3` environment variables (the Docker image sets them to `/app/weights/...`).

---

## Licensing / provenance

- **DINOv3** backbone and vendored `vendor/dinov3_repo/` code are under their original license (see
  `vendor/dinov3_repo/`); review Meta's DINOv3 license terms before commercial use.
- The trained `model.pt` was fine-tuned on real academic documents collected for this project — treat it as
  internal/confidential and do not redistribute outside your organization without clearing data rights.
- No personal documents or datasets are included in this repository.
