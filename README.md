# Document Tamper Detector — RGB + Content-Mask Branch

Per-pixel **tamper localization** for document images (marksheets, grade sheets, results, NOC / bonafide
letters and similar Indian academic documents). Given a page, it returns a probability map and a binary
**mask** of the regions that look forged, plus a simple `tampered: true/false` verdict.

This is the **A3 / "RGB + all-five content-mask"** configuration — the best content-fed model in our evaluation.
Architecture: a **DINOv3 ViT-B/16** vision backbone (LoRA-adapted) plus a small **content-mask branch**
(text / logo / signature / stamp / watermark priors), fused and decoded to a tamper mask.

> **Scope.** The model was trained on **academic documents**. On that domain it localizes logo/seal swaps,
> signature swaps and larger text edits well. It does **not** currently work on ID cards, PAN cards, bills or
> certificates, nor on single-digit edits. Deploy it for the academic-document use case.

**Contents:** [What's in here](#whats-in-here) · [Run locally](#run-locally) ·
[Deploy on AWS — step by step](#deploy-on-aws--step-by-step) · [API reference](#api-reference) ·
[Troubleshooting](#troubleshooting)

---

## What's in here

```
src/doc_tamper/
  inference.py      # TamperDetector: load model + content extractor, predict()
  content_mask.py   # 5-channel content mask (OCR text, YOLO logo, YOLOS signature, CV stamp/watermark)
  api.py            # FastAPI server: POST /predict, GET /health
  vendor/           # all model code, vendored so the repo is self-contained
scripts/download_weights.sh   # fetch the two weight files from the GitHub Release
examples/predict_cli.py       # one-image CLI
Dockerfile, requirements.txt  # GPU container
configs/model.yaml            # the deployed configuration, documented
```

The weights are not in the git tree (too large). They are attached to the **`weights-v1` release** and fetched by
`scripts/download_weights.sh` (no login needed):

| file | size | what |
|---|---|---|
| `model.pt` | 460 MB | the trained tamper localizer |
| `dinov3_vitb16.pth` | 343 MB | the DINOv3 backbone it builds on |

### ID-card model (`weights-v2-id` release)

A second set of weights, retrained with PAN and Aadhaar cards added, is attached to the **`weights-v2-id` release**.
Same architecture and code; only the training data differs. Fetch it with:

```bash
TAG=weights-v2-id scripts/download_weights.sh
```

Training data: the original academic set (201 forged + 250 genuine pages, used twice) + 15 forged / 28 genuine ID pages
+ 768 PAN and 81 Aadhaar forgeries (2-4 single-character edits each, cards from the Kaggle datasets
`nagendra048/pan-card-dataset` and `nagendra048/aadhar-dataset`) with their untouched originals.
Checkpoint = epoch 25 (best validation IoU 0.625).

Held-out test, threshold 0.5, tile 512, overlap 384 (cards split so the same physical card is never in train and test):

| test set | model | mean IoU | fakes found (IoU >= 0.5) | fakes missed | genuine pages flagged |
|---|---|---|---|---|---|
| PAN cards (214 fake / 214 genuine) | `weights-v1` | 0.043 | 0 | 18 | 213 |
| | **`weights-v2-id`** | **0.582** | **141** | **1** | **111** |
| Aadhaar cards (23 / 23) | `weights-v1` | 0.001 | 0 | 18 | 20 |
| | **`weights-v2-id`** | **0.291** | **4** | **2** | **5** |
| Academic hard set (50 / 50) | `weights-v1` | **0.407** | **21** | **8** | 42 |
| | `weights-v2-id` | 0.359 | 16 | 11 | 44 |

Use `weights-v2-id` for PAN / Aadhaar cards and `weights-v1` (the default) for academic documents.

The first prediction also downloads three small detector models (about 100 MB) from the Hugging Face Hub. The
Docker image downloads them at build time, so the running container needs no internet.

---

## Run locally

Needs Python 3.10 and, ideally, an NVIDIA GPU (CPU works but takes about 30–60 s per page).

```bash
git clone https://github.com/Aashish2302/doc-tamper-detector.git
cd doc-tamper-detector
scripts/download_weights.sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

PYTHONPATH=src python examples/predict_cli.py path/to/page.png out/   # writes out/mask.png, out/overlay.png
```

From Python:

```python
import sys; sys.path.insert(0, "src")
from doc_tamper import TamperDetector
det = TamperDetector()                    # overlap=384 (most accurate); overlap=128 is ~5x faster
r = det.predict("page.png", threshold=0.5)
print(r["tampered"], r["max_score"])      # r["mask"] = HxW mask, r["prob_map"] = HxW probabilities
```

---

## Deploy on AWS — step by step

This sets up **one GPU server on EC2** running the detector as a web API. It takes about 30–45 minutes the first
time, most of it waiting for the Docker build.

**What you'll end up with:** an EC2 `g6.xlarge` (1× NVIDIA L4 GPU, 24 GB) running the container, answering
`POST http://<server>:8080/predict`.

**Cost:** a `g6.xlarge` is roughly **$0.80/hour on demand** in `us-east-1` (about $580/month if left running
24/7). Check current pricing for your region. Stop the instance when you don't need it (step 11).

### Step 0 — What you need

- An AWS account, and permission to create EC2 instances, key pairs and security groups.
- The **AWS CLI v2** on your laptop, configured with `aws configure`. Every step also lists the Console route.
- An SSH client (macOS/Linux terminal, or Windows PowerShell).

Pick a region and use it throughout. The examples use `us-east-1`:

```bash
export AWS_REGION=us-east-1
```

### Step 1 — Make sure your account is allowed to launch a GPU instance

New AWS accounts often have a GPU quota of **0**, which makes the launch in step 3 fail. Check it:

```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
  --query "Quota.Value" --region $AWS_REGION
```

This is the quota *Running On-Demand G and VT instances*, counted in vCPUs. You need **at least 4**. If it says
`0.0`, request an increase:

```bash
aws service-quotas request-service-quota-increase --service-code ec2 \
  --quota-code L-DB2E81BA --desired-value 8 --region $AWS_REGION
```

Console route: **Service Quotas → AWS services → Amazon EC2 → "Running On-Demand G and VT instances" →
Request increase**. Approval can take from minutes to a day.

### Step 2 — Create an SSH key pair and a firewall (security group)

```bash
# SSH key (saved to your laptop; keep it safe)
aws ec2 create-key-pair --key-name doc-tamper-key --query KeyMaterial --output text \
  --region $AWS_REGION > doc-tamper-key.pem
chmod 400 doc-tamper-key.pem

# Security group in your default VPC
SG_ID=$(aws ec2 create-security-group --group-name doc-tamper-sg \
  --description "doc tamper detector" --query GroupId --output text --region $AWS_REGION)

MY_IP=$(curl -s https://checkip.amazonaws.com)/32
# SSH only from your own IP
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr $MY_IP --region $AWS_REGION
# API port only from your own IP for now (widen later to the machines that will call it)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8080 --cidr $MY_IP --region $AWS_REGION
echo $SG_ID
```

> **The API has no login of its own.** Never open port 8080 to `0.0.0.0/0`. Allow only the IP addresses that
> need to call it, or put it behind a load balancer as in step 9.

### Step 3 — Launch the GPU instance

Use AWS's **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**. It comes with the NVIDIA driver,
Docker and the NVIDIA container toolkit already installed.

```bash
AMI_ID=$(aws ssm get-parameter --region $AWS_REGION \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)

INSTANCE_ID=$(aws ec2 run-instances --region $AWS_REGION \
  --image-id $AMI_ID --instance-type g6.xlarge --key-name doc-tamper-key \
  --security-group-ids $SG_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=doc-tamper}]' \
  --query 'Instances[0].InstanceId' --output text)

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $AWS_REGION
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $AWS_REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Instance $INSTANCE_ID at $PUBLIC_IP"
```

Console route: **EC2 → Launch instance**. Search the AMI catalog for *"Deep Learning Base OSS Nvidia Driver GPU
AMI (Ubuntu 22.04)"*. Choose instance type **g6.xlarge**, key pair **doc-tamper-key**, security group
**doc-tamper-sg**, and a **100 GiB gp3** root volume. The Docker image is about 15 GB, so don't keep the default
volume size.

If `g6.xlarge` isn't available in your region, `g5.xlarge` (A10G) or `g4dn.xlarge` (T4, slower) also work.

### Step 4 — Connect and check the GPU

```bash
ssh -i doc-tamper-key.pem ubuntu@$PUBLIC_IP
```

On the server:

```bash
nvidia-smi                                                                   # should list an NVIDIA L4
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi    # Docker can see the GPU
```

If the second command fails, see [Troubleshooting](#troubleshooting).

### Step 5 — Get the code and the weights (on the server)

```bash
git clone https://github.com/Aashish2302/doc-tamper-detector.git
cd doc-tamper-detector
scripts/download_weights.sh          # downloads ~800 MB into ./weights/
ls -lh weights/                      # model.pt (~460 MB) and dinov3_vitb16.pth (~343 MB)
```

### Step 6 — Build and start the container

```bash
docker build -t doc-tamper:1.0 .     # first build takes ~15-25 min

docker run -d --name doc-tamper --restart unless-stopped --gpus all \
  -p 8080:8080 -v "$PWD/weights:/app/weights:ro" \
  doc-tamper:1.0

docker logs -f doc-tamper            # wait for "Uvicorn running on http://0.0.0.0:8080", then Ctrl-C
```

Optional settings, passed with `-e NAME=value` on `docker run`:

| variable | default | meaning |
|---|---|---|
| `DT_OVERLAP` | `384` | tile overlap. `384` = most accurate; `128` = about 5x faster, slightly lower accuracy |

### Step 7 — Test it

From the server:

```bash
curl http://localhost:8080/health
# {"status":"ok","device":"cuda"}     <- "cuda" means it is using the GPU

curl -s -X POST "http://localhost:8080/predict?threshold=0.5" -F "file=@/path/to/page.png" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['tampered'],d['max_score'],d['tampered_pixels'])"
```

From your laptop, which step 2 allowed:

```bash
curl http://$PUBLIC_IP:8080/health
curl -s -X POST "http://$PUBLIC_IP:8080/predict" -F "file=@page.png" > result.json
```

`result.json` has the verdict plus the mask and a red overlay as base64 PNGs. To save the overlay:

```bash
python3 -c "import json,base64;d=json.load(open('result.json'));open('overlay.png','wb').write(base64.b64decode(d['overlay_png_base64']))"
```

The first request is slow, roughly 20–60 s, while the models load into GPU memory. Later requests are fast. We
haven't benchmarked an L4, so measure on your own pages. On an H100 the model alone takes about 1 s per page at
overlap 384 and 0.2 s at overlap 128, and the content-mask detectors add some time per page.

### Step 8 — Make sure it survives reboots

`--restart unless-stopped` from step 6 already restarts the container whenever Docker starts, and Docker starts on
boot on this AMI. To check:

```bash
sudo reboot
# wait ~2 min, ssh back in, then:
docker ps                                # doc-tamper should be "Up"
curl http://localhost:8080/health
```

### Step 9 — Production: HTTPS and access control

Don't expose the raw port to the internet. Two common options:

- **Application Load Balancer (recommended):**
  1. Create a target group: protocol HTTP, port 8080, health-check path `/health`. Register the instance.
  2. Create an ALB with an HTTPS listener (443) using a certificate from **AWS Certificate Manager**, forwarding to
     the target group.
  3. Change the instance's security group so port 8080 accepts traffic **only from the ALB's security group**.
  4. Add authentication in front: an API Gateway with API keys, ALB + Cognito, or your own backend calling it.
  5. Raise the ALB idle timeout to **120 s**, so slow pages aren't cut off.
- **Nginx + Let's Encrypt on the instance:** put Nginx in front of `localhost:8080`, get a certificate with
  `certbot`, and set `client_max_body_size 25m;` and `proxy_read_timeout 120s;`.

Data handling: uploaded images are processed in memory and **not stored** by the service. Requests are not
logged beyond the standard access log line.

### Step 10 — Updating to a new version

```bash
cd ~/doc-tamper-detector
git pull
scripts/download_weights.sh                 # only if the weights changed
docker build -t doc-tamper:1.1 .
docker rm -f doc-tamper
docker run -d --name doc-tamper --restart unless-stopped --gpus all \
  -p 8080:8080 -v "$PWD/weights:/app/weights:ro" doc-tamper:1.1
```

### Step 11 — Stop paying when you're not using it

```bash
aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $AWS_REGION        # stops GPU billing; disk is kept
aws ec2 start-instances --instance-ids $INSTANCE_ID --region $AWS_REGION       # the container comes back by itself
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $AWS_REGION   # deletes everything
```

A stopped instance still costs a little for its 100 GB disk, about $8/month. When it starts again the **public IP
changes**, unless you attach an Elastic IP.

### Optional — scaling beyond one server

- **ECR + ECS:** push the image to Amazon ECR, then run it as an ECS service on a GPU capacity provider (g6
  instances) behind an ALB. Keep the two weight files in S3 and copy them to a volume at task start
  (`aws s3 cp s3://<bucket>/weights/ /app/weights/ --recursive`), instead of baking them into the image.
  ```bash
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
  aws ecr create-repository --repository-name doc-tamper --region $AWS_REGION
  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
  docker tag doc-tamper:1.0 $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/doc-tamper:1.0
  docker push $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/doc-tamper:1.0
  ```
- **SageMaker real-time endpoint:** use the same ECR image as a bring-your-own container on `ml.g6.xlarge`.
  SageMaker expects `GET /ping` and `POST /invocations` on port 8080, so add those two routes in `api.py`,
  pointing at `health` and `predict`.
- Each GPU handles one page at a time. For more traffic, run more instances behind the load balancer.

---

## API reference

**`GET /health`** returns `{"status": "ok", "device": "cuda"}`.

**`POST /predict?threshold=0.5&return_overlay=true`**, multipart form with field `file` = the image
(PNG/JPG). Response JSON:

| field | meaning |
|---|---|
| `tampered` | `true` if any pixel's tamper probability is above `threshold` |
| `tampered_pixels`, `tampered_fraction` | size of the flagged region |
| `max_score` | highest tamper probability on the page (0–1) |
| `mask_png_base64` | binary tamper mask (white = suspected forgery), PNG, base64 |
| `overlay_png_base64` | the page with the mask drawn in red (when `return_overlay=true`) |
| `height`, `width` | input size in pixels |

Python class: `TamperDetector(weights=, dinov3_ckpt=, device=, tile=512, overlap=384, batch=8)`. The weight paths
can also be set with the `DT_WEIGHTS` and `DT_DINOV3` environment variables. The Docker image sets them to
`/app/weights/...`.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `run-instances` fails with *VcpuLimitExceeded* | GPU quota is 0; do step 1 and wait for approval |
| `docker: could not select device driver "" with capabilities: [[gpu]]` | NVIDIA container toolkit is missing. Install it: `sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `docker build` fails with *no space left on device* | root volume too small; use 100 GB (step 3) or run `docker system prune -a` |
| `/health` says `"device":"cpu"` | the container started without `--gpus all`, or the GPU isn't visible; recheck step 4 |
| `FileNotFoundError: model weights not found` | `weights/` is empty or not mounted; rerun `scripts/download_weights.sh` and keep the `-v "$PWD/weights:/app/weights:ro"` mount |
| first request times out | normal model warm-up; send a `/health` and one test `/predict` after each start, and raise client / ALB timeouts to 120 s |
| `CUDA out of memory` | another process is using the GPU, or the page is huge; restart the container or use `-e DT_OVERLAP=128` |

**Testing status:** `requirements.txt`, the inference code and the FastAPI server were verified end-to-end in a clean
Python 3.10 environment. On a forged test page the detector found the edit with IoU 0.92 against the ground truth.
We have not yet run the Docker build itself (our cluster has no Docker access), so the first `docker build` on AWS is
the first real container build. Please report any problems.

---

## Licensing / provenance

- The **DINOv3** backbone and the vendored `vendor/dinov3_repo/` code are under Meta's DINOv3 license. Review its
  terms before commercial use.
- `model.pt` was fine-tuned on real academic documents collected for this project. No documents or datasets are
  included in this repository.
