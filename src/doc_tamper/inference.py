"""Document tamper-localization inference: RGB + content-mask branch (config A3 / D_all_five).

Loads the dual-branch model and the 5-channel content-mask extractor, runs tiled sliding-window
inference, and returns a per-pixel tamper probability map (and a binary mask at a threshold).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "vendor" / "dualbranch"))
from dual_branch_model import build_dual_branch, _tile_starts   # noqa: E402
from .content_mask import ContentMaskExtractor, CHANNELS          # noqa: E402

DEFAULT_WEIGHTS = os.environ.get("DT_WEIGHTS", str(_HERE.parents[2] / "weights" / "model.pt"))
DEFAULT_DINOV3 = os.environ.get("DT_DINOV3", str(_HERE.parents[2] / "weights" / "dinov3_vitb16.pth"))
DINOV3_REPO = str(_HERE / "vendor" / "dinov3_repo")


class TamperDetector:
    """RGB + all-five content-mask tamper localizer (the deployed 'A3' configuration)."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, dinov3_ckpt: str = DEFAULT_DINOV3,
                 device: str | None = None, tile: int = 512, overlap: int = 384, batch: int = 8):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tile, self.overlap, self.batch = tile, overlap, batch
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"model weights not found: {weights}\nRun scripts/download_weights.sh first.")
        if not os.path.isfile(dinov3_ckpt):
            raise FileNotFoundError(f"DINOv3 backbone not found: {dinov3_ckpt}\nRun scripts/download_weights.sh first.")
        self.model = build_dual_branch(
            dinov3_ckpt=dinov3_ckpt, dinov3_repo_dir=DINOV3_REPO, noise_v0_ckpt=None, freeze_noise=False,
            decoder_embed_dim=512, lora_rank=8, lora_alpha=16.0, lora_blocks=6, noise_backbone="b0",
            fusion_mode="residual", semantic_channels=None, dropout_p=0.0, use_noise=True)
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        state = ck["model_state"] if "model_state" in ck else ck
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
        self.model.eval().to(self.device)
        self.content = ContentMaskExtractor(device=str(self.device))

    @torch.no_grad()
    def _predict_prob(self, bgr: np.ndarray, sem: np.ndarray) -> np.ndarray:
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype("float32") / 255.0)
        rgb = rgb.unsqueeze(0).to(self.device)
        semt = torch.from_numpy(sem.astype("float32") / 255.0).unsqueeze(0).to(self.device)
        _, _, H, W = rgb.shape
        st = self.tile - self.overlap
        coords = [(y, x) for y in _tile_starts(H, self.tile, st) for x in _tile_starts(W, self.tile, st)]
        acc = torch.zeros(H, W, device=self.device); wsum = torch.zeros(H, W, device=self.device)
        for i in range(0, len(coords), self.batch):
            ch = coords[i:i + self.batch]
            p = torch.cat([rgb[:, :, y:y + self.tile, x:x + self.tile] for y, x in ch], 0)
            s = torch.cat([semt[:, :, y:y + self.tile, x:x + self.tile] for y, x in ch], 0)
            pr = torch.softmax(self.model(p, s)[0].float(), 1)[:, 1]
            for j, (y, x) in enumerate(ch):
                ph, pw = pr.shape[-2:]
                acc[y:y + ph, x:x + pw] += pr[j]; wsum[y:y + ph, x:x + pw] += 1
        return (acc / wsum.clamp(min=1)).cpu().numpy()

    def predict(self, image, threshold: float = 0.5, return_content_mask: bool = False):
        """image: path or BGR ndarray. Returns dict with prob map, binary mask, tampered flag, score."""
        bgr = cv2.imread(image) if isinstance(image, (str, os.PathLike)) else image
        if bgr is None:
            raise ValueError("could not read image")
        sem = self.content.extract(bgr)                 # (5,H,W)
        prob = self._predict_prob(bgr, sem)             # (H,W) in [0,1]
        mask = (prob > threshold).astype(np.uint8) * 255
        flagged = int((prob > threshold).sum())
        out = dict(tampered=flagged > 0, tampered_pixels=flagged,
                   tampered_fraction=float((prob > threshold).mean()),
                   max_score=float(prob.max()), threshold=threshold,
                   prob_map=prob, mask=mask, height=bgr.shape[0], width=bgr.shape[1])
        if return_content_mask:
            out["content_mask"] = {c: sem[i] for i, c in enumerate(CHANNELS)}
        return out

    @staticmethod
    def overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Red overlay of the tamper mask on the page, for visual review."""
        o = bgr.copy(); m = mask > 0
        o[m] = (alpha * np.array([0, 0, 255]) + (1 - alpha) * o[m]).astype(np.uint8)
        return o
