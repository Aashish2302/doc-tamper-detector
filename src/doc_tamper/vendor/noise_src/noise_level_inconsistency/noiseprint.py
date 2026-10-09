"""Noiseprint++ encoder — DnCNN-style self-supervised noise fingerprint extractor.

Extracts a per-pixel device/processing fingerprint from an image.  The fingerprint
captures the characteristic noise pattern left by the camera sensor, scanner, or
rendering pipeline.  Tampering is detected as an anomaly: tampered regions produce
a fingerprint that deviates from the document's dominant authentic fingerprint.

Architecture (DnCNN-style residual learning):
    Input:  RGB (3ch, H, W)
    Body:   N conv layers (Conv3×3 + BN + ReLU), default N=12
    Output: 1ch noise fingerprint map (H, W)

    The network learns: fingerprint = f(image)
    NOT a denoiser — the output IS the fingerprint, not the denoised image.

Self-supervised pre-training:
    Train on authentic images only.  The objective is consistency — all patches
    from the same image should produce similar fingerprints (low variance within
    document), while patches from different images should differ (high variance
    across documents).  See ``pretrain_noiseprint.py`` for the training loop.

At inference:
    Run on the full document → 1ch fingerprint map.
    Compute per-block statistics → regions where the fingerprint deviates from
    the document's median fingerprint are flagged as potentially tampered.

Reference:
    Cozzolino & Verdoliva, "Noiseprint: a CNN-based camera model fingerprint,"
    IEEE TIFS, 2020.
    Guillaro et al., "TruFor: Leveraging all-round clues for trustworthy image
    forgery detection and localization," CVPR 2023.

CAVEAT (from architecture review):
    Noiseprint was designed for camera sensor fingerprints from photographs.
    It works for scanned documents (consistent sensor noise) but may learn
    nothing useful on digitally rendered documents (PDFs, Word exports) where
    there is no acquisition hardware.  Must validate on rendered vs scanned
    before relying on this branch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NoiseprintEncoder(nn.Module):
    """DnCNN-style fully convolutional fingerprint extractor.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 3 for RGB).
    mid_channels : int
        Width of intermediate conv layers (default 64).
    num_layers : int
        Total number of conv layers including first and last (default 12).
        Layers 2..N-1 have BN+ReLU.  First and last do not.
    out_channels : int
        Number of output fingerprint channels (default 1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        num_layers: int = 12,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        assert num_layers >= 3, "Need at least 3 layers (first + middle + last)"

        layers: list[nn.Module] = []

        # First layer: Conv only (no BN, no ReLU — preserve input statistics)
        layers.append(nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv only → produces the fingerprint
        layers.append(nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=True))

        self.body = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract noise fingerprint.

        Parameters
        ----------
        x : (B, 3, H, W) RGB image in [0, 1]

        Returns
        -------
        fingerprint : (B, 1, H, W) — per-pixel noise fingerprint
        """
        return self.body(x)


class NoiseprintAnomalyDetector(nn.Module):
    """Wraps NoiseprintEncoder + computes per-pixel anomaly score.

    The anomaly score measures how much each pixel's fingerprint deviates
    from the document's median fingerprint.  High anomaly = likely tampered.

    This module is used during NP-NLI inference but NOT during NP-NLI training
    (during training, the raw fingerprint features are fused into Branch C).
    """

    def __init__(self, encoder: NoiseprintEncoder, block_size: int = 8) -> None:
        super().__init__()
        self.encoder = encoder
        self.block_size = block_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fingerprint + anomaly map.

        Parameters
        ----------
        x : (B, 3, H, W) RGB image

        Returns
        -------
        fingerprint : (B, 1, H, W) — raw fingerprint
        anomaly_map : (B, 1, H, W) — per-pixel anomaly score (higher = more suspicious)
        """
        fp = self.encoder(x)  # (B, 1, H, W)

        # Compute document-level median fingerprint per image in the batch
        B = fp.shape[0]
        anomaly = torch.zeros_like(fp)
        for i in range(B):
            fp_i = fp[i, 0]  # (H, W)
            median_val = fp_i.median()
            mad = (fp_i - median_val).abs().median()  # median absolute deviation
            if mad > 1e-8:
                anomaly[i, 0] = ((fp_i - median_val).abs() / (mad * 1.4826)).clamp(0, 10)
            else:
                anomaly[i, 0] = (fp_i - median_val).abs()

        return fp, anomaly


def build_noiseprint_encoder(
    in_channels: int = 3,
    mid_channels: int = 64,
    num_layers: int = 12,
    out_channels: int = 1,
) -> NoiseprintEncoder:
    """Factory function for Noiseprint++ encoder."""
    return NoiseprintEncoder(
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_layers=num_layers,
        out_channels=out_channels,
    )
