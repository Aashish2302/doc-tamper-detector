"""Per-channel dropout on the semantic guidance map.

Independently zeroes each of the 5 semantic channels with probability `p`
during training only.

Why per-CHANNEL and not per-element: at inference, detectors fail as *whole
channels* -- the watermark detector misses the watermark on a page, the layout
model finds no logo. Element-wise dropout simulates speckle, which is not the
failure mode we face. Zeroing whole channels during training forces the model
to stay functional when a detector produces nothing, instead of becoming
dependent on a channel that is measurably unreliable in deployment.

No rescaling is applied. Standard dropout divides by (1-p) to preserve
expected activation magnitude, but these channels are occupancy masks in
[0, 1] fed to a BatchNorm'd conv stem, and a 1/(1-p) rescale would push a
present channel to values never seen at eval. Keeping the train and eval
value ranges identical matters more here than matching expectations.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SemanticChannelDropout(nn.Module):
    """Randomly drop entire semantic channels (train mode only).

    Args:
        p: per-channel drop probability.
        keep_at_least_one: if True, never return an all-zero map -- one channel
            is retained at random. Prevents degenerate all-dropped samples from
            contributing a gradient signal that says "the prior is always
            useless".
    """

    def __init__(self, p: float = 0.15, keep_at_least_one: bool = True):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p
        self.keep_at_least_one = keep_at_least_one

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic: (B, C, H, W)
        Returns:
            same shape; in eval mode returned unchanged.
        """
        if not self.training or self.p == 0.0:
            return semantic
        B, C = semantic.shape[0], semantic.shape[1]
        keep = (torch.rand(B, C, device=semantic.device) >= self.p)
        if self.keep_at_least_one:
            empty = ~keep.any(dim=1)
            if empty.any():
                idx = torch.randint(0, C, (int(empty.sum()),), device=semantic.device)
                keep[empty, idx] = True
        return semantic * keep.to(semantic.dtype).unsqueeze(-1).unsqueeze(-1)

    def extra_repr(self) -> str:
        return f"p={self.p}, keep_at_least_one={self.keep_at_least_one}"
