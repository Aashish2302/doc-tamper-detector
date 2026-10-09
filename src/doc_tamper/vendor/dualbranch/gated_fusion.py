"""Per-scale fusion of the vision branch (A) and the semantic branch (B).

Two variants:

  GatedResidualFusion   (the proposal)
      G^k        = sigmoid(g_k(F_s^k))
      F_fuse^k   = F_v^k + alpha_k * G^k (*) psi_k(F_s^k)

  MultiplicativeFusion  (ablation config F -- the thing we argue against)
      F_fuse^k   = F_v^k (*) sigmoid(g_k(F_s^k))

Why the residual form is the default
------------------------------------
`F_v^k` is on an unconditional path, so however wrong the semantic map is, the
vision branch's own evidence still reaches the decoder. A bad prior can fail to
*help*; it cannot *erase* a confident detection. That property is load-bearing
here because two of the five semantic channels (watermark, stamp) are measured
to be unreliable -- the multiplicative form would let a single misfiring
detector delete a correct prediction, which is exactly the failure already
observed with the hard post-hoc mask.

`alpha_k` is a learnable scalar per scale, initialised to **exactly zero**, so
at initialisation `F_fuse^k == F_v^k` bit-for-bit and the dual-branch model
reproduces the single-branch model. The model then has to *earn* its way off
zero by reducing training loss. It doubles as a free diagnostic: if the trained
alphas stay ~0, Branch B contributed nothing and the checkpoint says so.

Consequence of alpha_init=0 you must know before training
---------------------------------------------------------
This is the ReZero/LayerScale trick, and it has a property that looks like a
bug the first time you see it in a gradient check:

    d(F_fuse)/d(F_s) = alpha * d(G (*) psi(F_s))/d(F_s) = 0   when alpha == 0

so at step 0 the **semantic encoder receives exactly zero gradient**. Only
`alpha` itself gets a gradient (its own derivative is `G (*) psi(F_s)`, which
is non-zero). Training is therefore sequential: alpha lifts off zero on the
first optimiser step, and from step 1 onward gradient reaches Branch B
normally. It self-starts and does not deadlock.

It does mean Branch B learns nothing on step 0, and if alpha's gradient is
very small it will stay near-dormant for a while. If Branch B looks dormant
in practice, `alpha_init=1e-2` trades exact single-branch equivalence at init
for an immediately-training Branch B. Equivalence is then approximate, so the
config-A-reproduction check must be relaxed accordingly.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class GatedResidualFusion(nn.Module):
    """Gated residual fusion at a single scale."""

    def __init__(self, channels: int, kernel_size: int = 1,
                 alpha_init: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        # gate: how much to trust the semantic feature, per position/channel
        self.gate = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        # transform: what the semantic feature contributes
        self.transform = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        # scalar mixing weight, learnable, starts at 0
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, f_v: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        if f_v.shape[-2:] != f_s.shape[-2:]:
            raise ValueError(
                f"spatial mismatch: vision {tuple(f_v.shape[-2:])} vs "
                f"semantic {tuple(f_s.shape[-2:])}")
        g = torch.sigmoid(self.gate(f_s))
        return f_v + self.alpha * g * self.transform(f_s)


class MultiplicativeFusion(nn.Module):
    """Hard multiplicative gating -- ablation config F only.

    Included so the design claim can be tested rather than asserted. Note there
    is no residual path: if the gate is near zero the vision signal is gone.
    """

    def __init__(self, channels: int, kernel_size: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.gate = nn.Conv2d(channels, channels, kernel_size, padding=pad)

    def forward(self, f_v: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        if f_v.shape[-2:] != f_s.shape[-2:]:
            raise ValueError(
                f"spatial mismatch: vision {tuple(f_v.shape[-2:])} vs "
                f"semantic {tuple(f_s.shape[-2:])}")
        return f_v * torch.sigmoid(self.gate(f_s))


class MultiScaleFusion(nn.Module):
    """Applies a fusion module at each of the 4 encoder scales."""

    MODES = ("residual", "multiplicative")

    def __init__(self, channels: Sequence[int], mode: str = "residual",
                 kernel_size: int = 1, alpha_init: float = 0.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        if mode == "residual":
            self.blocks = nn.ModuleList([
                GatedResidualFusion(c, kernel_size, alpha_init) for c in channels
            ])
        else:
            self.blocks = nn.ModuleList([
                MultiplicativeFusion(c, kernel_size) for c in channels
            ])

    def forward(self, vision_feats: List[torch.Tensor],
                semantic_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(vision_feats) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} vision scales, got {len(vision_feats)}")
        if len(semantic_feats) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} semantic scales, got {len(semantic_feats)}")
        return [blk(fv, fs) for blk, fv, fs
                in zip(self.blocks, vision_feats, semantic_feats)]

    def alphas(self) -> List[float]:
        """Current alpha per scale (empty for the multiplicative variant).

        Watch these during training: all-near-zero means Branch B is not
        contributing and the semantic prior is not earning its place.
        """
        if self.mode != "residual":
            return []
        return [float(b.alpha.detach().cpu()) for b in self.blocks]
