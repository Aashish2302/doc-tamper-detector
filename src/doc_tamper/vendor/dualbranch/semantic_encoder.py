"""Branch B: semantic-region encoder.

Consumes the stacked semantic guidance map

    S(x, y) = [P_text, P_logo, P_sig, P_stamp, P_water]      (B, 5, H, W)

and emits features at exactly the four scales the DINOv3 branch produces, so
they can be fused per-scale by `gated_fusion.GatedResidualFusion`.

Scales/channels are NOT guessed -- they were read off the real reconfix
checkpoint (`encoder.neck*`, `decoder.linears.*.fc.weight`):

    stride   /4    /8    /16   /32
    channels 64    128   320   512

The encoder is a plain strided conv-BN-ReLU cascade: a stem that takes the map
down to /4, then one downsampling stage per subsequent scale, with the feature
at each scale tapped off the cascade. That keeps the parameter count near 2M
(vs ~86M for the DINOv3 branch) -- Branch B is meant to inject a prior, not to
be a second backbone.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

# Must match noise_level_inconsistency.trufor_dinob.dinob_cmx_encoder.NECK_CHANNELS
NECK_CHANNELS: List[int] = [64, 128, 320, 512]

# Canonical channel order of the semantic map. Kept here so every consumer
# (encoder, dropout, dataset writer) agrees on which index means what.
SEMANTIC_CHANNELS: List[str] = ["text", "logo", "signature", "stamp", "watermark"]


def _cbr(in_ch: int, out_ch: int, stride: int = 1, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride,
                  padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _ds_cbr(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable conv-BN-ReLU.

    Used for the two widest downsampling stages. A dense 3x3 at 320->512 alone
    costs 1.47M parameters -- more than the whole rest of the branch -- and
    Branch B only has to encode a handful of near-binary occupancy masks, so
    that capacity is not needed. Separating it costs ~167k for the same
    receptive field.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SemanticChannelGate(nn.Module):
    """Input-conditioned, per-channel trust gate applied BEFORE the shared
    encoder trunk mixes the 5 semantic channels together.

    Root cause this targets: `SemanticEncoder`'s stem convolves all 5 input
    channels together in its very first layer, so a noisy channel (watermark,
    stamp -- still imperfect after Stage-2 robustness work) contaminates the
    shared feature that `GatedResidualFusion`'s single per-scale `alpha` then
    trusts or distrusts as a whole. Alpha has no way to keep a good channel
    while discarding a bad one, because by the time it sees anything, they are
    already mixed. This gate acts strictly earlier: it squeezes each channel
    to a scalar (global average pool), predicts a per-channel trust weight
    from that channel's OWN activation via a depthwise (groups=num_channels)
    1x1-conv MLP, and rescales the channel before the stem ever sees it -- so
    a channel that fired confidently but is characteristically unreliable in
    this image can be suppressed independently of the others.

    Initialised near pass-through (weight=0, bias=+4 -> sigmoid~=0.98) so it
    does not change behaviour before training, and does not interact with the
    alpha=0 zero-init-equivalence property in gated_fusion.py at all: that
    property is guaranteed by alpha alone, regardless of what this gate does
    to the semantic branch's *input*.
    """

    def __init__(self, num_channels: int, hidden: int = 4, init_bias: float = 4.0):
        super().__init__()
        self.num_channels = num_channels
        self.fc1 = nn.Conv2d(num_channels, num_channels * hidden, 1, groups=num_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(num_channels * hidden, num_channels, 1, groups=num_channels)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_bias)
        self.last_gate: torch.Tensor | None = None  # (B, C), for inspection/logging

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(pooled))))
        self.last_gate = gate.detach().view(x.shape[0], self.num_channels)
        return x * gate


class SemanticEncoder(nn.Module):
    """5-channel semantic map -> 4-scale features matching the vision branch.

    Args:
        in_channels: number of semantic channels fed in (5 by default).
        out_channels: per-scale output channels; must match the vision branch.
        width: stem width multiplier knob.
    """

    def __init__(
        self,
        in_channels: int = len(SEMANTIC_CHANNELS),
        out_channels: Sequence[int] = tuple(NECK_CHANNELS),
        width: int = 32,
    ):
        super().__init__()
        out_channels = list(out_channels)
        if len(out_channels) != 4:
            raise ValueError(f"expected 4 scales, got {len(out_channels)}")
        self.in_channels = in_channels
        self.out_channels = out_channels

        # stem: full res -> /4
        self.stem = nn.Sequential(
            _cbr(in_channels, width, stride=2),      # /2
            _cbr(width, width, stride=1),
            _cbr(width, out_channels[0], stride=2),  # /4
        )
        # one downsample stage per subsequent scale; the two widest use
        # depthwise-separable convs to keep the branch near 1M params
        self.down1 = _cbr(out_channels[0], out_channels[1], stride=2)      # /8
        self.down2 = _cbr(out_channels[1], out_channels[2], stride=2)      # /16
        self.down3 = _ds_cbr(out_channels[2], out_channels[3], stride=2)   # /32

        # per-scale refinement so each tap is not merely the raw downsample.
        # 1x1: spatial context already comes from the strided cascade, and a
        # 3x3 here would add 3.5M params for little benefit.
        self.refine = nn.ModuleList([_cbr(c, c, stride=1, k=1) for c in out_channels])

    def forward(self, semantic: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            semantic: (B, C_in, H, W), values in [0, 1].
        Returns:
            [S0, S1, S2, S3] at strides [/4, /8, /16, /32] with channels
            matching `out_channels`.
        """
        if semantic.dim() != 4:
            raise ValueError(f"expected (B,C,H,W), got {tuple(semantic.shape)}")
        if semantic.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} semantic channels, "
                f"got {semantic.shape[1]}")
        s0 = self.stem(semantic)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        feats = [s0, s1, s2, s3]
        return [r(f) for r, f in zip(self.refine, feats)]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_semantic_encoder(
    in_channels: int = len(SEMANTIC_CHANNELS),
    out_channels: Sequence[int] = tuple(NECK_CHANNELS),
) -> SemanticEncoder:
    return SemanticEncoder(in_channels=in_channels, out_channels=out_channels)
