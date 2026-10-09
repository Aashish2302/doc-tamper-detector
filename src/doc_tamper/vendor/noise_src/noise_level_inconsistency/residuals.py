"""Noise residual extraction for NLI.

Provides four residual modes, ordered from weakest to strongest at isolating
true acquisition noise while suppressing document structure:

  laplacian   Classic 3×3 Laplacian high-pass.  Fast; responds strongly to
              edges and text strokes — NOT recommended for production NLI.

  wavelet     One-level Haar wavelet detail energy.  Better than Laplacian
              but still edge-sensitive.

  denoiser    Noise residual = original − median_filtered(original).
              The median filter suppresses both noise AND edges, so the residual
              contains mostly noise with greatly reduced edge response.
              Recommended default for document images.

  srm         Simplified Spatial Rich Model: average of five linear predictors
              chosen to emphasise noise while cancelling local signal.
              Provides a complementary view to the denoiser residual.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shared convolution helper
# ---------------------------------------------------------------------------

def convolve_3x3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3×3 convolution with reflect-padding. Returns float32 same shape as image."""
    padded = np.pad(image, 1, mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(3):
        for col in range(3):
            output += kernel[row, col] * padded[row : row + image.shape[0], col : col + image.shape[1]]
    return output


# ---------------------------------------------------------------------------
# Laplacian residual (legacy; kept for backward compatibility)
# ---------------------------------------------------------------------------

def laplacian_residual(gray: np.ndarray) -> np.ndarray:
    """Absolute Laplacian response (8-neighbour kernel)."""
    kernel = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    return np.abs(convolve_3x3(gray, kernel))


# ---------------------------------------------------------------------------
# Wavelet residual (legacy; kept for backward compatibility)
# ---------------------------------------------------------------------------

def _haar_detail_energy(image: np.ndarray) -> np.ndarray:
    height = image.shape[0] - (image.shape[0] % 2)
    width = image.shape[1] - (image.shape[1] % 2)
    cropped = image[:height, :width]
    if height == 0 or width == 0:
        return np.zeros_like(image, dtype=np.float32)

    a = cropped[0::2, 0::2]
    b = cropped[0::2, 1::2]
    c = cropped[1::2, 0::2]
    d = cropped[1::2, 1::2]

    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    energy = np.sqrt(lh * lh + hl * hl + hh * hh)
    upsampled = np.repeat(np.repeat(energy, 2, axis=0), 2, axis=1)
    if upsampled.shape != image.shape:
        padded = np.zeros_like(image, dtype=np.float32)
        padded[: upsampled.shape[0], : upsampled.shape[1]] = upsampled
        return padded
    return upsampled.astype(np.float32, copy=False)


def wavelet_residual(gray: np.ndarray) -> np.ndarray:
    """Haar wavelet detail energy (one level)."""
    return _haar_detail_energy(gray)


# ---------------------------------------------------------------------------
# Denoiser residual (Phase 2)
# ---------------------------------------------------------------------------

def _median_filter_2d(image: np.ndarray, radius: int) -> np.ndarray:
    """Pure-numpy 2-D median filter using stride-trick sliding window views.

    Available in numpy ≥ 1.20 (project requires ≥ 1.26). The sliding window
    view creates a *view* (no copy) so the only allocation is the final
    median output array.
    """
    ksize = 2 * radius + 1
    padded = np.pad(image.astype(np.float32), radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
    # windows shape: (H, W, ksize, ksize) — compute median over last two axes
    return np.median(windows, axis=(-2, -1)).astype(np.float32)


def denoiser_residual(gray: np.ndarray, radius: int = 2) -> np.ndarray:
    """Noise residual = |original − median_filtered|.

    The median filter preserves edges while smoothing noise, so the residual
    captures mostly noise.  Edge responses are dramatically reduced compared
    to the Laplacian, making this far better suited for NLI on document images
    where text strokes would otherwise dominate.

    Parameters
    ----------
    radius:
        Half-width of the median-filter kernel (kernel size = 2·radius + 1).
        Default 2 → 5×5 kernel.  Larger radius = more noise suppression but
        slower.
    """
    denoised = _median_filter_2d(gray, radius)
    return np.abs(gray.astype(np.float32) - denoised)


# ---------------------------------------------------------------------------
# SRM-inspired residual (Phase 2)
# ---------------------------------------------------------------------------

def srm_residual(gray: np.ndarray) -> np.ndarray:
    """Simplified Spatial Rich Model residual.

    Averages the absolute responses of five small linear predictors designed
    to suppress local DC signal while amplifying noise:

      K1  horizontal 1st-order difference  [-0.5, 0, 0.5]
      K2  vertical   1st-order difference  (transposed K1)
      K3  horizontal 2nd-order Laplacian   [0.25, -0.5, 0.25]
      K4  vertical   2nd-order Laplacian   (transposed K3)
      K5  diagonal   cross-term            corner pattern

    The average suppresses direction bias and provides a more isotropic
    noise estimate than any single kernel.
    """
    k1 = np.asarray([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    k2 = np.asarray([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    k3 = np.asarray([[0.0, 0.0, 0.0], [0.25, -0.5, 0.25], [0.0, 0.0, 0.0]], dtype=np.float32)
    k4 = np.asarray([[0.0, 0.25, 0.0], [0.0, -0.5, 0.0], [0.0, 0.25, 0.0]], dtype=np.float32)
    k5 = np.asarray([[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]], dtype=np.float32)

    acc = np.zeros_like(gray, dtype=np.float32)
    for kernel in (k1, k2, k3, k4, k5):
        acc += np.abs(convolve_3x3(gray, kernel))
    return acc / 5.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_residual(gray: np.ndarray, mode: str, **kwargs: object) -> np.ndarray:
    """Return the noise residual array for *gray* using the given *mode*.

    Supported modes: ``laplacian``, ``wavelet``, ``denoiser``, ``srm``.
    Extra keyword arguments are forwarded to the mode-specific function
    (e.g. ``radius`` for the denoiser).
    """
    selected = mode.lower()
    if selected == "laplacian":
        return laplacian_residual(gray)
    if selected == "wavelet":
        return wavelet_residual(gray)
    if selected == "denoiser":
        return denoiser_residual(gray, radius=int(kwargs.get("radius", 2)))
    if selected == "srm":
        return srm_residual(gray)
    raise ValueError(f"Unsupported residual mode: {mode!r}. Choose from: laplacian, wavelet, denoiser, srm")
