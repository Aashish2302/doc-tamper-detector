#!/usr/bin/env python
"""Robustness-improved v2 detectors for stamp / logo / watermark.

Design notes (why each change):
 * WATERMARK v1 failed because "some texture + light tone" also describes
   paper grain, scan noise and JPEG ringing. v2 replaces the texture/tone
   heuristic with (a) an explicit background-residual AMPLITUDE band -- a
   watermark is faint but far above sensor noise -- and (b) a PERIODICITY
   test on the residual (a real watermark repeats; grain does not).
 * LOGO v1 was a hand-tuned position+shape rule that cannot generalise.
   v2 uses a pretrained DocLayNet layout model and takes its `Picture`
   class directly.
 * STAMP v1 fired on any saturated pixel, so on colour-printed certificates
   it flagged the whole background. v2 requires LOCAL saturation contrast
   (a stamp is more saturated than its immediate surroundings) plus a
   compactness/ink-density test.
 * SHARED: every detector gets a coverage cap. A prior covering most of the
   page carries no information, so it is safer to emit nothing.
"""
import numpy as np, cv2

# A prior covering more than this fraction of the page carries ~no
# information; emitting nothing is strictly safer than emitting noise.
COVERAGE_CAP = 0.35


def dilate(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    return cv2.dilate(mask, k)


def _cap(mask, cap=COVERAGE_CAP):
    """Reject a detector output that covers implausibly much of the page."""
    if mask is None:
        return np.zeros((1, 1), np.uint8), 0.0, False
    covg = float((mask > 0).mean())
    if covg > cap:
        return np.zeros_like(mask), covg, True
    return mask, covg, False


# ----------------------------------------------------------------- WATERMARK
def watermark_mask_v2(rgb, exclude_mask, cap=COVERAGE_CAP,
                      amp_lo=4.0, amp_hi=45.0, peak_ratio=1.35):
    """Background-residual amplitude band + block-level periodicity test."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # Local background (paper tone). Kernel must exceed watermark stroke width.
    bg = cv2.medianBlur(gray.astype(np.uint8), 31).astype(np.float32)
    resid = bg - gray                      # watermark is darker than paper

    # Ink (text/lines) is far darker than any watermark -> exclude it.
    ink = (resid > amp_hi)
    excl = (exclude_mask > 0) | ink
    excl_d = cv2.dilate(excl.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    blk = 64
    hb, wb = h // blk, w // blk
    if hb < 2 or wb < 2:
        return np.zeros((h, w), np.uint8), 0.0, False, {}

    keep_blocks = np.zeros((hb, wb), np.uint8)
    for i in range(hb):
        for j in range(wb):
            ys, xs = slice(i * blk, (i + 1) * blk), slice(j * blk, (j + 1) * blk)
            if excl_d[ys, xs].mean() > 0.25:
                continue
            r = resid[ys, xs].copy()
            m = ~excl_d[ys, xs]
            if m.sum() < 0.5 * r.size:
                continue
            vals = r[m]
            # (a) amplitude band: faint, but clearly above sensor noise
            amp = float(np.percentile(vals, 97) - np.percentile(vals, 50))
            if not (amp_lo < amp < amp_hi):
                continue
            # (b) periodicity: dominant non-DC frequency peak in the block
            rr = r - r.mean()
            rr[~m] = 0.0
            F = np.abs(np.fft.rfft2(rr))
            F[0, 0] = 0.0
            if F.size < 8:
                continue
            flat = np.sort(F.ravel())[::-1]
            peak = float(flat[:3].mean())
            base = float(np.median(F))
            if base <= 1e-6:
                continue
            if peak / base < peak_ratio * 8:   # peak must dominate the spectrum
                continue
            keep_blocks[i, j] = 255

    if keep_blocks.sum() == 0:
        return np.zeros((h, w), np.uint8), 0.0, False, dict(blocks=0)

    # A watermark is a page-scale phenomenon: keep only large block clusters.
    kb = cv2.morphologyEx(keep_blocks, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(kb, connectivity=8)
    keep = np.zeros_like(kb)
    nblk = hb * wb
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.02 * nblk:
            keep[lbl == i] = 255

    full = cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST)
    full[exclude_mask > 0] = 0
    out, covg, rejected = _cap(full, cap)
    return out, covg, rejected, dict(blocks=int((keep_blocks > 0).sum()), nblk=nblk)


# ---------------------------------------------------------------------- LOGO
def logo_mask_from_layout(layout_boxes, shape, dilate_px=20,
                          cap=COVERAGE_CAP, max_frac=0.25):
    """Build a logo mask from DocLayNet `Picture` detections."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    used = []
    page_area = h * w
    for (cls, conf, x0, y0, x1, y1) in layout_boxes:
        if cls != "Picture":
            continue
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        # a "Picture" spanning most of the page is the scan itself, not a logo
        if (x1 - x0) * (y1 - y0) > max_frac * page_area:
            continue
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        used.append((conf, x0, y0, x1, y1))
    mask = dilate(mask, dilate_px)
    out, covg, rejected = _cap(mask, cap)
    return out, covg, rejected, used


# --------------------------------------------------------------------- STAMP
def stamp_mask_v2(rgb, cap=COVERAGE_CAP, sat_margin=35):
    """Require LOCAL saturation contrast, not just absolute saturation, so a
    colour-printed background does not trigger the whole page."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    hue = hsv[..., 0]

    # local saturation baseline: what the surrounding paper looks like
    k = max(31, (min(rgb.shape[:2]) // 20) | 1)
    sat_bg = cv2.medianBlur(hsv[..., 1], k).astype(np.float32)
    local_pop = (sat - sat_bg) > sat_margin      # stands out from its background

    reddish = (hue < 15) | (hue > 165)
    purplish = (hue >= 130) & (hue <= 165)
    bluish = (hue >= 95) & (hue <= 129)
    stamp_hue = reddish | purplish | bluish

    mask = (local_pop & stamp_hue & (sat > 55) & (val > 50) & (val < 250))
    mask = (mask.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    page_area = rgb.shape[0] * rgb.shape[1]
    keep = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (0.0002 * page_area < area < 0.04 * page_area):
            continue
        ar = bw / max(1, bh)
        if not (0.25 < ar < 4.0):            # reject long thin colour bands
            continue
        if area / max(1, bw * bh) < 0.12:    # reject wispy scatter
            continue
        keep[lbl == i] = 255

    keep = dilate(keep, 15)
    out, covg, rejected = _cap(keep, cap)
    return out, covg, rejected
