"""Training targets for the wedge detector.

Outputs are full resolution (stride 1):
  apex_hm   (1, H, W)  Gaussian peak at every wedge apex
  aux_hm    (2, H, W)  Gaussian peaks at head tips (channel 0) and tail ends (channel 1)
  reg       (8, H, W)  at pixels within reg_radius of an apex pixel: offsets from that pixel to
                       apex, head_a, head_b, tail, divided by REG_SCALE
  reg_mask  (1, H, W)  1 where reg is supervised
  valid     (1, H, W)  0 around strokes that are not canonical wedges (triangles, malformed):
                       their heads are in the image but have no wedge label, so the heatmap
                       losses ignore those regions instead of treating them as negatives
"""
from __future__ import annotations

import numpy as np

REG_SCALE = 32.0
POINT_ORDER = ("apex", "head_a", "head_b", "tail")


def _splat(hm: np.ndarray, x: float, y: float, sigma: float) -> None:
    """Gaussian peak centred on the pixel nearest (x, y), so the peak value is exactly 1.

    The focal loss treats only pixels above 0.99 as positives. Centred on the exact sub-pixel
    position, a peak reaches 0.99 only when the point lies within 0.28 px of a pixel centre, which
    after augmentation or rescaling is about a quarter of the time, so most wedges had no positive
    pixel at all. The sub-pixel part is carried by the offset regression instead."""
    x, y = float(round(x)), float(round(y))
    H, W = hm.shape
    r = int(3 * sigma + 1)
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 2)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 2)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1)[None, :]
    ys = np.arange(y0, y1)[:, None]
    g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    np.maximum(hm[y0:y1, x0:x1], g, out=hm[y0:y1, x0:x1])


def build_targets(wedges: np.ndarray, ignore_points: np.ndarray, size: int = 256, sigma: float = 2.0,
                  reg_radius: int = 1, ignore_radius: int = 14) -> dict[str, np.ndarray]:
    """wedges: (N, 4, 2) float in the order apex, head_a, head_b, tail. ignore_points: (M, 2)."""
    apex_hm = np.zeros((size, size), np.float32)
    aux = np.zeros((2, size, size), np.float32)
    reg = np.zeros((8, size, size), np.float32)
    reg_mask = np.zeros((1, size, size), np.float32)
    valid = np.ones((1, size, size), np.float32)
    for w in wedges:
        ax, ay = w[0]
        if not (0 <= ax < size and 0 <= ay < size):
            continue
        _splat(apex_hm, ax, ay, sigma)
        for k in (1, 2):
            _splat(aux[0], w[k, 0], w[k, 1], sigma)
        _splat(aux[1], w[3, 0], w[3, 1], sigma)
        cx, cy = int(round(ax)), int(round(ay))
        for py in range(max(0, cy - reg_radius), min(size, cy + reg_radius + 1)):
            for px in range(max(0, cx - reg_radius), min(size, cx + reg_radius + 1)):
                reg[:, py, px] = ((w - np.array([px, py])) / REG_SCALE).reshape(-1)
                reg_mask[0, py, px] = 1.0
    if len(ignore_points):
        ys, xs = np.mgrid[0:size, 0:size]
        for x, y in ignore_points:
            valid[0][(xs - x) ** 2 + (ys - y) ** 2 <= ignore_radius ** 2] = 0.0
        # never ignore a labelled apex neighbourhood
        valid[0][apex_hm > 0.3] = 1.0
    return {"apex_hm": apex_hm[None], "aux_hm": aux, "reg": reg, "reg_mask": reg_mask, "valid": valid}
