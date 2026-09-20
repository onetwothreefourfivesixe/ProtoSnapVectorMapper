"""Turn network outputs into Wedge objects."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..data import Wedge, _order_heads
from .targets import REG_SCALE


@dataclass(frozen=True)
class DecodeConfig:
    threshold: float = 0.3      # apex heatmap score
    nms_kernel: int = 5
    max_wedges: int = 40
    snap_radius: float = 0.0    # px; snap head tips and tail to the nearest aux-heatmap peak within this radius (0 = off)
    snap_threshold: float = 0.3
    min_apex_sep: float = 3.0


def _local_peak(hm: np.ndarray, x: float, y: float, radius: float, thr: float) -> tuple[float, float] | None:
    H, W = hm.shape
    r = int(np.ceil(radius))
    x0, x1 = max(0, int(round(x)) - r), min(W, int(round(x)) + r + 1)
    y0, y1 = max(0, int(round(y)) - r), min(H, int(round(y)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    win = hm[y0:y1, x0:x1]
    iy, ix = np.unravel_index(int(win.argmax()), win.shape)
    if win[iy, ix] < thr:
        return None
    px, py = x0 + ix, y0 + iy
    if (px - x) ** 2 + (py - y) ** 2 > radius ** 2:
        return None
    # sub-pixel: intensity-weighted centroid of the 3x3 around the peak
    xa, xb, ya, yb = max(0, px - 1), min(W, px + 2), max(0, py - 1), min(H, py + 2)
    w = hm[ya:yb, xa:xb]
    ys, xs = np.mgrid[ya:yb, xa:xb]
    s = w.sum()
    return (float((w * xs).sum() / s), float((w * ys).sum() / s)) if s > 0 else (float(px), float(py))


def decode(out: dict[str, torch.Tensor], cfg: DecodeConfig = DecodeConfig()) -> tuple[list[Wedge], list[float]]:
    """`out` holds single-image tensors: apex (1,H,W) logits, aux (2,H,W) logits, reg (8,H,W)."""
    hm = torch.sigmoid(out["apex"].float())[None]                     # (1,1,H,W)
    pooled = F.max_pool2d(hm, cfg.nms_kernel, stride=1, padding=cfg.nms_kernel // 2)
    keep = (hm == pooled) & (hm > cfg.threshold)
    ys, xs = torch.nonzero(keep[0, 0], as_tuple=True)
    scores = hm[0, 0, ys, xs]
    order = torch.argsort(scores, descending=True)[: cfg.max_wedges]
    ys, xs, scores = ys[order].cpu().numpy(), xs[order].cpu().numpy(), scores[order].cpu().numpy()
    reg = out["reg"].float().cpu().numpy() * REG_SCALE
    aux = torch.sigmoid(out["aux"].float()).cpu().numpy() if cfg.snap_radius > 0 else None
    wedges, kept_scores = [], []
    for x, y, s in zip(xs, ys, scores):
        pts = reg[:, y, x].reshape(4, 2) + np.array([x, y], dtype=np.float64)
        if aux is not None:
            for k, ch in ((1, 0), (2, 0), (3, 1)):
                snapped = _local_peak(aux[ch], pts[k, 0], pts[k, 1], cfg.snap_radius, cfg.snap_threshold)
                if snapped is not None:
                    pts[k] = snapped
        apex = (float(pts[0, 0]), float(pts[0, 1]))
        if any((apex[0] - w.apex[0]) ** 2 + (apex[1] - w.apex[1]) ** 2 < cfg.min_apex_sep ** 2 for w in wedges):
            continue
        ha, hb = _order_heads(apex, tuple(map(float, pts[1])), tuple(map(float, pts[2])))
        wedges.append(Wedge(apex, ha, hb, tuple(map(float, pts[3]))))
        kept_scores.append(float(s))
    return wedges, kept_scores
