"""Failure-class breakdown shared by the Stage 2 and Stage 3 reports, so the classical and
learned models are described in the same terms."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .data import load_image
from .metrics import DEFAULT_MATCH_THRESHOLD, match_wedges

TAIL_ARC = (-40.0, 130.0)   # 95% of ground-truth tails point inside this arc (0 = right, 90 = down)
CLASSES = [
    "head tip off by >10 px",
    "head tip off by >10 px (rare orientation)",
    "prong and tail swapped",
    "tail stopped early (>10 px short)",
    "tail overshoot or sideways (>10 px)",
    "tail off by >10 px (rare orientation)",
    "near miss: FN with a prediction 10-20 px away",
    "near miss: FP within 10-20 px of a wedge",
    "missed wedge (FN, nothing nearby)",
    "spurious wedge (FP, nothing nearby)",
]


def failure_classes(predictor, pairs, skeletons, threshold: float = DEFAULT_MATCH_THRESHOLD) -> Counter:
    c: Counter = Counter({k: 0 for k in CLASSES})
    for p in pairs:
        gt = skeletons[p.key].wedges
        pred = predictor.predict(load_image(p), p)
        m, up, ug = match_wedges(pred, gt, threshold)
        for gi in ug:
            near = min((math.dist(w.apex, gt[gi].apex) for w in pred), default=99.0)
            c["near miss: FN with a prediction 10-20 px away" if near < 20 else "missed wedge (FN, nothing nearby)"] += 1
        for pi in up:
            near = min((math.dist(pred[pi].apex, g.apex) for g in gt), default=99.0)
            c["near miss: FP within 10-20 px of a wedge" if near < 20 else "spurious wedge (FP, nothing nearby)"] += 1
        for mm in m:
            P, G = pred[mm.pred_index].as_array(), gt[mm.gt_index].as_array()
            t = G[3] - G[0]
            ang = math.degrees(math.atan2(t[1], t[0]))
            rare = not (TAIL_ARC[0] <= ang <= TAIL_ARC[1])
            if max(mm.point_errors[1], mm.point_errors[2]) > 10:
                swap = min(np.linalg.norm(P[k] - G[3]) for k in (1, 2)) < 6
                c["prong and tail swapped" if swap else ("head tip off by >10 px (rare orientation)" if rare else "head tip off by >10 px")] += 1
            if mm.point_errors[3] > 10:
                along = float((P[3] - G[3]) @ (t / (np.linalg.norm(t) + 1e-9)))
                if rare:
                    c["tail off by >10 px (rare orientation)"] += 1
                else:
                    c["tail stopped early (>10 px short)" if along < -5 else "tail overshoot or sideways (>10 px)"] += 1
    return c


def failure_table(columns: dict[str, Counter]) -> str:
    names = list(columns)
    lines = ["| Failure class | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in CLASSES:
        lines.append(f"| {k} | " + " | ".join(str(columns[n][k]) for n in names) + " |")
    return "\n".join(lines)
