"""Stage 1 evaluation harness.

A prediction for a glyph is a list of canonical Wedge objects. Ground truth is
Skeleton.wedges (triangle and malformed strokes are not evaluated).

Matching: Hungarian assignment between predicted and ground-truth wedges on apex
distance; an assigned pair counts as a match only if its apex distance is at most
`match_threshold` pixels.

Per glyph:
  tp, fp, fn            matched, unmatched predicted, unmatched ground-truth wedges
  corner_error          mean over matched wedges of the mean distance over the 4 points
                        (apex, head_a, head_b, tail); NaN when nothing matched
  exact                 fp == fn == 0 and every matched point within `match_threshold`

Aggregates (overall and per font): micro precision/recall/F1 over wedges, mean
corner error over matched wedges, per-point-type errors, glyph exact-match rate.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import KIND_WEDGE, Pair, Skeleton, Wedge, load_image, load_skeleton

DEFAULT_MATCH_THRESHOLD = 10.0  # px; median head width is ~21 px
POINT_NAMES = ("apex", "head_a", "head_b", "tail")


class Predictor(Protocol):
    name: str

    def predict(self, image: np.ndarray, pair: Pair) -> list[Wedge]: ...


# --------------------------------------------------------------------------- matching

@dataclass
class Match:
    pred_index: int
    gt_index: int
    apex_distance: float
    point_errors: tuple[float, float, float, float]   # apex, head_a, head_b, tail

    @property
    def corner_error(self) -> float:
        return float(np.mean(self.point_errors))


def match_wedges(pred: Sequence[Wedge], gt: Sequence[Wedge], match_threshold: float = DEFAULT_MATCH_THRESHOLD
                 ) -> tuple[list[Match], list[int], list[int]]:
    """Return (matches, unmatched_pred_indices, unmatched_gt_indices)."""
    if not pred or not gt:
        return [], list(range(len(pred))), list(range(len(gt)))
    P = np.array([w.as_array() for w in pred])   # (np, 4, 2)
    G = np.array([w.as_array() for w in gt])     # (ng, 4, 2)
    apex_cost = np.linalg.norm(P[:, None, 0, :] - G[None, :, 0, :], axis=-1)
    rows, cols = linear_sum_assignment(apex_cost)
    matches, used_p, used_g = [], set(), set()
    for i, j in zip(rows, cols):
        if apex_cost[i, j] <= match_threshold:
            errs = np.linalg.norm(P[i] - G[j], axis=-1)
            matches.append(Match(int(i), int(j), float(apex_cost[i, j]), tuple(float(e) for e in errs)))
            used_p.add(int(i))
            used_g.add(int(j))
    return (matches,
            [i for i in range(len(pred)) if i not in used_p],
            [j for j in range(len(gt)) if j not in used_g])


# --------------------------------------------------------------------------- per-glyph

@dataclass
class GlyphResult:
    key: str
    font: str
    codepoint: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    corner_error: float                 # NaN if tp == 0
    max_point_error: float              # NaN if tp == 0
    exact: bool
    point_errors: list[list[float]] = field(default_factory=list)  # per match, 4 values
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)
    matches: list[tuple[int, int]] = field(default_factory=list)   # (pred_index, gt_index)

    @property
    def severity(self) -> float:
        """Sorting key for the failure gallery: worse first."""
        err = 0.0 if math.isnan(self.corner_error) else self.corner_error
        return (self.fp + self.fn) * 100.0 + err


def score_glyph(pair: Pair, pred: Sequence[Wedge], gt: Sequence[Wedge],
                match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> GlyphResult:
    matches, up, ug = match_wedges(pred, gt, match_threshold)
    tp = len(matches)
    errs = [list(m.point_errors) for m in matches]
    corner = float(np.mean([m.corner_error for m in matches])) if tp else float("nan")
    maxpt = float(max(max(m.point_errors) for m in matches)) if tp else float("nan")
    exact = (not up) and (not ug) and (tp == 0 or maxpt <= match_threshold)
    return GlyphResult(pair.key, pair.font, pair.codepoint, len(gt), len(pred), tp, len(up), len(ug),
                       corner, maxpt, exact, errs, up, ug, [(m.pred_index, m.gt_index) for m in matches])


# --------------------------------------------------------------------------- aggregates

@dataclass
class Aggregate:
    n_glyphs: int
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    corner_error: float                 # mean over matched wedges
    point_error: dict[str, float]       # per point type, mean over matched wedges
    exact_rate: float                   # fraction of glyphs solved exactly


def aggregate(results: Iterable[GlyphResult]) -> Aggregate:
    rs = list(results)
    tp = sum(r.tp for r in rs)
    fp = sum(r.fp for r in rs)
    fn = sum(r.fn for r in rs)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    errs = np.array([e for r in rs for e in r.point_errors], dtype=np.float64).reshape(-1, 4)
    corner = float(errs.mean()) if len(errs) else float("nan")
    per_pt = {n: (float(errs[:, k].mean()) if len(errs) else float("nan")) for k, n in enumerate(POINT_NAMES)}
    exact = sum(r.exact for r in rs) / len(rs) if rs else 0.0
    return Aggregate(len(rs), sum(r.n_gt for r in rs), sum(r.n_pred for r in rs), tp, fp, fn,
                     prec, rec, f1, corner, per_pt, exact)


@dataclass
class EvalResult:
    predictor: str
    split: str
    match_threshold: float
    glyphs: list[GlyphResult]
    overall: Aggregate
    per_font: dict[str, Aggregate]

    def worst(self, k: int = 12) -> list[GlyphResult]:
        return sorted(self.glyphs, key=lambda r: r.severity, reverse=True)[:k]

    def to_json(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"predictor": self.predictor, "split": self.split, "match_threshold": self.match_threshold,
                       "overall": asdict(self.overall), "per_font": {k: asdict(v) for k, v in self.per_font.items()},
                       "glyphs": [asdict(g) for g in self.glyphs]}, f, indent=1)


def non_wedge_points(skeleton: Skeleton) -> np.ndarray:
    """Points of strokes that are not canonical wedges (tail-less triangles, malformed strokes)."""
    return np.array([q for st in skeleton.strokes if st.kind != KIND_WEDGE for q in st.points], dtype=float).reshape(-1, 2)


def score_glyph_ignoring(pair: Pair, pred: Sequence[Wedge], skeleton: Skeleton, radius: float,
                         match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> GlyphResult:
    """Secondary metric. Match first, then discard *unmatched* predictions whose apex lies within
    `radius` of a non-wedge stroke: those heads are real ink without a wedge label, so a
    detection there is unscoreable rather than wrong. Matched predictions are never discarded."""
    gt = skeleton.wedges
    pts = non_wedge_points(skeleton)
    if len(pts):
        _, unmatched, _ = match_wedges(pred, gt, match_threshold)
        skip = {i for i in unmatched if np.linalg.norm(pts - np.array(pred[i].apex), axis=1).min() <= radius}
        pred = [w for i, w in enumerate(pred) if i not in skip]
    return score_glyph(pair, pred, gt, match_threshold)


def evaluate(predictor: Predictor, pairs: Sequence[Pair], split: str = "all",
             match_threshold: float = DEFAULT_MATCH_THRESHOLD,
             skeletons: dict[str, Skeleton] | None = None,
             ignore_non_wedge: float | None = None) -> EvalResult:
    """`ignore_non_wedge` (px) switches on the secondary metric that skips predictions near
    non-wedge strokes. The frozen Stage 1 metric is the default, None."""
    rows: list[GlyphResult] = []
    for p in pairs:
        sk = skeletons[p.key] if skeletons else load_skeleton(p)
        pred = predictor.predict(load_image(p), p)
        if ignore_non_wedge is not None:
            rows.append(score_glyph_ignoring(p, pred, sk, ignore_non_wedge, match_threshold))
        else:
            rows.append(score_glyph(p, pred, sk.wedges, match_threshold))
    fonts = sorted({p.font for p in pairs})
    return EvalResult(predictor.name, split, match_threshold, rows, aggregate(rows),
                      {f: aggregate(r for r in rows if r.font == f) for f in fonts})


# --------------------------------------------------------------------------- reference predictors

class NullPredictor:
    """Predicts nothing: the floor. Precision undefined (reported 0), recall 0."""
    name = "null"

    def predict(self, image: np.ndarray, pair: Pair) -> list[Wedge]:
        return []


class OraclePredictor:
    """Returns the ground truth: the ceiling. Must score perfectly."""
    name = "oracle"

    def __init__(self, skeletons: dict[str, Skeleton] | None = None):
        self._sk = skeletons

    def predict(self, image: np.ndarray, pair: Pair) -> list[Wedge]:
        s = self._sk[pair.key] if self._sk else load_skeleton(pair)
        return list(s.wedges)


class JitteredOraclePredictor:
    """Ground truth with Gaussian jitter on every point, a fraction of wedges dropped and a
    fraction of spurious wedges added. Exercises the metric; not a real model."""

    def __init__(self, sigma: float = 2.0, drop: float = 0.1, spurious: float = 0.05, seed: int = 0,
                 skeletons: dict[str, Skeleton] | None = None):
        self.sigma, self.drop, self.spurious = sigma, drop, spurious
        self.rng = np.random.default_rng(seed)
        self._sk = skeletons
        self.name = f"jittered_oracle(sigma={sigma},drop={drop},spurious={spurious})"

    def predict(self, image: np.ndarray, pair: Pair) -> list[Wedge]:
        s = self._sk[pair.key] if self._sk else load_skeleton(pair)
        out = []
        for w in s.wedges:
            if self.rng.random() < self.drop:
                continue
            arr = w.as_array() + self.rng.normal(0, self.sigma, (4, 2))
            out.append(Wedge(*[tuple(map(float, p)) for p in arr]))
            if self.rng.random() < self.spurious:
                arr2 = arr + self.rng.uniform(25, 60, 2)
                out.append(Wedge(*[tuple(map(float, p)) for p in arr2]))
        return out


# --------------------------------------------------------------------------- reporting

def _f(x: float, nd: int = 3) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def markdown_table(results: Sequence[EvalResult]) -> str:
    """One row per (predictor, split, font) with the headline numbers."""
    lines = ["| Predictor | Split | Font | Glyphs | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Corner err (px) | Apex | Head a | Head b | Tail | Exact |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        rows = [("all", r.overall)] + sorted(r.per_font.items())
        for font, a in rows:
            pe = a.point_error
            lines.append(f"| {r.predictor} | {r.split} | {font} | {a.n_glyphs} | {a.n_gt} | {a.n_pred} | {a.tp} | {a.fp} | {a.fn} "
                         f"| {_f(a.precision)} | {_f(a.recall)} | {_f(a.f1)} | {_f(a.corner_error, 2)} "
                         f"| {_f(pe['apex'], 2)} | {_f(pe['head_a'], 2)} | {_f(pe['head_b'], 2)} | {_f(pe['tail'], 2)} | {_f(a.exact_rate)} |")
    return "\n".join(lines)
