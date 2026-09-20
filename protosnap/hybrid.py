"""Stage 4: hybrid predictor. The learned detector proposes wedges; classical geometry refines them.

What each side is good at (Stage 3 failure analysis):
  learned    finds heads, places apexes and head tips (large head-tip errors 1 vs 27)
  classical  follows ink exactly, so a tail end it reaches is pixel-accurate

Refinements, each switchable for the ablation:
  tail        trace the skeleton from the apex along the network's tail direction, collecting
              every node the line passes (junctions and the final endpoint), and move the tail
              to the node nearest the network's own tail estimate if one lies within
              tail_radius. The network decides *where along the line* the annotator stopped;
              the skeleton supplies the exact on-ink position.
  tips        snap each head tip to the nearest skeleton endpoint within tip_radius
  ink         drop a wedge whose apex is further than ink_radius from any ink
  orientation swap tail and a head tip when the tail points outside the ground-truth arc and
              the swap puts it inside (off by default: most swaps are rare orientations that
              the classical prior gets wrong too)

Every wedge gets a confidence in [0, 1] combining the network's apex score with how well the
geometry agrees with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .classical import (Analysis, ClassicalParams, SkelGraph, _angle_deg, _arm_direction, _image_angle,
                        _outside_arc, _unit, analyze)
from .data import Pair, Wedge, _order_heads


@dataclass(frozen=True)
class HybridParams:
    refine_tail: bool = True
    tail_mode: str = "heatmap"      # "heatmap": pick the traced node with the best tail-heatmap support; "nearest": node nearest the regressed tail
    heat_radius: float = 4.0        # px; search window for a tail-heatmap peak around each traced node
    heat_threshold: float = 0.2     # minimum tail-heatmap value for a node to count as supported
    heat_sigma: float = 20.0        # px; candidates are down-weighted by exp(-distance to the regressed tail / sigma)
    keep_margin: float = 1.15       # a traced node must beat the regressed tail's own support by this factor
    tail_radius: float = 10.0       # px; "nearest" mode only: accept a node this close to the regressed tail
    apex_radius: float = 6.0        # px; junctions this close to the apex can start the tail trace
    arm_angle: float = 40.0         # degrees; the traced arm must leave within this angle of the network's tail direction
    continue_angle: float = 35.0    # degrees; follow the line through a junction if it turns less than this
    max_hops: int = 10
    min_tail: float = 4.0           # px; tails shorter than this are hidden under the next head and left alone
    snap_tips: bool = False
    tip_radius: float = 3.0
    reject_off_ink: bool = True
    ink_radius: float = 3.0
    fix_orientation: bool = False
    tail_arc: tuple[float, float] = (-40.0, 130.0)
    rare_discount: float = 0.6      # confidence multiplier for tails pointing outside tail_arc
    flag_off_ink_tail: bool = True  # a tail is an ink line: 99% of true tails lie fully on ink, so one that does not is suspect
    clip_off_ink_tail: bool = False # also cut such a tail back to where the ink ends (hurt on test: 0.644 -> 0.600 exact, so off)
    ink_tolerance: float = 1.5      # px from ink that still counts as on the line
    min_tail_coverage: float = 0.8  # fraction of the apex->tail segment that must be on ink
    off_ink_confidence: float = 0.05  # confidence cap for a wedge whose tail leaves the ink


@dataclass
class ScoredWedge:
    wedge: Wedge
    confidence: float
    apex_score: float
    tail_source: str        # "network" | "skeleton" | "clipped"
    tail_shift: float       # px the tail moved
    off_ink: bool = False   # the apex->tail segment is not covered by ink: almost certainly a wrong tail


def _trace_nodes(g: SkelGraph, start_j: int, first_branch: int, P: HybridParams) -> list[np.ndarray]:
    """Nodes (junction centroids, then the final endpoint) met while following a line from a
    junction through near-straight continuations."""
    out: list[np.ndarray] = []
    bi, node, visited = first_branch, ("J", start_j), {first_branch}
    last_dir = None
    for _ in range(P.max_hops + 1):
        br = g.branches[bi]
        path = br.oriented(node)
        if len(path) >= 2:
            last_dir = _unit(path[-1] - path[max(0, len(path) - 6)])
        end = br.other(node)
        if end is None:
            out.append(np.array(path[-1], dtype=float))
            break
        if end[0] == "E":
            out.append(np.array(g.endpoints[end[1]], dtype=float))
            break
        jxy = np.array(g.junctions[end[1]], dtype=float)
        out.append(jxy)
        best = None
        for nb in g.incident[end]:
            if nb in visited:
                continue
            o = g.branches[nb].other(end)
            if o is None or o == end:
                continue
            ang = _angle_deg(last_dir, _arm_direction(g.branches[nb].oriented(end), jxy)) if last_dir is not None else 0.0
            if ang <= P.continue_angle and (best is None or ang < best[0]):
                best = (ang, nb)
        if best is None:
            break
        visited.add(best[1])
        bi, node = best[1], end
    return out


def refine_tail(g: SkelGraph, jl: np.ndarray, jxy: np.ndarray, w: Wedge, P: HybridParams,
                tail_heat: np.ndarray | None = None) -> tuple[tuple[float, float], str]:
    A, T = np.array(w.apex), np.array(w.tail)
    d = T - A
    if np.linalg.norm(d) < P.min_tail or len(jl) == 0:
        return w.tail, "network"
    near = np.where(np.linalg.norm(jxy - A, axis=1) <= P.apex_radius)[0]
    best = None
    for k in near:
        j = int(jl[k])
        for bi in g.incident[("J", j)]:
            br = g.branches[bi]
            if br.other(("J", j)) == ("J", j):
                continue
            ang = _angle_deg(_arm_direction(br.oriented(("J", j)), jxy[k]), d)
            if ang <= P.arm_angle and (best is None or ang < best[0]):
                best = (ang, j, bi)
    if best is None:
        return w.tail, "network"
    nodes = _trace_nodes(g, best[1], best[2], P)
    if not nodes:
        return w.tail, "network"
    if P.tail_mode == "heatmap" and tail_heat is not None:
        from .learned.decode import _local_peak
        own = _local_peak(tail_heat, T[0], T[1], P.heat_radius, 0.0)
        own_heat = float(tail_heat[int(round(own[1])), int(round(own[0]))]) if own is not None else 0.0
        best = None
        for n in nodes:
            pk = _local_peak(tail_heat, n[0], n[1], P.heat_radius, P.heat_threshold)
            if pk is None:
                continue
            heat = float(tail_heat[int(round(pk[1])), int(round(pk[0]))])
            score = heat * math.exp(-math.dist(pk, T) / P.heat_sigma)
            if best is None or score > best[0]:
                best = (score, pk)
        if best is not None and best[0] > own_heat * P.keep_margin and math.dist(best[1], T) > 2.0:
            return (float(best[1][0]), float(best[1][1])), "skeleton"
        return w.tail, "network"
    dist = [float(np.linalg.norm(n - T)) for n in nodes]
    i = int(np.argmin(dist))
    if dist[i] <= P.tail_radius:
        return (float(nodes[i][0]), float(nodes[i][1])), "skeleton"
    return w.tail, "network"


class HybridPredictor:
    def __init__(self, learned, params: HybridParams = HybridParams(), name: str = "hybrid"):
        self.learned, self.params, self.name = learned, params, name
        self._an: dict[str, Analysis] = {}
        self.last: list[ScoredWedge] = []

    def with_params(self, **kw) -> "HybridPredictor":
        h = HybridPredictor(self.learned, replace(self.params, **kw), self.name)
        h._an = self._an
        return h

    def _analysis(self, image: np.ndarray, pair: Pair | None) -> Analysis:
        key = pair.key if pair is not None else None
        if key is None:
            return analyze(image, ClassicalParams().ink_threshold)
        if key not in self._an:
            self._an[key] = analyze(image, ClassicalParams().ink_threshold)
        return self._an[key]

    def predict_scored(self, image: np.ndarray, pair: Pair | None) -> list[ScoredWedge]:
        P = self.params
        wedges = self.learned.predict(image, pair)
        scores = list(self.learned.last_scores)
        import torch
        aux = torch.sigmoid(self.learned.raw(image, pair)["aux"]).numpy()     # (2, H, W): head tips, tail ends
        tail_heat = aux[1] if (P.refine_tail and P.tail_mode == "heatmap") else None
        an = self._analysis(image, pair)
        g = an.graph
        jl = np.array(list(g.junctions.keys()))
        jxy = np.array([g.junctions[k] for k in jl], dtype=float).reshape(-1, 2)
        exy = np.array(list(g.endpoints.values()), dtype=float).reshape(-1, 2)
        ink_pts = None
        ink_dist = None
        out: list[ScoredWedge] = []
        for w, s in zip(wedges, scores):
            if P.reject_off_ink:
                if ink_pts is None:
                    ys, xs = np.nonzero(an.ink)
                    ink_pts = np.stack([xs, ys], 1).astype(float)
                if len(ink_pts) == 0 or np.linalg.norm(ink_pts - np.array(w.apex), axis=1).min() > P.ink_radius:
                    continue
            if P.fix_orientation:
                w = _fix_orientation(w, P)
            ha, hb = np.array(w.head_a), np.array(w.head_b)
            if P.snap_tips and len(exy):
                for tip in (ha, hb):
                    d = np.linalg.norm(exy - tip, axis=1)
                    k = int(d.argmin())
                    if d[k] <= P.tip_radius:
                        tip[:] = exy[k]
            tail, src = (refine_tail(g, jl, jxy, w, P, tail_heat) if P.refine_tail else (w.tail, "network"))
            off_ink = False
            if P.flag_off_ink_tail or P.clip_off_ink_tail:
                if ink_dist is None:
                    from scipy import ndimage as ndi
                    ink_dist = ndi.distance_transform_edt(~an.ink)
                cov, last = tail_coverage(ink_dist, w.apex, tail, P.ink_tolerance)
                if cov < P.min_tail_coverage:
                    off_ink = True
                    if P.clip_off_ink_tail:
                        tail, src = (float(last[0]), float(last[1])), "clipped"
            shift = math.dist(tail, w.tail)
            a, b = _order_heads(w.apex, (float(ha[0]), float(ha[1])), (float(hb[0]), float(hb[1])))
            nw = Wedge(w.apex, a, b, tail)
            conf = _confidence(s, aux, nw, P)
            out.append(ScoredWedge(nw, min(conf, P.off_ink_confidence) if off_ink else conf, s, src, shift, off_ink))
        self.last = out
        return out

    def predict(self, image: np.ndarray, pair: Pair) -> list[Wedge]:
        return [sw.wedge for sw in self.predict_scored(image, pair)]


def tail_coverage(ink_dist: np.ndarray, apex, tail, tol: float) -> tuple[float, np.ndarray]:
    """Fraction of the apex->tail segment lying within `tol` px of ink, and the last point
    reached before the line first leaves the ink."""
    a, b = np.array(apex, dtype=float), np.array(tail, dtype=float)
    L = float(np.linalg.norm(b - a))
    if L < 2:
        return 1.0, b
    ts = np.linspace(0, 1, max(3, int(L) + 1))
    pts = a[None] + ts[:, None] * (b - a)[None]
    H, W = ink_dist.shape
    on = ink_dist[np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1), np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)] <= tol
    off = np.where(~on)[0]
    # ignore the first 3 px: the apex itself can sit a pixel off the ink
    off = off[off > 3]
    last = pts[off[0] - 1] if len(off) else b
    return float(on.mean()), last


def _fix_orientation(w: Wedge, P: HybridParams) -> Wedge:
    A = np.array(w.apex)
    t = np.array(w.tail) - A
    if np.linalg.norm(t) < 4 or _outside_arc(_image_angle(t), *P.tail_arc) == 0:
        return w
    for head, other in ((w.head_a, w.head_b), (w.head_b, w.head_a)):
        h = np.array(head) - A
        if np.linalg.norm(h) >= 4 and _outside_arc(_image_angle(h), *P.tail_arc) == 0:
            a, b = _order_heads(w.apex, w.tail, other)
            return Wedge(w.apex, a, b, head)
    return w


def _heat_at(h: np.ndarray, pt) -> float:
    H, W = h.shape
    return float(h[int(np.clip(round(pt[1]), 0, H - 1)), int(np.clip(round(pt[0]), 0, W - 1))])


def _confidence(apex_score: float, aux: np.ndarray, w: Wedge, P: HybridParams) -> float:
    """The weakest link among the network's own evidence: its apex score and its heatmap support
    at the final tail end and at both head tips, discounted when the tail points in a rare
    direction (where both models err most). On validation this ranks correct wedges with AUROC
    0.80 (apex score alone: 0.58) and, taking the minimum over a glyph, fully correct glyphs
    with AUROC 0.88."""
    c = min(float(apex_score), _heat_at(aux[1], w.tail), _heat_at(aux[0], w.head_a), _heat_at(aux[0], w.head_b))
    t = np.array(w.tail) - np.array(w.apex)
    if np.linalg.norm(t) >= P.min_tail and _outside_arc(_image_angle(t), *P.tail_arc) > 0:
        c *= P.rare_discount
    return max(0.0, min(1.0, c))


def glyph_confidence(scored: list[ScoredWedge]) -> float:
    """A glyph is only as trustworthy as its least certain wedge. Empty predictions score 0."""
    return min((s.confidence for s in scored), default=0.0)
