"""Stage 2: classical geometric baseline.

Glyphs are drawn as thin outlines. A wedge head is a small hollow triangle whose three
corners continue into two short prongs (the head tips) and one long line (the tail).

Pipeline per image (analyze):
  1. ink = gray < ink_threshold
  2. skeletonize the ink and build a graph: junction clusters and endpoints are nodes,
     the skeleton paths between them are branches
  3. holes = enclosed background regions; small holes are head-interior candidates

Assembly per head hole (assemble, cheap, parameterised so it can be tuned on val):
  4. head junctions = junction nodes within head_radius of the hole centroid
  5. arms = branches leaving the head junctions that are not triangle sides
  6. choose (prong, prong, tail) among the arms: prongs are short leaves, the tail
     points away from the prongs
  7. head tips = prong endpoints extended by tip_extension along the prong direction,
     apex = tail junction shifted by apex_shift along the tail, tail end = follow the tail
     branch through near-straight junctions until an endpoint or a sharp turn
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import permutations
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .data import Pair, Pt, Wedge, _order_heads


# --------------------------------------------------------------------------- parameters

@dataclass(frozen=True)
class ClassicalParams:
    """Defaults are the values tuned on the validation split (see protosnap.tune)."""
    ink_threshold: int = 128
    max_hole_area: int = 24         # px; head interiors have median area 10, p95 13
    head_radius: float = 4.0        # corner junctions lie within head_radius + head_radius_scale * sqrt(area)
    head_radius_scale: float = 0.8
    side_max_len: float = 20.0      # branches shorter than this between head junctions are triangle sides
    opening_weight: float = 0.5     # cost per degree of opening-angle deviation from the prior
    no_tail_penalty: float = 15.0   # cost when the apex candidate has no arm leaving it
    tail_max_dev: float = 45.0      # degrees; an arm this far from the axis cannot be the tail
    missing_prong_penalty: float = 8.0
    prong_dev_weight: float = 0.0   # cost per degree between a prong and its triangle side (prongs extend the sides)
    prong_dir_from_centroid: bool = True  # prong reference direction: hole centroid -> corner (True) or apex -> corner
    # Orientation priors (image angles: 0 = right, 90 = down, -90 = up). In ground truth 95% of
    # tails point within [-40, 130] and only 2% of prongs point inside [-20, 120].
    tail_dir_weight: float = 4.0    # cost per degree the tail direction lies outside tail_arc
    tail_arc: tuple[float, float] = (-40.0, 130.0)
    prong_dir_weight: float = 4.0   # cost per degree a prong direction lies inside prong_forbidden_arc
    prong_forbidden_arc: tuple[float, float] = (-20.0, 120.0)
    capped_prong_penalty: float = 50.0  # a prong longer than prong_max_len is almost certainly a tail
    prong_max_dev: float = 50.0     # degrees; an arm this far from the outward corner direction is not a prong
    prong_max_len: float = 16.0     # px; prongs are followed at most this far (p95 in ground truth is 19.7)
    prong_default_len: float = 6.0  # px beyond the corner when a prong is missing
    tip_extension: float = 0.0      # px added beyond a skeleton endpoint (thinning shortens tips)
    apex_shift: float = 1.0         # px along the tail direction from the apex junction
    continue_angle: float = 35.0    # degrees; follow a line through a junction if the turn is smaller
    max_tail_hops: int = 8
    max_prong_hops: int = 1         # prongs normally end where they meet another stroke
    stop_at_heads: bool = True      # a line stops when it reaches another wedge's head junction
    pass_through_heads: bool = True  # ...unless it crosses that head and continues on the far side (annotators mostly stop; off by default)
    merge_angle: float = 60.0       # degrees; a tail arriving this aligned with the other wedge's tail merges into it
    nms_apex_dist: float = 3.0     # px; two wedges this close at the apex with similar tails are one head seen twice
    nms_angle: float = 90.0        # degrees; tail directions closer than this count as similar


# --------------------------------------------------------------------------- skeleton graph

@dataclass
class Branch:
    path: np.ndarray                      # (N, 2) float, (x, y), ordered from node_a to node_b
    node_a: tuple[str, int] | None        # ("J", label) or ("E", index) or None
    node_b: tuple[str, int] | None

    @property
    def length(self) -> float:
        if len(self.path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.path, axis=0), axis=1)))

    def other(self, node):
        return self.node_b if node == self.node_a else self.node_a

    def oriented(self, from_node) -> np.ndarray:
        """Path ordered so that it starts at from_node."""
        return self.path if from_node == self.node_a else self.path[::-1]


@dataclass
class SkelGraph:
    junctions: dict[int, Pt]              # label -> centroid (x, y)
    endpoints: dict[int, Pt]              # index -> (x, y)
    branches: list[Branch]
    incident: dict[tuple[str, int], list[int]] = field(default_factory=lambda: defaultdict(list))

    def node_xy(self, node) -> Pt:
        return self.junctions[node[1]] if node[0] == "J" else self.endpoints[node[1]]


_N8 = np.ones((3, 3), dtype=int)


def skeleton_graph(ink: np.ndarray) -> SkelGraph:
    skel = skeletonize(ink)
    nb = ndi.convolve(skel.astype(int), _N8, mode="constant") - 1
    nb[~skel] = 0
    junc_px = skel & (nb >= 3)
    end_px = skel & (nb == 1)
    jlab, nj = ndi.label(junc_px, structure=_N8)
    junctions = {}
    if nj:
        for k, (cy, cx) in enumerate(ndi.center_of_mass(junc_px, jlab, range(1, nj + 1)), start=1):
            junctions[k] = (float(cx), float(cy))
    ey, ex = np.where(end_px)
    endpoints = {i: (float(x), float(y)) for i, (x, y) in enumerate(zip(ex, ey))}
    end_index = {(int(y), int(x)): i for i, (x, y) in endpoints.items()}

    blab, nbr = ndi.label(skel & ~junc_px, structure=_N8)
    # junction label adjacent to each pixel (max over 8-neighbourhood)
    jadj = ndi.maximum_filter(jlab, size=3, mode="constant")
    g = SkelGraph(junctions, endpoints, [])
    objects = ndi.find_objects(blab)
    for b in range(1, nbr + 1):
        sl = objects[b - 1]
        sub = blab[sl] == b
        ys, xs = np.where(sub)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        pix = list(zip(ys.tolist(), xs.tolist()))
        path = _order_path(pix)
        ends = [path[0], path[-1]] if len(path) > 1 else [path[0], path[0]]
        nodes = []
        for (y, x) in ends:
            if (y, x) in end_index:
                nodes.append(("E", end_index[(y, x)]))
            elif jadj[y, x] > 0:
                nodes.append(("J", int(jadj[y, x])))
            else:
                nodes.append(None)
        br = Branch(np.array([(x, y) for (y, x) in path], dtype=float), nodes[0], nodes[1])
        idx = len(g.branches)
        g.branches.append(br)
        for n in nodes:
            if n is not None:
                g.incident[n].append(idx)
    return g


def _order_path(pix: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Order the pixels of a thin path component from one end to the other."""
    if len(pix) <= 2:
        return pix
    S = set(pix)

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (y + dy, x + dx) in S]

    ends = [p for p in pix if len(nbrs(p)) == 1]
    start = ends[0] if ends else pix[0]
    order, seen, cur = [start], {start}, start
    while True:
        nxt = [q for q in nbrs(cur) if q not in seen]
        if not nxt:
            break
        # prefer 4-connected steps to avoid skipping pixels on staircases
        nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        cur = nxt[0]
        seen.add(cur)
        order.append(cur)
    if len(order) < len(pix):   # component was not a simple path; append the rest
        order += [p for p in pix if p not in seen]
    return order


# --------------------------------------------------------------------------- analysis

@dataclass
class Hole:
    centroid: Pt
    area: int


@dataclass
class Analysis:
    ink: np.ndarray
    graph: SkelGraph
    holes: list[Hole]


def analyze(image: np.ndarray, ink_threshold: int = 128) -> Analysis:
    ink = image < ink_threshold
    graph = skeleton_graph(ink)
    holes_mask = ndi.binary_fill_holes(ink) & ~ink
    lab, n = ndi.label(holes_mask)
    holes = []
    if n:
        areas = ndi.sum(holes_mask, lab, range(1, n + 1))
        cms = ndi.center_of_mass(holes_mask, lab, range(1, n + 1))
        for a, (cy, cx) in zip(areas, cms):
            holes.append(Hole((float(cx), float(cy)), int(a)))
    return Analysis(ink, graph, holes)


# --------------------------------------------------------------------------- assembly

OPENING_ANGLE_PRIOR = 94.0   # degrees; median opening angle at the apex in ground truth (p5 68, p95 117)


@dataclass
class Arm:
    branch: int
    junction: int                 # junction label the arm leaves from
    start: np.ndarray             # junction xy
    direction: np.ndarray         # unit vector leaving the head
    length: float
    is_leaf: bool
    end_node: tuple[str, int] | None


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(_unit(u), _unit(v)), -1, 1))
    return math.degrees(math.acos(c))


def _arm_direction(path: np.ndarray, start_xy: np.ndarray, n: int = 6) -> np.ndarray:
    tgt = path[min(n, len(path) - 1)] if len(path) > 1 else path[0]
    d = _unit(tgt - start_xy)
    if np.linalg.norm(d) < 1e-9:
        d = _unit(path[-1] - start_xy)
    return d


def _arms_of(g: SkelGraph, j: int, head: set[int], P: ClassicalParams) -> list[Arm]:
    """Branches leaving junction j that are not sides of this head's triangle."""
    out = []
    for bi in g.incident[("J", j)]:
        br = g.branches[bi]
        other = br.other(("J", j))
        if other == ("J", j):
            continue
        if other is not None and other[0] == "J" and other[1] in head and br.length <= P.side_max_len:
            continue
        start = np.array(g.junctions[j])
        path = br.oriented(("J", j))
        out.append(Arm(bi, j, start, _arm_direction(path, start), br.length,
                       other is not None and other[0] == "E", other))
    return out


def _point_along(path: np.ndarray, dist: float) -> np.ndarray:
    if len(path) < 2:
        return path[0]
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if dist >= cum[-1]:
        return path[-1]
    k = int(np.searchsorted(cum, dist, side="right")) - 1
    f = (dist - cum[k]) / max(seg[k], 1e-9)
    return path[k] + f * (path[k + 1] - path[k])


@dataclass
class Trace:
    end: np.ndarray
    at_endpoint: bool     # the line ended in air
    walked: float         # arc length followed
    capped: bool          # stopped because max_len was reached
    stop_junction: int | None = None   # junction label where a stop_junctions stop happened
    reason: str = ""      # endpoint | head | hops | turn | dead | capped
    branches: frozenset = frozenset()   # skeleton branches walked


def _follow(g: SkelGraph, arm: Arm, P: ClassicalParams, stop_junctions: set[int],
            max_len: float | None, max_hops: int | None = None) -> Trace:
    """Follow an arm through near-straight junctions. Stops at an endpoint, at a junction in
    stop_junctions, at a sharp turn, after max_tail_hops, or at max_len arc length."""
    bi, node = arm.branch, ("J", arm.junction)
    if max_hops is None:
        max_hops = P.max_tail_hops
    walked, hops, last_dir, visited = 0.0, 0, arm.direction, {bi}

    def done(end, at_endpoint, capped=False, stop=None, reason=""):
        return Trace(end, at_endpoint, walked, capped, stop, reason, frozenset(visited))

    while True:
        br = g.branches[bi]
        path = br.oriented(node)
        if max_len is not None and walked + br.length > max_len:
            return done(_point_along(path, max_len - walked), False, True, reason="capped")
        walked += br.length
        if len(path) >= 2:
            last_dir = _unit(path[-1] - path[max(0, len(path) - 6)])
        end_node = br.other(node)
        if end_node is None:
            return done(path[-1], False, reason="dead")
        if end_node[0] == "E":
            return done(np.array(g.endpoints[end_node[1]]), True, reason="endpoint")
        jxy = np.array(g.junctions[end_node[1]])
        hops += 1
        if P.stop_at_heads and end_node[1] in stop_junctions:
            return done(jxy, False, stop=end_node[1], reason="head")
        if hops > max_hops:
            return done(jxy, False, reason="hops")
        best = None
        for nb in g.incident[end_node]:
            if nb in visited:
                continue
            nbr = g.branches[nb]
            if nbr.other(end_node) is None or nbr.other(end_node) == end_node:
                continue
            ang = _angle_deg(last_dir, _arm_direction(nbr.oriented(end_node), jxy))
            if ang <= P.continue_angle and (best is None or ang < best[0]):
                best = (ang, nb)
        if best is None:
            return done(jxy, False, reason="turn")
        visited.add(best[1])
        bi, node = best[1], end_node


def _tip(g: SkelGraph, arm: Arm, P: ClassicalParams, stop: set[int], max_len: float | None) -> Trace:
    tr = _follow(g, arm, P, stop, max_len, P.max_prong_hops)
    if tr.at_endpoint:
        path = g.branches[arm.branch].oriented(("J", arm.junction))
        d = _unit(tr.end - path[max(0, len(path) - 5)]) if len(path) >= 2 else arm.direction
        tr.end = tr.end + P.tip_extension * d
    return tr


def _best_arm(arms: list[Arm], direction: np.ndarray, max_dev: float) -> tuple[Arm | None, float]:
    best, best_ang = None, max_dev
    for a in arms:
        ang = _angle_deg(a.direction, direction)
        if ang < best_ang:
            best, best_ang = a, ang
    return best, best_ang


def _prong(g: SkelGraph, arms: list[Arm], outward: np.ndarray, corner: np.ndarray,
           P: ClassicalParams, others: set[int]) -> tuple[np.ndarray, float, frozenset]:
    """Tip of the prong leaving a base corner, and its cost. A prong continues its triangle
    side, so its deviation from the apex->corner direction is part of the cost."""
    arm, dev = _best_arm(arms, outward, P.prong_max_dev)
    if arm is None:
        return corner + P.prong_default_len * outward, P.missing_prong_penalty, frozenset()
    tr = _tip(g, arm, P, others, P.prong_max_len)
    return tr.end, (P.capped_prong_penalty if tr.capped else 0.0) + P.prong_dev_weight * dev, tr.branches


def _score(apex: np.ndarray, tip_b: np.ndarray, tip_c: np.ndarray, tail_dev: float, P: ClassicalParams) -> float:
    opening = _angle_deg(tip_b - apex, tip_c - apex)
    return P.opening_weight * abs(opening - OPENING_ANGLE_PRIOR) + tail_dev


def _image_angle(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _outside_arc(angle: float, lo: float, hi: float) -> float:
    """Degrees by which `angle` lies outside the arc [lo, hi] (0 when inside)."""
    a = (angle - lo) % 360.0
    span = (hi - lo) % 360.0
    if a <= span:
        return 0.0
    return min(a - span, 360.0 - a)


def _inside_arc(angle: float, lo: float, hi: float) -> float:
    """Depth (degrees to the nearest edge) by which `angle` lies inside [lo, hi] (0 when outside)."""
    a = (angle - lo) % 360.0
    span = (hi - lo) % 360.0
    if a > span:
        return 0.0
    return min(a, span - a)


def _direction_cost(apex: np.ndarray, tail_end: np.ndarray, tips: list[np.ndarray], P: ClassicalParams) -> float:
    cost = 0.0
    if np.linalg.norm(tail_end - apex) >= 2.0:
        cost += P.tail_dir_weight * _outside_arc(_image_angle(tail_end - apex), *P.tail_arc)
    for tip in tips:
        if np.linalg.norm(tip - apex) >= 2.0:
            cost += P.prong_dir_weight * _inside_arc(_image_angle(tip - apex), *P.prong_forbidden_arc)
    return cost


def _assemble_triangle(g: SkelGraph, corners: list[int], P: ClassicalParams, others: set[int],
                       centroid: np.ndarray | None = None) -> tuple[float, Wedge] | None:
    """Head with three corner junctions. Each corner is a candidate apex: its tail is the arm
    best aligned with the axis away from the opposite side, the prongs leave the other two
    corners. Score by tail alignment, opening angle between the prong tips, and prong caps."""
    head = set(corners)
    xy = {j: np.array(g.junctions[j]) for j in corners}
    arms = {j: _arms_of(g, j, head, P) for j in corners}
    best = None
    for i, a in enumerate(corners):
        b, c = corners[(i + 1) % 3], corners[(i + 2) % 3]
        axis = _unit(xy[a] - (xy[b] + xy[c]) / 2)
        tail, dev = _best_arm(arms[a], axis, P.tail_max_dev)
        cost = dev if tail is not None else P.no_tail_penalty
        tips, claimed = [], set()
        for j in (b, c):
            # prongs leave the head radially: the hole centroid -> corner direction predicts the
            # prong direction with 15 deg median deviation, versus 42 deg for apex -> corner
            outward = _unit(xy[j] - centroid) if (centroid is not None and P.prong_dir_from_centroid) else _unit(xy[j] - xy[a])
            tip, pc, brs = _prong(g, arms[j], outward, xy[j], P, others)
            tips.append(tip)
            cost += pc
            claimed |= brs
        cost += _score(xy[a], tips[0], tips[1], 0.0, P)
        if tail is not None:
            ttr = _follow(g, tail, P, others, None)
            tail_end = ttr.end
            claimed |= ttr.branches
        else:
            tail_end = xy[a] + P.prong_default_len * axis
        cost += _direction_cost(xy[a], tail_end, tips, P)
        if best is None or cost < best[0]:
            best = (cost, a, axis, tail, tips, tail_end, claimed)
    cost, a, axis, tail, tips, tail_end, claimed = best
    apex = xy[a] + P.apex_shift * (tail.direction if tail is not None else axis)
    ha, hb = _order_heads(tuple(map(float, apex)), tuple(map(float, tips[0])), tuple(map(float, tips[1])))
    return cost, Wedge(tuple(map(float, apex)), ha, hb, tuple(map(float, tail_end))), tail, frozenset(claimed)


def _assemble_from_arms(g: SkelGraph, near: list[int], P: ClassicalParams, others: set[int]
                        ) -> tuple[float, Wedge] | None:
    """Head with two junctions (merged corners): choose roles among the arms directly."""
    head = set(near)
    arms = [a for j in near for a in _arms_of(g, j, head, P)]
    if len(arms) < 2:
        return None
    traces = {a.branch: _tip(g, a, P, others, P.prong_max_len) for a in arms}
    best = None
    if len(arms) == 2:
        for pr, tl in ((arms[0], arms[1]), (arms[1], arms[0])):
            cost = (P.capped_prong_penalty if traces[pr.branch].capped else 0.0) + P.missing_prong_penalty
            cost += 0.3 * max(0.0, pr.length - tl.length)
            cost += _direction_cost(tl.start, tl.start + 10 * tl.direction, [traces[pr.branch].end], P)
            if best is None or cost < best[0]:
                best = (cost, pr, None, tl)
    else:
        for pa, pb, tl in permutations(arms, 3):
            if pa.branch >= pb.branch:
                continue
            apex = tl.start
            cost = _score(apex, traces[pa.branch].end, traces[pb.branch].end, 0.0, P)
            cost += _angle_deg(tl.direction, -_unit(pa.direction + pb.direction))
            cost += sum(P.capped_prong_penalty for x in (pa, pb) if traces[x.branch].capped)
            cost += _direction_cost(apex, apex + 10 * tl.direction, [traces[pa.branch].end, traces[pb.branch].end], P)
            if best is None or cost < best[0]:
                best = (cost, pa, pb, tl)
    cost, pa, pb, tl = best
    apex = tl.start + P.apex_shift * tl.direction
    ttr = _follow(g, tl, P, others, None)
    tail_end = ttr.end
    claimed = set(ttr.branches) | set(traces[pa.branch].branches) | (set(traces[pb.branch].branches) if pb is not None else set())
    tip_a = traces[pa.branch].end
    if pb is not None:
        tip_b = traces[pb.branch].end
    else:
        d = tl.direction
        v = tip_a - apex
        tip_b = apex + 2 * (v @ d) * d - v
    ha, hb = _order_heads(tuple(map(float, apex)), tuple(map(float, tip_a)), tuple(map(float, tip_b)))
    return cost + 10.0, Wedge(tuple(map(float, apex)), ha, hb, tuple(map(float, tail_end))), tl, frozenset(claimed)


def assemble(an: Analysis, P: ClassicalParams) -> list[Wedge]:
    g = an.graph
    if not g.junctions:
        return []
    jl = np.array(list(g.junctions.keys()))
    jxy = np.array([g.junctions[k] for k in jl])
    heads, centroids = [], []
    for h in an.holes:
        if h.area > P.max_hole_area:
            continue
        c = np.array(h.centroid)
        radius = P.head_radius + P.head_radius_scale * math.sqrt(h.area)
        d = np.linalg.norm(jxy - c, axis=1)
        order = np.argsort(d)
        near = [int(jl[k]) for k in order if d[k] <= radius][:3]
        if len(near) >= 2:
            heads.append(near)
            centroids.append(c)
    all_head_junctions = {j for near in heads for j in near}
    head_of = {j: i for i, near in enumerate(heads) for j in near}
    built = []   # (wedge, tail_arm, own_junctions)
    for near, c in zip(heads, centroids):
        others = all_head_junctions - set(near)
        res = _assemble_triangle(g, near, P, others, c) if len(near) == 3 else _assemble_from_arms(g, near, P, others)
        if res is not None:
            built.append((res[1], res[2], set(near), res[3], res[0]))
    if P.pass_through_heads and P.stop_at_heads:
        built = _pass_through(g, built, heads, head_of, all_head_junctions, P)
    return _suppress(built, P)


def _suppress(built, P: ClassicalParams) -> list[Wedge]:
    """Cost-ranked non-maximum suppression: a wedge is dropped when a cheaper wedge with a tail
    pointing the same way shares two of its corner junctions or has its apex within
    nms_apex_dist (a head seen through two adjacent holes). Stacked heads with different
    tails survive."""
    kept: list[tuple[Wedge, set[int], np.ndarray]] = []
    for w, arm, own, claimed, cost in sorted(built, key=lambda b: b[4]):
        d = np.array(w.tail) - np.array(w.apex)
        dup = False
        for kw, kown, kd in kept:
            similar = np.linalg.norm(d) < 1 or np.linalg.norm(kd) < 1 or _angle_deg(d, kd) <= P.nms_angle
            if similar and (len(own & kown) >= 2 or math.dist(w.apex, kw.apex) <= P.nms_apex_dist):
                dup = True
                break
        if not dup:
            kept.append((w, own, d))
    return [k[0] for k in kept]


def _pass_through(g: SkelGraph, built, heads, head_of, all_head_junctions: set[int], P: ClassicalParams):
    """Second pass over tails that stopped at another wedge's head. The tail is allowed to
    cross that head only if the line beyond it is not claimed by any other wedge (as its tail
    or a prong) and does not merge into that wedge's tail: annotators stop a tail where it
    runs into the next wedge, but continue it when it merely crosses one."""
    tail_dir_of_head = {}
    for w, arm, own, claimed, cost in built:
        d = np.array(w.tail) - np.array(w.apex)
        for j in own:
            tail_dir_of_head[j] = d
    out = []
    for i, (w, arm, own, claimed, cost) in enumerate(built):
        if arm is None:
            out.append((w, arm, own, claimed, cost))
            continue
        others_claimed = set().union(*(b[3] for k, b in enumerate(built) if k != i)) if len(built) > 1 else set()
        allowed: set[int] = set()
        tr = _follow(g, arm, P, all_head_junctions - own, None)
        final = tr
        for _ in range(3):
            if tr.stop_junction is None:
                break
            hid = head_of.get(tr.stop_junction)
            if hid is None:
                break
            other_dir = tail_dir_of_head.get(tr.stop_junction)
            arrive = tr.end - np.array(w.apex)
            if other_dir is not None and np.linalg.norm(other_dir) > 1 and _angle_deg(arrive, other_dir) <= P.merge_angle:
                break
            allowed |= set(heads[hid])
            nxt = _follow(g, arm, P, all_head_junctions - own - allowed, None)
            if (set(nxt.branches) - set(tr.branches)) & others_claimed:
                break   # the line beyond the head belongs to another wedge
            final, tr = nxt, nxt
        if np.linalg.norm(final.end - np.array(w.tail)) > 0.5:
            w = Wedge(w.apex, w.head_a, w.head_b, (float(final.end[0]), float(final.end[1])))
            claimed = frozenset(set(claimed) | set(final.branches))
        out.append((w, arm, own, claimed, cost))
    return out


def _dedupe(wedges: list[Wedge], min_apex_sep: float = 3.0) -> list[Wedge]:
    out: list[Wedge] = []
    for w in wedges:
        if all(math.dist(w.apex, o.apex) > min_apex_sep for o in out):
            out.append(w)
    return out


# --------------------------------------------------------------------------- predictor

class ClassicalPredictor:
    """Stage 2 baseline. Caches the expensive analysis per glyph so parameters can be tuned."""

    def __init__(self, params: ClassicalParams = ClassicalParams()):
        self.params = params
        self._cache: dict[tuple[str, int], Analysis] = {}

    @property
    def name(self) -> str:
        return "classical"

    def analysis(self, image: np.ndarray, pair: Pair | None) -> Analysis:
        key = (pair.key if pair else None, self.params.ink_threshold)
        if key[0] is None:
            return analyze(image, self.params.ink_threshold)
        if key not in self._cache:
            self._cache[key] = analyze(image, self.params.ink_threshold)
        return self._cache[key]

    def predict(self, image: np.ndarray, pair: Pair) -> list[Wedge]:
        return assemble(self.analysis(image, pair), self.params)

    def with_params(self, **kw) -> "ClassicalPredictor":
        p = ClassicalPredictor(replace(self.params, **kw))
        p._cache = self._cache
        return p
