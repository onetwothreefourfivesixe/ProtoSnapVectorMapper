from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from protosnap import find_pairs, load_image, load_skeleton
from protosnap.classical import (ClassicalParams, ClassicalPredictor, _inside_arc, _outside_arc, analyze,
                                 skeleton_graph)
from protosnap.metrics import evaluate, match_wedges

ROOT = Path(__file__).resolve().parents[1]


def synthetic_wedge(apex=(120, 120), tips=((100, 100), (140, 100)), tail=(120, 200)):
    """Draw an outline wedge like the fonts do: hollow head triangle with prongs and a tail."""
    im = Image.new("L", (256, 256), 255)
    d = ImageDraw.Draw(im)
    a, (t1, t2), t = apex, tips, tail
    # inner triangle corners sit ~6 px from the apex along the tip directions
    def towards(p, q, dist):
        v = np.array(q, float) - np.array(p, float)
        return tuple(np.array(p, float) + v / np.linalg.norm(v) * dist)
    c1, c2 = towards(a, t1, 8), towards(a, t2, 8)
    for seg in ((a, c1), (a, c2), (c1, c2), (c1, t1), (c2, t2), (a, t)):
        d.line(seg, fill=0, width=3)
    return np.asarray(im)


def test_skeleton_graph_on_synthetic_wedge():
    img = synthetic_wedge()
    an = analyze(img)
    assert len(an.holes) == 1 and an.holes[0].area < 40
    g = an.graph
    assert len(g.junctions) >= 2 and len(g.endpoints) == 3
    assert all(b.length >= 0 for b in g.branches)


def test_assemble_synthetic_wedge_roles():
    img = synthetic_wedge()
    pred = ClassicalPredictor().predict(img, None)
    assert len(pred) == 1
    w = pred[0]
    assert abs(w.apex[0] - 120) < 6 and abs(w.apex[1] - 120) < 6
    assert w.tail[1] > 180                          # tail found and followed downwards
    assert {round(w.head_a[1] / 10), round(w.head_b[1] / 10)} == {10}   # both tips near y=100


def test_arc_helpers():
    assert _outside_arc(0, -40, 130) == 0 and _outside_arc(-90, -40, 130) == 50 and _outside_arc(180, -40, 130) == 50
    assert _inside_arc(-90, -20, 120) == 0 and _inside_arc(50, -20, 120) == 70 and _inside_arc(115, -20, 120) == 5


def test_real_glyph_a():
    """Sign A: the left wedge is recovered closely. The other two are stacked (a tiny wedge
    whose head sits directly above the next wedge's head); the baseline merges them, which is
    a known failure class, so at most one may be missed and nothing spurious may appear."""
    pairs = {p.key: p for p in find_pairs(ROOT)}
    p = pairs["Santakku/0x12000"]
    pred = ClassicalPredictor().predict(load_image(p), p)
    gt = load_skeleton(p).wedges
    m, up, ug = match_wedges(pred, gt)
    assert len(m) >= 2 and not up and len(ug) <= 1
    left = [x for x in m if x.gt_index == 0]
    assert len(left) == 1 and left[0].corner_error < 8


def test_predictor_cache_and_params():
    pairs = find_pairs(ROOT)[:3]
    pr = ClassicalPredictor()
    for p in pairs:
        pr.predict(load_image(p), p)
    assert len(pr._cache) == 3
    pr2 = pr.with_params(apex_shift=2.0)
    assert pr2.params.apex_shift == 2.0 and pr2._cache is pr._cache
    assert pr2.predict(load_image(pairs[0]), pairs[0]) != pr.predict(load_image(pairs[0]), pairs[0])


def test_baseline_beats_floor_on_val():
    from protosnap.split import load_splits, pairs_in_split
    pairs = find_pairs(ROOT)
    val = pairs_in_split(pairs, load_splits(ROOT / "data" / "splits.json"), "val")
    sk = {p.key: load_skeleton(p) for p in val}
    a = evaluate(ClassicalPredictor(), val, skeletons=sk).overall
    assert a.f1 > 0.9 and a.corner_error < 15


# ----------------------------------------------------------------------------- geometry fixes

def test_suppress_drops_duplicates_but_keeps_stacked_heads():
    from protosnap.classical import _suppress, ClassicalParams
    from protosnap import Wedge
    P = ClassicalParams(nms_apex_dist=6.0, nms_angle=45.0)
    down = lambda ax, ay: Wedge((ax, ay), (ax - 10, ay - 8), (ax + 10, ay - 8), (ax, ay + 40))
    # same head seen through two holes: apexes 4 px apart, both tails down, shared corners
    a = (down(100, 100), None, {1, 2, 3}, frozenset(), 5.0)
    b = (down(104, 100), None, {2, 3, 9}, frozenset(), 12.0)
    # a stacked head 11 px below sharing two corners but tail pointing right
    c = (Wedge((100, 111), (90, 103), (110, 119), (140, 111)), None, {2, 3, 7}, frozenset(), 8.0)
    out = _suppress([b, a, c], P)
    assert len(out) == 2 and out[0] == a[0] and c[0] in out


def test_follow_reports_reason_and_branches():
    from protosnap.classical import ClassicalPredictor, _arms_of, _follow
    img = synthetic_wedge()
    pr = ClassicalPredictor()
    an = pr.analysis(img, None)
    g = an.graph
    # any arm followed to the end of the synthetic tail ends in air
    for j in g.junctions:
        for arm in _arms_of(g, j, set(g.junctions), pr.params):
            tr = _follow(g, arm, pr.params, set(), None)
            assert tr.reason in {"endpoint", "turn", "head", "hops", "dead"}
            assert arm.branch in tr.branches
            if tr.reason == "endpoint":
                assert tr.at_endpoint


def test_prong_direction_param_changes_tips():
    pairs = {p.key: p for p in find_pairs(ROOT)}
    p = pairs["Santakku/0x12108"]
    img = load_image(p)
    a = ClassicalPredictor().with_params(prong_dir_from_centroid=True).predict(img, p)
    b = ClassicalPredictor().with_params(prong_dir_from_centroid=False).predict(img, p)
    assert a != b


def test_pass_through_is_optional_and_stable():
    pairs = {p.key: p for p in find_pairs(ROOT)}
    p = pairs["Santakku/0x12000"]
    img = load_image(p)
    on = ClassicalPredictor().with_params(pass_through_heads=True).predict(img, p)
    off = ClassicalPredictor().with_params(pass_through_heads=False).predict(img, p)
    assert len(on) == len(off)
    assert all(w1.apex == w2.apex for w1, w2 in zip(on, off))
