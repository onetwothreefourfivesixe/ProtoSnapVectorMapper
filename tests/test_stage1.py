import math
from pathlib import Path

import numpy as np
import pytest

from protosnap import Wedge, find_pairs, load_skeleton
from protosnap.metrics import (JitteredOraclePredictor, NullPredictor, OraclePredictor, aggregate, evaluate,
                               match_wedges, score_glyph)
from protosnap.render import comparison_panel
from protosnap.data import load_image

ROOT = Path(__file__).resolve().parents[1]


def W(ax, ay, dx=0.0, dy=0.0):
    """A wedge with apex (ax, ay), heads 10 px either side above, tail 40 px below, shifted by (dx, dy)."""
    return Wedge((ax + dx, ay + dy), (ax - 10 + dx, ay - 8 + dy), (ax + 10 + dx, ay - 8 + dy), (ax + dx, ay + 40 + dy))


@pytest.fixture(scope="module")
def pairs():
    return find_pairs(ROOT)


def test_perfect_match():
    gt = [W(50, 50), W(120, 60), W(90, 150)]
    m, up, ug = match_wedges(gt, gt)
    assert len(m) == 3 and not up and not ug
    assert all(x.corner_error == 0 for x in m)


def test_order_invariant_and_threshold():
    gt = [W(50, 50), W(120, 60)]
    pred = [W(120, 60, 3, 4), W(50, 50, -6, 8)]      # 5 px and 10 px apex shifts, reversed order
    m, up, ug = match_wedges(pred, gt, match_threshold=10)
    assert {(x.pred_index, x.gt_index) for x in m} == {(0, 1), (1, 0)}
    assert sorted(round(x.apex_distance) for x in m) == [5, 10]
    m2, up2, ug2 = match_wedges(pred, gt, match_threshold=6)
    assert len(m2) == 1 and up2 == [1] and ug2 == [0]


def test_fp_fn_and_exact():
    p = find_pairs(ROOT)[0]
    gt = [W(50, 50), W(120, 60)]
    r = score_glyph(p, [W(50, 50), W(120, 60), W(200, 200)], gt)
    assert (r.tp, r.fp, r.fn, r.exact) == (2, 1, 0, False)
    r = score_glyph(p, [W(50, 50)], gt)
    assert (r.tp, r.fp, r.fn, r.exact) == (1, 0, 1, False)
    r = score_glyph(p, [W(50, 50, 1, 1), W(120, 60)], gt)
    assert r.exact and abs(r.corner_error - math.sqrt(2) / 2) < 1e-9   # one wedge shifted by sqrt2 on all 4 pts, other 0
    r = score_glyph(p, [], [])
    assert r.exact and math.isnan(r.corner_error)


def test_jitter_error_is_measured():
    p = find_pairs(ROOT)[0]
    gt = [W(50 + 30 * i, 100) for i in range(5)]
    pred = [Wedge(*[(x + 3.0, y + 4.0) for x, y in w.as_array()]) for w in gt]   # every point off by 5 px
    r = score_glyph(p, pred, gt)
    assert r.tp == 5 and abs(r.corner_error - 5.0) < 1e-9
    a = aggregate([r])
    assert abs(a.point_error["apex"] - 5.0) < 1e-9 and a.precision == a.recall == 1.0 and a.exact_rate == 1.0


def test_null_and_oracle_on_corpus(pairs):
    sk = {p.key: load_skeleton(p) for p in pairs}
    null = evaluate(NullPredictor(), pairs, skeletons=sk)
    assert null.overall.tp == 0 and null.overall.recall == 0.0 and null.overall.fn == 2726
    oracle = evaluate(OraclePredictor(sk), pairs, skeletons=sk)
    a = oracle.overall
    assert a.precision == a.recall == a.f1 == 1.0 and a.corner_error == 0.0 and a.exact_rate == 1.0
    assert set(oracle.per_font) == {"Assurbanipal", "Santakku"}
    assert sum(v.n_glyphs for v in oracle.per_font.values()) == 449


def test_jittered_oracle_between_floor_and_ceiling(pairs):
    sk = {p.key: load_skeleton(p) for p in pairs}
    sub = pairs[:60]
    r = evaluate(JitteredOraclePredictor(sigma=2.0, drop=0.1, spurious=0.05, skeletons=sk), sub, skeletons=sk)
    a = r.overall
    assert 0.8 < a.recall < 1.0 and 0.85 < a.precision <= 1.0
    assert 1.5 < a.corner_error < 4.0
    assert 0 < a.exact_rate < 1
    assert r.worst(3)[0].severity >= r.worst(3)[-1].severity


def test_comparison_panel(pairs):
    p = pairs[0]
    gt = load_skeleton(p).wedges
    m, up, ug = match_wedges(gt[:-1], gt)
    im = comparison_panel(load_image(p), gt, gt[:-1], m, up, ug, scale=1)
    assert im.size == (256 * 2 + 6, 256)
