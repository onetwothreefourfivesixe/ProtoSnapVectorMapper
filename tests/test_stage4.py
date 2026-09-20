from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from protosnap import Wedge, find_pairs, load_image, load_skeleton
from protosnap.hybrid import HybridParams, HybridPredictor, ScoredWedge, _fix_orientation, glyph_confidence
from protosnap.metrics import evaluate, score_glyph, score_glyph_ignoring
from protosnap.stage4 import auroc

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "stage4_unet_xc.pt"          # shipped weights, so this runs in a fresh clone too


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) - 0.5) < 1e-9
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))


def test_glyph_confidence_is_the_weakest_wedge():
    w = Wedge((0, 0), (1, 0), (0, 1), (5, 5))
    sw = [ScoredWedge(w, c, 1.0, "network", 0.0) for c in (0.9, 0.3, 0.7)]
    assert glyph_confidence(sw) == 0.3 and glyph_confidence([]) == 0.0


def test_orientation_rule_swaps_only_rare_tails():
    P = HybridParams()
    ok = Wedge((100, 100), (90, 90), (90, 110), (140, 100))          # tail right: untouched
    assert _fix_orientation(ok, P) == ok
    rare = Wedge((100, 100), (140, 100), (90, 110), (100, 60))       # tail up, one head points right
    fixed = _fix_orientation(rare, P)
    assert fixed.tail == (140, 100) and (100, 60) in (fixed.head_a, fixed.head_b)


def test_secondary_metric_skips_only_unmatched_predictions_near_non_wedge_strokes():
    pairs = {p.key: p for p in find_pairs(ROOT)}
    p = pairs["Santakku/0x1238f"]                 # strokes 1, 2, 5, 6, 7 are tail-less triangles
    s = load_skeleton(p)
    tri = next(st for st in s.strokes if st.kind == "triangle")
    x, y = tri.points[0]
    spurious = Wedge((x + 2, y), (x - 8, y - 8), (x + 12, y - 8), (x + 2, y + 30))
    pred = list(s.wedges) + [spurious]
    assert score_glyph(p, pred, s.wedges).fp == 1
    r = score_glyph_ignoring(p, pred, s, radius=14.0)
    assert r.fp == 0 and r.tp == len(s.wedges)     # matched wedges are never discarded


@pytest.mark.skipif(not CKPT.exists(), reason="no shipped weights")
def test_hybrid_refines_tails_and_scores_confidence():
    from protosnap.learned.predictor import LearnedPredictor
    from protosnap.split import load_splits, pairs_in_split
    pairs = find_pairs(ROOT)
    val = pairs_in_split(pairs, load_splits(ROOT / "data" / "splits.json"), "val")[:12]
    sk = {p.key: load_skeleton(p) for p in val}
    learned = LearnedPredictor.from_checkpoint(CKPT)
    hybrid = HybridPredictor(learned)
    off = hybrid.with_params(refine_tail=False, reject_off_ink=False)
    for p in val[:4]:
        img = load_image(p)
        assert off.predict(img, p) == learned.predict(img, p)        # all refinements off == proposer
        sw = hybrid.predict_scored(img, p)
        assert all(0.0 <= s.confidence <= 1.0 and s.tail_source in ("network", "skeleton") for s in sw)
        base = learned.predict(img, p)
        assert [s.wedge.apex for s in sw] == [w.apex for w in base]   # apexes and heads are the network's
    a, b = evaluate(hybrid, val, skeletons=sk).overall, evaluate(learned, val, skeletons=sk).overall
    assert a.f1 == b.f1 and a.point_error["tail"] <= b.point_error["tail"] + 0.5
