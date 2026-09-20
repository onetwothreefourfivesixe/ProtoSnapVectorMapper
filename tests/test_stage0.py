from pathlib import Path

import numpy as np
import pytest

from protosnap import (KIND_MALFORMED, KIND_TRIANGLE, KIND_WEDGE, find_pairs, load_image,
                       load_skeleton, make_splits, pairs_in_split)
from protosnap.data import classify_stroke
from protosnap.render import contact_sheet, overlay

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pairs():
    """Raw corpus, corrections bypassed."""
    return find_pairs(ROOT, corrections=None)


@pytest.fixture(scope="module")
def skeletons(pairs):
    return [load_skeleton(p) for p in pairs]


def test_corrections_change_only_targeted_glyphs(pairs):
    corrected = find_pairs(ROOT)
    assert len(corrected) == len(pairs)
    diff = [(a, b) for a, b in zip(pairs, corrected) if a != b]
    assert all(b.corrected and a.key == b.key for a, b in diff)
    assert {b.key for _, b in diff} >= {"Santakku/0x12130"}


def test_pair_counts(pairs):
    by_font = {}
    for p in pairs:
        by_font[p.font] = by_font.get(p.font, 0) + 1
    assert by_font == {"Assurbanipal": 161, "Santakku": 288}
    assert len({p.key for p in pairs}) == len(pairs)


def test_known_wedge_canonicalization(pairs):
    p = next(p for p in pairs if p.font == "Santakku" and p.codepoint == "0x12000")
    s = load_skeleton(p)
    assert [st.label for st in s.strokes] == ["Stroke 1", "Stroke 2", "Stroke 3"]
    w = s.strokes[0].wedge
    assert w.apex == (119.0, 103.0)
    assert w.tail == (119.0, 149.0)
    assert {w.head_a, w.head_b} == {(106.0, 90.0), (126.0, 97.0)}


def test_apex_found_by_topology_not_index():
    # apex at index 0, tail at index 1
    pts = [(10.0, 10.0), (10.0, 50.0), (0.0, 0.0), (20.0, 0.0)]
    edges = [(0, 2), (0, 3), (0, 1), (2, 3)]
    st = classify_stroke("Stroke 7", pts, edges)
    assert st.kind == KIND_WEDGE and st.number == 7
    assert st.wedge.apex == (10.0, 10.0) and st.wedge.tail == (10.0, 50.0)


def test_head_order_is_deterministic():
    pts = [(0.0, 0.0), (20.0, 0.0), (10.0, 10.0), (10.0, 50.0)]
    a = classify_stroke("Stroke 1", pts, [(2, 0), (2, 1), (2, 3), (0, 1)]).wedge
    b = classify_stroke("Stroke 1", [pts[1], pts[0], pts[2], pts[3]], [(2, 0), (2, 1), (2, 3), (0, 1)]).wedge
    assert a == b
    ax, ay = a.head_a[0] - a.apex[0], a.head_a[1] - a.apex[1]
    bx, by = a.head_b[0] - a.apex[0], a.head_b[1] - a.apex[1]
    assert ax * by - ay * bx > 0


def test_orphans_are_dropped_from_wedge():
    pts = [(0.0, 0.0), (20.0, 0.0), (10.0, 10.0), (10.0, 50.0), (99.0, 99.0)]
    st = classify_stroke("Stroke 1", pts, [(2, 0), (2, 1), (2, 3), (0, 1)])
    assert st.kind == KIND_WEDGE and st.orphans == [4]
    assert st.wedge.as_array().shape == (4, 2)


def test_triangle_and_malformed():
    tri = classify_stroke("Stroke 1", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [(0, 1), (1, 2), (2, 0)])
    assert tri.kind == KIND_TRIANGLE
    bad = classify_stroke("Stroke 1", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (5.0, 5.0)], [(0, 2), (1, 0), (1, 2), (3, 3)])
    assert bad.kind == KIND_MALFORMED


def test_corpus_stroke_kind_totals(skeletons):
    kinds = {}
    orphans = 0
    for s in skeletons:
        for st in s.strokes:
            kinds[st.kind] = kinds.get(st.kind, 0) + 1
            orphans += bool(st.orphans)
    assert kinds == {KIND_WEDGE: 2720, KIND_TRIANGLE: 34, KIND_MALFORMED: 1}
    assert orphans == 35
    assert sum(len(s.wedges) for s in skeletons) == 2720


def test_all_wedge_points_inside_image(skeletons):
    for s in skeletons:
        for w in s.wedges:
            arr = w.as_array()
            assert (arr >= 0).all() and (arr <= 256).all(), s.pair.key


def test_stroke_numbers_sorted(skeletons):
    for s in skeletons:
        nums = [st.number for st in s.strokes]
        assert nums == sorted(nums) == list(range(1, len(nums) + 1)), s.pair.key


def test_images_load(pairs):
    im = load_image(pairs[0])
    assert im.shape == (256, 256) and im.dtype == np.uint8


def test_splits_disjoint_complete_and_font_consistent(pairs):
    splits = make_splits(pairs, seed=0)
    sets = {n: set(splits[n]) for n in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert sets["train"] | sets["val"] | sets["test"] == {p.codepoint for p in pairs}
    assert "0x122b9" in sets["train"]
    # a codepoint present in both fonts lands in the same split for both
    for name in sets:
        got = pairs_in_split(pairs, splits, name)
        assert {p.codepoint for p in got} == sets[name]
    total = sum(len(pairs_in_split(pairs, splits, n)) for n in sets)
    assert total == len(pairs)
    assert make_splits(pairs, seed=0) == splits  # deterministic


def test_overlay_and_sheet(pairs):
    p = pairs[0]
    im = overlay(load_image(p), load_skeleton(p), scale=2)
    assert im.size == (512, 512)
    sheet = contact_sheet([(im, "a"), (im, "b")], cols=2)
    assert sheet.width > 1024


# ----------------------------------------------------------------------------- corrections

def test_reconstruct_missing_apex_geometry():
    from protosnap.corrections import reconstruct_missing_apex
    # heads 20 apart on y=0, tail 60 below the midpoint -> apex 0.45*20 = 9 below midpoint
    out = reconstruct_missing_apex([(0.0, 0.0), (50.0, 60.0), (20.0, 0.0)], k=0.45)
    assert out[3] == (50.0, 60.0)
    assert {out[0], out[1]} == {(0.0, 0.0), (20.0, 0.0)}
    M = np.array([10.0, 0.0]); axis = np.array([40.0, 60.0])
    expected = M + axis / np.linalg.norm(axis) * 0.45 * 20.0
    assert np.allclose(out[2], expected)


def test_hi_times_bad_is_corrected():
    orig = {p.key: p for p in find_pairs(ROOT, corrections=None)}["Santakku/0x12130"]
    assert not orig.corrected
    assert all(st.kind == KIND_TRIANGLE for st in load_skeleton(orig).strokes)
    corr = {p.key: p for p in find_pairs(ROOT)}["Santakku/0x12130"]
    assert corr.corrected and "corrections" in str(corr.adf_path)
    s = load_skeleton(corr)
    assert s.is_clean and len(s.wedges) == 6
    # heads are unchanged from the original annotation
    o = load_skeleton(orig).strokes[0].points
    w = s.strokes[0].wedge
    assert {w.head_a, w.head_b} <= set(o) and w.tail in o and w.apex not in o


def test_write_tables_roundtrip(tmp_path):
    from protosnap.corrections import corrected_tables, write_tables
    from protosnap.data import read_edges, read_points
    orig = {p.key: p for p in find_pairs(ROOT, corrections=None)}["Santakku/0x12130"]
    pts, edges, changed = corrected_tables(load_skeleton(orig))
    assert changed == [1, 2, 3, 4, 5, 6]
    write_tables(pts, edges, tmp_path / "a.csv", tmp_path / "c.csv")
    back = read_points(tmp_path / "a.csv")
    assert back.keys() == pts.keys()
    for k in pts:
        assert np.allclose(back[k], pts[k], atol=0.006)
    assert read_edges(tmp_path / "c.csv") == edges
