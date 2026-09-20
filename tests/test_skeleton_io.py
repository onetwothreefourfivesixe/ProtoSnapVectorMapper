"""Generated skeleton CSVs must be written the way the original annotations are."""
import csv
from pathlib import Path

import pytest

from protosnap import Pair, Wedge, find_pairs, load_skeleton
from protosnap.skeleton_io import LAYOUT_A, LAYOUT_B, deviations, layout_for, ordered_points, write_wedges

ROOT = Path(__file__).resolve().parents[1]
W1 = Wedge((119.4, 103.2), (126.7, 97.1), (106.2, 90.6), (119.0, 149.8))      # head_a is the lower-right one here
W2 = Wedge((60.0, 80.0), (50.0, 70.0), (50.0, 92.0), (100.0, 80.0))


def test_layouts_match_the_dominant_original_layout_per_font():
    """Read the layout off the original files rather than trusting the constants."""
    seen = {}
    for p in find_pairs(ROOT, corrections=None):
        head = p.adf_path.read_bytes().split(b"\n")[0].strip().decode()
        seen.setdefault(p.font, {}).setdefault(head, 0)
        seen[p.font][head] += 1
    for font, counts in seen.items():
        dominant = max(counts, key=counts.get)
        assert ",".join(layout_for(font).adf_header) == dominant, (font, counts)
    assert layout_for("Esagil") is LAYOUT_A and layout_for("Santakku") is LAYOUT_A and layout_for("Assurbanipal") is LAYOUT_B


@pytest.mark.parametrize("font,layout", [("Santakku", LAYOUT_A), ("Esagil", LAYOUT_A), ("Assurbanipal", LAYOUT_B)])
def test_written_bytes_follow_the_original_format(tmp_path, font, layout):
    adf, con = tmp_path / "g_adf.csv", tmp_path / "g_con.csv"
    write_wedges([W1, W2], adf, con, font)
    assert deviations(adf, con, font) == []
    raw = adf.read_bytes()
    assert raw.startswith(",".join(layout.adf_header).encode() + b"\r\n") and raw.endswith(b"\r\n") and b"\n\n" not in raw
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert [r["label"] for r in rows] == ["Stroke 1"] * 4 + ["Stroke 2"] * 4
    assert rows[0]["x"] == ("50" if layout.integers else "50.0")                 # W2 has the smaller apex x, so it is Stroke 1
    assert all(("." in r["x"]) != layout.integers for r in rows)
    edges = [(int(r["i"]), int(r["j"])) for r in csv.DictReader(con.read_bytes().decode().splitlines())]
    assert tuple(edges[:4]) == layout.edges and tuple(edges[4:]) == layout.edges


def test_point_and_head_order():
    pts = ordered_points(W1)
    assert pts[2] == W1.apex and pts[3] == W1.tail            # head, head, apex, tail
    assert pts[0] == (106.2, 90.6) and pts[1] == (126.7, 97.1)   # upper head first
    assert ordered_points(W2)[:2] == [(50.0, 70.0), (50.0, 92.0)]


@pytest.mark.parametrize("font,tol", [("Assurbanipal", 1e-9), ("Esagil", 0.5)])
def test_roundtrip_through_the_loader(tmp_path, font, tol):
    adf, con = tmp_path / "g_adf.csv", tmp_path / "g_con.csv"
    write_wedges([W1, W2], adf, con, font)
    s = load_skeleton(Pair(font, "0x12000", adf, adf, con))
    assert s.is_clean and len(s.wedges) == 2
    got = sorted(s.wedges, key=lambda w: w.apex[0])
    close = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= tol
    for g, w in zip(got, sorted([W1, W2], key=lambda w: w.apex[0])):
        assert close(g.apex, w.apex) and close(g.tail, w.tail)
        # the loader puts the two head tips into its own canonical order, so compare them as a pair
        assert (close(g.head_a, w.head_a) and close(g.head_b, w.head_b)) or (close(g.head_a, w.head_b) and close(g.head_b, w.head_a))


@pytest.mark.skipif(not (ROOT / "generated" / "manifest.csv").exists(), reason="stage 5 has not been run")
def test_every_generated_file_conforms():
    rows = list(csv.DictReader(open(ROOT / "generated" / "manifest.csv")))
    model = [r for r in rows if r["source"] == "model"]
    bad = {}
    for r in model:
        adf = ROOT / "generated" / r["font"] / f"{r['codepoint']}_adf.csv"
        d = deviations(adf, adf.with_name(adf.name.replace("_adf", "_con")), r["font"])
        if d:
            bad[f"{r['font']}/{r['codepoint']}"] = d
    assert not bad, dict(list(bad.items())[:3])
    reviewed = ROOT / "data" / "reviewed"
    for adf in reviewed.glob("*/*_adf.csv"):
        assert deviations(adf, adf.with_name(adf.name.replace("_adf", "_con")), adf.parent.name) == [], adf
