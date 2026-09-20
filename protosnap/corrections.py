"""Manual and semi-automatic corrections to skeleton annotations.

A correction is a pair of CSV files with the same format and names as the originals,
stored under  data/corrections/<Font>/<hex>_adf.csv  and  _con.csv.  When present,
find_pairs() uses them instead of the originals and marks the Pair as corrected. The
original files are never modified, so a correction can be reviewed, hand-edited or
deleted at any time.

The one automatic repair implemented here handles the "missing apex" annotation
pattern seen in HI TIMES BAD (Santakku 0x12130): each stroke has the two head corners
and the tail endpoint but no apex, so it parses as a 3-point triangle. The apex is
reconstructed on the axis from the head midpoint toward the tail, at a fixed fraction of
the head width. The fraction (default 0.45) is the median of the same measurement over
the 2,720 clean wedges in the corpus (p10 0.33, p90 0.62).

    python -m protosnap.corrections reconstruct --font Santakku --codepoint 0x12130
    python -m protosnap.corrections render      --font Santakku --codepoint 0x12130
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .data import (KIND_TRIANGLE, Pair, Pt, Skeleton, find_pairs, load_image, load_skeleton,
                   read_edges, read_points)
from .render import overlay

DEFAULT_CORRECTIONS_DIR = "data/corrections"
APEX_FRACTION_OF_HEAD_WIDTH = 0.45
# Output order and edges for a reconstructed wedge, matching the dominant convention in the
# corpus: points [head_a, head_b, apex, tail], edges apex-head_a, apex-head_b, apex-tail, head_a-head_b.
WEDGE_EDGES = [(2, 0), (2, 1), (2, 3), (0, 1)]


def reconstruct_missing_apex(points: list[Pt], k: float = APEX_FRACTION_OF_HEAD_WIDTH) -> list[Pt]:
    """Given the 3 points of a tail-less triangle (two head corners + tail end), return
    [head_a, head_b, apex, tail]. The tail is the vertex opposite the shortest edge."""
    if len(points) != 3:
        raise ValueError("expected 3 points")
    P = np.array(points, dtype=np.float64)
    edge_len = {(0, 1): np.linalg.norm(P[0] - P[1]), (1, 2): np.linalg.norm(P[1] - P[2]), (0, 2): np.linalg.norm(P[0] - P[2])}
    ha, hb = min(edge_len, key=edge_len.get)
    t = ({0, 1, 2} - {ha, hb}).pop()
    M = (P[ha] + P[hb]) / 2
    axis = P[t] - M
    L = np.linalg.norm(axis)
    if L < 1e-9:
        raise ValueError("tail coincides with head midpoint")
    apex = M + axis / L * (k * edge_len[(ha, hb)])
    return [tuple(P[ha]), tuple(P[hb]), (float(apex[0]), float(apex[1])), tuple(P[t])]


def corrected_tables(skeleton: Skeleton, strokes: set[int] | None = None,
                     k: float = APEX_FRACTION_OF_HEAD_WIDTH) -> tuple[dict, dict, list[int]]:
    """Rebuild (points_by_label, edges_by_label) with missing apexes reconstructed for the
    triangle strokes selected (all triangle strokes when `strokes` is None).
    Returns the tables and the list of stroke numbers that were changed."""
    pts = read_points(skeleton.pair.adf_path)
    edges = read_edges(skeleton.pair.con_path)
    changed = []
    for st in skeleton.strokes:
        if st.kind != KIND_TRIANGLE or (strokes is not None and st.number not in strokes):
            continue
        pts[st.label] = reconstruct_missing_apex(st.points, k)
        edges[st.label] = list(WEDGE_EDGES)
        changed.append(st.number)
    return pts, edges, changed


def write_tables(pts: dict[str, list[Pt]], edges: dict[str, list[tuple[int, int]]], adf: Path, con: Path,
                 font: str | None = None) -> None:
    """Write strokes keeping their labels, numbering and edge lists exactly as given: a correction
    of an original annotation must not renumber the annotator's strokes. With `font`, the column
    order and number format follow that font's original layout (see protosnap.skeleton_io):
    integers for Santakku and Esagil, full-precision floats for Assurbanipal."""
    from .skeleton_io import LAYOUT_A, layout_for
    layout = layout_for(font) if font else LAYOUT_A
    adf.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(pts, key=lambda s: int(s.split()[1]))

    def fmt(v: float) -> str:
        if font:
            return layout.number(v)
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"

    def row(header, label, a, b):
        rest = [h for h in header if h != "label"]
        vals = {"label": label, rest[0]: a, rest[1]: b}
        return [vals[h] for h in header]

    with open(adf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(layout.adf_header)
        for lab in labels:
            for x, y in pts[lab]:
                w.writerow(row(layout.adf_header, lab, fmt(x), fmt(y)))
    with open(con, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(layout.con_header)
        for lab in labels:
            for i, j in edges.get(lab, []):
                w.writerow(row(layout.con_header, lab, i, j))


def correction_paths(root: Path, corrections_dir: Path | str, pair: Pair) -> tuple[Path, Path]:
    d = root / corrections_dir / pair.font
    return d / f"{pair.codepoint}_adf.csv", d / f"{pair.codepoint}_con.csv"


def render_before_after(root: Path, pair_original: Pair, pair_corrected: Pair, out_png: Path) -> None:
    img = load_image(pair_original)
    a = overlay(img, load_skeleton(pair_original), scale=3)
    b = overlay(img, load_skeleton(pair_corrected), scale=3)
    sheet = Image.new("RGB", (a.width + b.width + 12, a.height), (240, 240, 240))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (a.width + 12, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["reconstruct", "render"])
    ap.add_argument("--font", required=True)
    ap.add_argument("--codepoint", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--corrections", default=DEFAULT_CORRECTIONS_DIR)
    ap.add_argument("--k", type=float, default=APEX_FRACTION_OF_HEAD_WIDTH,
                    help="apex distance from head midpoint, as a fraction of head width")
    ap.add_argument("--strokes", default=None, help="comma-separated stroke numbers to repair (default: all triangle strokes)")
    ap.add_argument("--from-original", action="store_true",
                    help="start from the original annotation, discarding any existing correction (default: build on the existing correction)")
    ap.add_argument("--report", default="reports/stage0/corrections")
    args = ap.parse_args(argv)

    root = Path(args.root)
    originals = {p.key: p for p in find_pairs(root, corrections=None)}
    key = f"{args.font}/{args.codepoint}"
    if key not in originals:
        ap.error(f"no pair {key}")
    orig = originals[key]
    adf, con = correction_paths(root, args.corrections, orig)

    if args.command == "reconstruct":
        sel = {int(s) for s in args.strokes.split(",")} if args.strokes else None
        base = orig
        if adf.exists() and con.exists() and not args.from_original:
            base = Pair(orig.font, orig.codepoint, orig.image_path, adf, con, corrected=True)
            print(f"building on existing correction {adf} (use --from-original to discard it)")
        pts, edges, changed = corrected_tables(load_skeleton(base), sel, args.k)
        if not changed:
            print("nothing to repair: no triangle strokes selected")
            return 1
        write_tables(pts, edges, adf, con, font=orig.font)
        print(f"reconstructed apex for strokes {changed} -> {adf}, {con}")
    elif not (adf.exists() and con.exists()):
        ap.error(f"no correction files at {adf}")

    corrected = Pair(orig.font, orig.codepoint, orig.image_path, adf, con, corrected=True)
    s = load_skeleton(corrected)
    out_png = Path(args.report) / f"{orig.font}_{orig.codepoint}.png"
    render_before_after(root, orig, corrected, out_png)
    print(f"corrected skeleton: {len(s.wedges)} wedges, flags={s.flags or 'none'}; before/after -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
