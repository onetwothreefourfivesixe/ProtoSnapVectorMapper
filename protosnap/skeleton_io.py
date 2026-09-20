"""Write skeleton CSVs the way the original annotations are written.

The originals come in two layouts. Both use CRLF line endings, a final newline, "Stroke N" labels,
four points per wedge in the order head, head, apex, tail, and four edge rows per wedge.

  Layout A  adf "label,x,y"   con "label,i,j"   integer coordinates      edges 2-0, 2-1, 2-3, 0-1
            all 288 Santakku files and 70 Assurbanipal files
  Layout B  adf "x,y,label"   con "i,j,label"   full-precision floats    edges 0-1, 0-2, 1-2, 2-3
            91 Assurbanipal files (the majority in that font)

Generated files use the dominant layout of their font: Santakku A, Assurbanipal B. Esagil has no
original annotations; it uses layout A, the only layout every font with annotations shares.

Two ordering conventions are followed as closely as the originals allow (neither is strict in
the originals): strokes run left to right by apex (71% of consecutive original strokes do), and
of the two head tips the upper one is listed first, the left one on ties (67% of originals).

    python -m protosnap.skeleton_io convert generated data/reviewed          # rewrite existing files in place
    python -m protosnap.skeleton_io check generated                          # report files that deviate
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .data import ALL_FONTS, Pair, Wedge, load_skeleton


@dataclass(frozen=True)
class Layout:
    name: str
    adf_header: tuple[str, str, str]
    con_header: tuple[str, str, str]
    edges: tuple[tuple[int, int], ...]
    integers: bool

    def number(self, v: float) -> str:
        return str(int(round(v))) if self.integers else repr(float(v))


LAYOUT_A = Layout("A", ("label", "x", "y"), ("label", "i", "j"), ((2, 0), (2, 1), (2, 3), (0, 1)), True)
LAYOUT_B = Layout("B", ("x", "y", "label"), ("i", "j", "label"), ((0, 1), (0, 2), (1, 2), (2, 3)), False)
FONT_LAYOUT = {"Santakku": LAYOUT_A, "Assurbanipal": LAYOUT_B, "Esagil": LAYOUT_A}


def layout_for(font: str) -> Layout:
    return FONT_LAYOUT.get(font, LAYOUT_A)


def ordered_points(w: Wedge) -> list[tuple[float, float]]:
    """head, head, apex, tail with the upper head first (left one on ties)."""
    h0, h1 = sorted((w.head_a, w.head_b), key=lambda p: (p[1], p[0]))
    return [h0, h1, w.apex, w.tail]


def _row(layout_header: Sequence[str], label: str, a: str, b: str) -> list[str]:
    vals = {"label": label, layout_header[[h != "label" for h in layout_header].index(True)]: a}
    rest = [h for h in layout_header if h != "label"]
    vals[rest[0]], vals[rest[1]] = a, b
    return [vals[h] for h in layout_header]


def write_wedges(wedges: Iterable[Wedge], adf: Path, con: Path, font: str, layout: Layout | None = None) -> Layout:
    L = layout or layout_for(font)
    ws = sorted(wedges, key=lambda w: (w.apex[0], w.apex[1]))
    adf.parent.mkdir(parents=True, exist_ok=True)
    with open(adf, "w", newline="") as f:                      # csv.writer terminates rows with CRLF, like the originals
        out = csv.writer(f)
        out.writerow(L.adf_header)
        for i, w in enumerate(ws, start=1):
            for x, y in ordered_points(w):
                out.writerow(_row(L.adf_header, f"Stroke {i}", L.number(x), L.number(y)))
    with open(con, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(L.con_header)
        for i in range(1, len(ws) + 1):
            for a, b in L.edges:
                out.writerow(_row(L.con_header, f"Stroke {i}", str(a), str(b)))
    return L


def deviations(adf: Path, con: Path, font: str) -> list[str]:
    """Ways a file differs from the layout its font should use (empty list = conforms)."""
    L = layout_for(font)
    out = []
    raw, rawc = adf.read_bytes(), con.read_bytes()
    for name, data, header in (("adf", raw, L.adf_header), ("con", rawc, L.con_header)):
        if data.split(b"\n")[0].strip().decode() != ",".join(header):
            out.append(f"{name} header is not {','.join(header)}")
        if b"\r\n" not in data or data.replace(b"\r\n", b"").count(b"\n"):
            out.append(f"{name} does not use CRLF line endings")
        if not data.endswith(b"\n"):
            out.append(f"{name} has no final newline")
    rows = list(csv.DictReader(raw.decode().splitlines()))
    if L.integers and any("." in r["x"] or "." in r["y"] for r in rows):
        out.append("coordinates are not integers")
    labels = [r["label"] for r in rows]
    if labels[::4] != [f"Stroke {i}" for i in range(1, len(labels) // 4 + 1)] or len(labels) % 4:
        out.append("strokes are not four points each, labelled Stroke 1..N in order")
    crows = list(csv.DictReader(rawc.decode().splitlines()))
    per = [tuple((int(r["i"]), int(r["j"])) for r in crows[k:k + 4]) for k in range(0, len(crows), 4)]
    if any(p != L.edges for p in per):
        out.append(f"edge rows are not {L.edges}")
    return out


def _files(root: Path):
    for font in ALL_FONTS:
        for adf in sorted((root / font).glob("*_adf.csv")):
            con = adf.with_name(adf.name.replace("_adf.csv", "_con.csv"))
            if con.exists():
                yield font, adf.name[:-8], adf, con


def convert_tree(root: Path, skip: set[tuple[str, str]] = frozenset(), only_fonts: set[str] | None = None,
                 only: set[tuple[str, str]] | None = None) -> dict:
    done, skipped, unclean = 0, 0, []
    for font, cp, adf, con in _files(root):
        if (font, cp) in skip or (only_fonts and font not in only_fonts) or (only is not None and (font, cp) not in only):
            skipped += 1
            continue
        s = load_skeleton(Pair(font, cp, adf, adf, con))
        if not s.is_clean:
            unclean.append(f"{font}/{cp}")        # has non-wedge strokes: leave exactly as it is
            continue
        write_wedges(s.wedges, adf, con, font)
        done += 1
    return {"converted": done, "skipped": skipped, "left_alone_non_wedge": unclean}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["convert", "check"])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--font", action="append", choices=list(ALL_FONTS))
    args = ap.parse_args(argv)
    for d in args.dirs:
        root = Path(d)
        if args.command == "check":
            bad = {f"{font}/{cp}": dv for font, cp, adf, con in _files(root) if (dv := deviations(adf, con, font))}
            n = sum(1 for _ in _files(root))
            print(f"{d}: {n - len(bad)} of {n} files conform" + ("" if not bad else f"; first deviations: {dict(list(bad.items())[:3])}"))
        else:
            skip = set()
            man = root / "manifest.csv"
            if man.exists():          # byte copies of original human annotations keep their original bytes
                with open(man, newline="") as f:
                    skip = {(r["font"], r["codepoint"]) for r in csv.DictReader(f) if r["source"].startswith("copied_human")}
            print(d, convert_tree(root, skip, set(args.font) if args.font else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
