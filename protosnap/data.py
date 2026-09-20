"""Dataset indexing, CSV parsing and wedge canonicalization for ProtoSnapVectorMapper.

Data conventions (see README):
  prototypes/<Font>/<range>/<hex>.png        rendered glyph, 256x256 for Assurbanipal/Santakku
  skeletons/<Font>/<hex>_adf.csv             label,x,y   one row per point, grouped by stroke
  skeletons/<Font>/<hex>_con.csv             label,i,j   edges between point indices within a stroke

A canonical wedge is a stroke whose connectivity graph is:
    apex -- head_a, apex -- head_b, apex -- tail, head_a -- head_b
i.e. a triangle (apex, head_a, head_b) with a tail hanging off the apex. The apex is
identified by topology (degree 3), never by index position, because annotators did not
use a consistent index order.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

FONTS_WITH_SKELETONS: tuple[str, ...] = ("Assurbanipal", "Santakku")
ALL_FONTS: tuple[str, ...] = ("Assurbanipal", "Santakku", "Esagil")
IMAGE_SIZE = 256
PROMOTED_LOG = "data/promoted.csv"   # written by `stage5 promote`: which skeletons under skeletons/ came from the model

# Stroke kinds produced by classify_stroke.
KIND_WEDGE = "wedge"          # exact 4-point wedge (possibly with stray isolated points, see orphans)
KIND_TRIANGLE = "triangle"    # 3-point closed triangle, no tail
KIND_MALFORMED = "malformed"  # anything else (self loops, duplicate edges, unknown topology)

_STROKE_RE = re.compile(r"^Stroke\s+(\d+)$")
Pt = tuple[float, float]


@dataclass(frozen=True)
class Pair:
    """One (image, skeleton) training pair."""
    font: str
    codepoint: str          # e.g. "0x12000"
    image_path: Path
    adf_path: Path
    con_path: Path
    corrected: bool = False   # True when the skeleton paths point at data/corrections/
    promoted: bool = False    # True when the skeleton was promoted from a model prediction (reviewed or hand-corrected),
                              # not drawn by the original annotators. Never part of validation or test.

    @property
    def key(self) -> str:
        return f"{self.font}/{self.codepoint}"


@dataclass(frozen=True)
class Wedge:
    """Canonical wedge. Head corners are ordered so that
    cross(head_a - apex, head_b - apex) > 0 in image coordinates (y down)."""
    apex: Pt
    head_a: Pt
    head_b: Pt
    tail: Pt

    def as_array(self) -> np.ndarray:
        """(4, 2) float array in the fixed order apex, head_a, head_b, tail."""
        return np.array([self.apex, self.head_a, self.head_b, self.tail], dtype=np.float64)


@dataclass
class Stroke:
    label: str
    number: int                       # parsed from "Stroke N"
    points: list[Pt]
    edges: list[tuple[int, int]]
    kind: str
    wedge: Wedge | None = None        # set when kind == KIND_WEDGE
    orphans: list[int] = field(default_factory=list)  # isolated point indices dropped from wedge
    note: str = ""                    # human-readable reason for triangle/malformed


@dataclass
class Skeleton:
    pair: Pair
    strokes: list[Stroke]

    @property
    def wedges(self) -> list[Wedge]:
        return [s.wedge for s in self.strokes if s.kind == KIND_WEDGE and s.wedge is not None]

    @property
    def flags(self) -> list[str]:
        out = []
        for s in self.strokes:
            if s.kind != KIND_WEDGE:
                out.append(f"{s.label}:{s.kind}")
            elif s.orphans:
                out.append(f"{s.label}:orphans={len(s.orphans)}")
        return out

    @property
    def is_clean(self) -> bool:
        return not self.flags


# --------------------------------------------------------------------------- indexing

def read_promoted(root: Path | str) -> set[tuple[str, str]]:
    path = Path(root) / PROMOTED_LOG
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["font"], r["codepoint"]) for r in csv.DictReader(f)}


def find_pairs(root: Path | str, fonts: Iterable[str] | None = None,
               corrections: Path | str | None = "data/corrections", include_promoted: bool = False) -> list[Pair]:
    """Index every codepoint that has an image and both skeleton files in the same font.

    By default only the original human annotations are returned, so every earlier stage and
    every evaluation sees the same 449-pair corpus it always did. Pass include_promoted=True to
    also get skeletons promoted from reviewed model output (marked Pair.promoted), which may
    include fonts whose images are not 256x256 (Esagil): use protosnap.imageprep for those.

    If `corrections` names a directory (relative to root) holding <Font>/<hex>_adf.csv and
    _con.csv for a glyph, those files replace the originals and the Pair is marked
    corrected. Pass corrections=None to always read the originals."""
    root = Path(root)
    corr_dir = None if corrections is None else root / corrections
    promoted = read_promoted(root)
    if fonts is None:
        fonts = ALL_FONTS if include_promoted else FONTS_WITH_SKELETONS
    pairs: list[Pair] = []
    for font in fonts:
        images = {p.stem: p for p in (root / "prototypes" / font).glob("*/*.png")}
        skel_dir = root / "skeletons" / font
        for adf in sorted(skel_dir.glob("*_adf.csv")):
            cp = adf.name[: -len("_adf.csv")]
            con = skel_dir / f"{cp}_con.csv"
            if cp not in images or not con.exists():
                continue
            is_promoted = (font, cp) in promoted
            if is_promoted and not include_promoted:
                continue
            if corr_dir is not None:
                c_adf, c_con = corr_dir / font / adf.name, corr_dir / font / con.name
                if c_adf.exists() and c_con.exists():
                    pairs.append(Pair(font, cp, images[cp], c_adf, c_con, corrected=True, promoted=is_promoted))
                    continue
            pairs.append(Pair(font, cp, images[cp], adf, con, promoted=is_promoted))
    return pairs


def load_metadata(root: Path | str) -> dict[str, dict[str, str]]:
    """metadata.csv keyed by hex codepoint."""
    with open(Path(root) / "prototypes" / "metadata.csv", newline="") as f:
        return {row["hex"]: row for row in csv.DictReader(f)}


# --------------------------------------------------------------------------- parsing

def _stroke_number(label: str) -> int:
    m = _STROKE_RE.match(label)
    if not m:
        raise ValueError(f"unexpected stroke label {label!r}")
    return int(m.group(1))


def read_points(adf_path: Path) -> dict[str, list[Pt]]:
    pts: dict[str, list[Pt]] = defaultdict(list)
    with open(adf_path, newline="") as f:
        for row in csv.DictReader(f):
            pts[row["label"]].append((float(row["x"]), float(row["y"])))
    return dict(pts)


def read_edges(con_path: Path) -> dict[str, list[tuple[int, int]]]:
    edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(con_path, newline="") as f:
        for row in csv.DictReader(f):
            edges[row["label"]].append((int(row["i"]), int(row["j"])))
    return dict(edges)


def _order_heads(apex: Pt, a: Pt, b: Pt) -> tuple[Pt, Pt]:
    ax, ay = a[0] - apex[0], a[1] - apex[1]
    bx, by = b[0] - apex[0], b[1] - apex[1]
    cross = ax * by - ay * bx
    if cross > 0 or (cross == 0 and a <= b):
        return a, b
    return b, a


def classify_stroke(label: str, points: list[Pt], edges: list[tuple[int, int]]) -> Stroke:
    """Turn raw points and edges into a Stroke with a canonical Wedge where possible."""
    n = len(points)
    number = _stroke_number(label)
    stroke = Stroke(label, number, points, edges, KIND_MALFORMED)

    if any(i == j or not (0 <= i < n and 0 <= j < n) for i, j in edges):
        stroke.note = "self loop or index out of range"
        return stroke
    eset = {frozenset(e) for e in edges}
    if len(eset) != len(edges):
        stroke.note = "duplicate edge"
        return stroke

    deg = [0] * n
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1

    if n == 3 and len(edges) == 3 and all(d == 2 for d in deg):
        stroke.kind = KIND_TRIANGLE
        stroke.note = "closed triangle with no tail"
        return stroke

    if len(edges) == 4:
        apex = [k for k in range(n) if deg[k] == 3]
        tail = [k for k in range(n) if deg[k] == 1]
        heads = [k for k in range(n) if deg[k] == 2]
        orphans = [k for k in range(n) if deg[k] == 0]
        if (len(apex), len(tail), len(heads)) == (1, 1, 2):
            a, t, (h1, h2) = apex[0], tail[0], heads
            wanted = {frozenset((a, h1)), frozenset((a, h2)), frozenset((a, t)), frozenset((h1, h2))}
            if eset == wanted:
                ha, hb = _order_heads(points[a], points[h1], points[h2])
                stroke.kind = KIND_WEDGE
                stroke.wedge = Wedge(points[a], ha, hb, points[t])
                stroke.orphans = orphans
                return stroke

    stroke.note = f"unrecognised topology: {n} points, {len(edges)} edges, degrees {sorted(deg, reverse=True)}"
    return stroke


def load_skeleton(pair: Pair) -> Skeleton:
    pts = read_points(pair.adf_path)
    edges = read_edges(pair.con_path)
    strokes = [classify_stroke(lab, p, edges.get(lab, [])) for lab, p in pts.items()]
    strokes.sort(key=lambda s: s.number)  # one file lists Stroke 13 before Stroke 12
    return Skeleton(pair, strokes)


def load_image(pair: Pair) -> np.ndarray:
    """Grayscale uint8 array, shape (H, W). Ink is dark on a light background."""
    with Image.open(pair.image_path) as im:
        return np.asarray(im.convert("L"))


def load_all(root: Path | str, fonts: Iterable[str] = FONTS_WITH_SKELETONS) -> list[Skeleton]:
    return [load_skeleton(p) for p in find_pairs(root, fonts)]
