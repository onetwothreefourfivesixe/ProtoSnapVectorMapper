"""Overlay and contact-sheet rendering for skeleton review."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import KIND_MALFORMED, KIND_TRIANGLE, KIND_WEDGE, Skeleton

PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]
COLOR_TRIANGLE = (255, 0, 255)
COLOR_MALFORMED = (255, 0, 0)
COLOR_ORPHAN = (255, 0, 0)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def overlay(image: np.ndarray, skeleton: Skeleton, scale: int = 2, line: int = 2) -> Image.Image:
    """Draw the skeleton on the glyph. Wedges: coloured triangle + tail, apex filled dot,
    tail hollow dot. Triangles (no tail) magenta. Malformed strokes red raw edges.
    Orphan points red crosses."""
    base = Image.fromarray(image).convert("RGB")
    if scale != 1:
        base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
    # lighten so the overlay reads
    base = Image.blend(base, Image.new("RGB", base.size, (255, 255, 255)), 0.45)
    d = ImageDraw.Draw(base)
    S = lambda p: (p[0] * scale, p[1] * scale)
    r = 3 * scale // 2 + 1

    for k, s in enumerate(skeleton.strokes):
        col = PALETTE[k % len(PALETTE)]
        if s.kind == KIND_WEDGE and s.wedge is not None:
            w = s.wedge
            d.polygon([S(w.apex), S(w.head_a), S(w.head_b)], outline=col, width=line)
            d.line([S(w.apex), S(w.tail)], fill=col, width=line)
            ax, ay = S(w.apex)
            d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=col)
            tx, ty = S(w.tail)
            d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=col, width=line)
            for o in s.orphans:
                ox, oy = S(s.points[o])
                d.line([ox - r, oy - r, ox + r, oy + r], fill=COLOR_ORPHAN, width=line)
                d.line([ox - r, oy + r, ox + r, oy - r], fill=COLOR_ORPHAN, width=line)
        else:
            col = COLOR_TRIANGLE if s.kind == KIND_TRIANGLE else COLOR_MALFORMED
            for i, j in s.edges:
                d.line([S(s.points[i]), S(s.points[j])], fill=col, width=line)
            for p in s.points:
                px, py = S(p)
                d.ellipse([px - r, py - r, px + r, py + r], outline=col, width=line)
    return base


def contact_sheet(
    cells: Sequence[tuple[Image.Image, str]],
    cols: int = 8,
    label_height: int = 36,
    pad: int = 4,
    title: str | None = None,
) -> Image.Image:
    """Grid of (image, caption) cells. All images are assumed the same size."""
    if not cells:
        raise ValueError("no cells")
    cw, ch = cells[0][0].size
    rows = (len(cells) + cols - 1) // cols
    title_h = 40 if title else 0
    W = cols * (cw + pad) + pad
    H = title_h + rows * (ch + label_height + pad) + pad
    sheet = Image.new("RGB", (W, H), (240, 240, 240))
    d = ImageDraw.Draw(sheet)
    f_label, f_title = _font(13), _font(22)
    if title:
        d.text((pad + 4, 8), title, fill=(0, 0, 0), font=f_title)
    for idx, (im, caption) in enumerate(cells):
        r, c = divmod(idx, cols)
        x = pad + c * (cw + pad)
        y = title_h + pad + r * (ch + label_height + pad)
        sheet.paste(im, (x, y))
        d.rectangle([x, y + ch, x + cw, y + ch + label_height], fill=(255, 255, 255))
        d.multiline_text((x + 3, y + ch + 2), caption, fill=(0, 0, 0), font=f_label, spacing=1)
    return sheet


def save_sheets(cells: Sequence[tuple[Image.Image, str]], out_dir: Path, stem: str,
                per_page: int = 48, cols: int = 8, title: str = "") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pages = (len(cells) + per_page - 1) // per_page
    for pg in range(pages):
        chunk = cells[pg * per_page:(pg + 1) * per_page]
        sheet = contact_sheet(chunk, cols=cols, title=f"{title}  page {pg + 1}/{pages}")
        p = out_dir / f"{stem}_{pg + 1:02d}.png"
        sheet.save(p, optimize=True)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- evaluation views

COLOR_MATCH = (31, 119, 180)   # blue: matched wedge
COLOR_MISS = (214, 39, 40)     # red: unmatched (FN on the GT panel, FP on the prediction panel)


def _draw_wedge(d: ImageDraw.ImageDraw, w, scale: int, col, line: int, r: int) -> None:
    S = lambda p: (p[0] * scale, p[1] * scale)
    d.polygon([S(w.apex), S(w.head_a), S(w.head_b)], outline=col, width=line)
    d.line([S(w.apex), S(w.tail)], fill=col, width=line)
    ax, ay = S(w.apex)
    d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=col)
    tx, ty = S(w.tail)
    d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=col, width=line)


def overlay_wedges(image: np.ndarray, wedges, colors, scale: int = 2, line: int = 2) -> Image.Image:
    """Draw a list of wedges with one colour per wedge."""
    base = Image.fromarray(image).convert("RGB")
    if scale != 1:
        base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
    base = Image.blend(base, Image.new("RGB", base.size, (255, 255, 255)), 0.45)
    d = ImageDraw.Draw(base)
    r = 3 * scale // 2 + 1
    for w, col in zip(wedges, colors):
        _draw_wedge(d, w, scale, col, line, r)
    return base


def comparison_panel(image: np.ndarray, gt, pred, matches, unmatched_pred, unmatched_gt,
                     scale: int = 2) -> Image.Image:
    """Ground truth (left) next to prediction (right). Matched wedges blue, unmatched red."""
    fn = set(unmatched_gt)
    fp = set(unmatched_pred)
    left = overlay_wedges(image, gt, [COLOR_MISS if i in fn else COLOR_MATCH for i in range(len(gt))], scale)
    right = overlay_wedges(image, pred, [COLOR_MISS if i in fp else COLOR_MATCH for i in range(len(pred))], scale)
    out = Image.new("RGB", (left.width * 2 + 6, left.height), (200, 200, 200))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 6, 0))
    return out
