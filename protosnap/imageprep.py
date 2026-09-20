"""Bring any glyph image to the 256x256 canvas and scale the models are trained at.

Native 256x256 glyphs (Assurbanipal, Santakku) pass through unchanged. Esagil is the same outline
style rendered about 6.4 times larger at variable width; it is area-resampled so its 512 px em
height becomes 80 px and centred. The transform is an exact inverse, so skeleton coordinates
can be kept in original-image pixels on disk and mapped to model space only when needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .data import IMAGE_SIZE

TARGET_HEIGHT = 80      # px; a 512 px Esagil em box maps to the scale the training fonts are drawn at
MAX_WIDTH = 240         # px; very wide glyphs are scaled down further to fit the 256 px canvas


@dataclass(frozen=True)
class Transform:
    """Maps between original image pixels and the 256x256 model canvas (pixel centres at integers)."""
    sx: float = 1.0
    sy: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    def to_model(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=float)
        return (p + 0.5) * np.array([self.sx, self.sy]) + np.array([self.ox, self.oy]) - 0.5

    def to_original(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=float)
        return (p + 0.5 - np.array([self.ox, self.oy])) / np.array([self.sx, self.sy]) - 0.5


def prepare_image(path: Path) -> tuple[np.ndarray, Transform]:
    """Grayscale 256x256 model input. Native 256x256 glyphs pass through unchanged; anything else
    (Esagil: variable width, 512 px high) is area-resampled to the training scale and centred."""
    with Image.open(path) as im:
        g = im.convert("L")
        if g.size == (IMAGE_SIZE, IMAGE_SIZE):
            return np.asarray(g), Transform()
        s = min(TARGET_HEIGHT / g.height, MAX_WIDTH / g.width)
        w, h = max(1, round(g.width * s)), max(1, round(g.height * s))
        small = g.resize((w, h), Image.LANCZOS)
        canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        ox, oy = (IMAGE_SIZE - w) // 2, (IMAGE_SIZE - h) // 2
        canvas.paste(small, (ox, oy))
        return np.asarray(canvas), Transform(w / g.width, h / g.height, float(ox), float(oy))
