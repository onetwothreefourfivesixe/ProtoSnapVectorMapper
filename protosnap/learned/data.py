"""Dataset and augmentation for the wedge detector.

Augmentation is deliberately mild on rotation and has no flips: Stage 2 showed that wedge
orientation carries role information (95% of tails point right, down or down-right), so the
model should be allowed to learn it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from ..data import KIND_WEDGE, Pair, Skeleton
from ..imageprep import prepare_image
from .targets import build_targets


@dataclass(frozen=True)
class AugmentConfig:
    rotate_deg: float = 8.0
    scale: tuple[float, float] = (0.85, 1.15)
    shear: float = 0.08
    translate: float = 10.0
    p_thicken: float = 0.25
    p_thin: float = 0.10
    blur_sigma: float = 0.7
    noise: float = 0.03
    gamma: tuple[float, float] = (0.8, 1.25)


def affine_matrix(rng: np.random.Generator, cfg: AugmentConfig, size: int) -> np.ndarray:
    """3x3 forward map in continuous coordinates (pixel i spans [i, i+1))."""
    th = np.deg2rad(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    s = rng.uniform(*cfg.scale)
    sx, sy = s * rng.uniform(0.95, 1.05), s * rng.uniform(0.95, 1.05)
    k = rng.uniform(-cfg.shear, cfg.shear)
    tx, ty = rng.uniform(-cfg.translate, cfg.translate, 2)
    c = size / 2.0
    T1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], float)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Sh = np.array([[1, k, 0], [0, 1, 0], [0, 0, 1]], float)
    Sc = np.diag([sx, sy, 1.0])
    T2 = np.array([[1, 0, c + tx], [0, 1, c + ty], [0, 0, 1]], float)
    return T2 @ R @ Sh @ Sc @ T1


def apply_affine_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts (..., 2) in pixel-index coordinates (pixel centre at integer)."""
    if pts.size == 0:
        return pts
    p = pts + 0.5
    out = p @ M[:2, :2].T + M[:2, 2]
    return out - 0.5


def apply_affine_image(M: np.ndarray, image: np.ndarray) -> np.ndarray:
    Minv = np.linalg.inv(M)
    data = (Minv[0, 0], Minv[0, 1], Minv[0, 2], Minv[1, 0], Minv[1, 1], Minv[1, 2])
    im = Image.fromarray(image).transform(image.shape[::-1], Image.AFFINE, data, resample=Image.BILINEAR, fillcolor=255)
    return np.asarray(im)


def augment(image: np.ndarray, wedges: np.ndarray, ignore_pts: np.ndarray, rng: np.random.Generator,
            cfg: AugmentConfig = AugmentConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = image.shape[0]
    M = affine_matrix(rng, cfg, size)
    img = apply_affine_image(M, image)
    wedges = apply_affine_points(M, wedges)
    ignore_pts = apply_affine_points(M, ignore_pts)
    u = rng.random()
    if u < cfg.p_thicken:
        img = ndi.grey_erosion(img, size=(2, 2))          # ink is dark: erosion thickens strokes
    elif u < cfg.p_thicken + cfg.p_thin:
        img = ndi.grey_dilation(img, size=(2, 2))
    f = img.astype(np.float32) / 255.0
    if cfg.blur_sigma > 0 and rng.random() < 0.5:
        f = ndi.gaussian_filter(f, rng.uniform(0.2, cfg.blur_sigma))
    f = np.clip(f, 0, 1) ** rng.uniform(*cfg.gamma)
    if cfg.noise > 0:
        f = f + rng.normal(0, cfg.noise * rng.random(), f.shape)
    img = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    keep = np.all((wedges[:, 0] >= 0) & (wedges[:, 0] < size), axis=1) if len(wedges) else np.zeros(0, bool)
    return img, wedges[keep], ignore_pts


def to_input(image: np.ndarray, extra_channels: bool) -> np.ndarray:
    """(C, H, W) float32 with ink = 1. Optional classical channels: clipped distance transform
    of the ink and the thinned skeleton, the same features the Stage 2 baseline runs on."""
    ink_f = 1.0 - image.astype(np.float32) / 255.0
    if not extra_channels:
        return ink_f[None]
    ink = image < 128
    edt = np.clip(ndi.distance_transform_edt(ink) / 4.0, 0, 1).astype(np.float32)
    skel = skeletonize(ink).astype(np.float32)
    return np.stack([ink_f, edt, skel])


def skeleton_arrays(s: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    wedges = np.array([w.as_array() for w in s.wedges], dtype=np.float64).reshape(-1, 4, 2)
    ign = [p for st in s.strokes if st.kind != KIND_WEDGE for p in st.points]
    return wedges, np.array(ign, dtype=np.float64).reshape(-1, 2)


class GlyphDataset:
    """Map-style dataset. Images and labels are held in memory (449 glyphs)."""

    def __init__(self, pairs: list[Pair], skeletons: dict[str, Skeleton], train: bool,
                 extra_channels: bool = False, cfg: AugmentConfig = AugmentConfig(), repeats: int = 1, seed: int = 0):
        self.items = []
        for p in pairs:
            wedges, ign = skeleton_arrays(skeletons[p.key])
            image, tf = prepare_image(p.image_path)          # identity for 256x256 fonts, rescale for Esagil
            self.items.append((p.key, image, tf.to_model(wedges) if len(wedges) else wedges, tf.to_model(ign) if len(ign) else ign))
        self.train, self.extra, self.cfg, self.repeats, self.seed = train, extra_channels, cfg, repeats, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items) * (self.repeats if self.train else 1)

    def __getitem__(self, i: int):
        import torch
        key, img, wedges, ign = self.items[i % len(self.items)]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            img, wedges, ign = augment(img, wedges, ign, rng, self.cfg)
        t = build_targets(wedges, ign, size=img.shape[0])
        out = {k: torch.from_numpy(v) for k, v in t.items()}
        out["image"] = torch.from_numpy(to_input(img, self.extra))
        return out
