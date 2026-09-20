"""Predictor wrapper so the learned model plugs into the Stage 1 harness."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ..data import Pair, Wedge
from .data import to_input
from .decode import DecodeConfig, decode
from .model import build_model


class LearnedPredictor:
    def __init__(self, model: torch.nn.Module, extra_channels: bool, device: str = "cuda",
                 cfg: DecodeConfig = DecodeConfig(), name: str = "learned"):
        self.model, self.extra, self.device, self.cfg, self.name = model.to(device).eval(), extra_channels, device, cfg, name
        self._cache: dict[str, dict] = {}
        self.last_scores: list[float] = []

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "LearnedPredictor":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device, weights_only=False)
        a = ck["args"]
        model = build_model(a["encoder"], 3 if a["extra_channels"] else 1, a.get("base", 32))
        model.load_state_dict(ck["model"])
        cfg = DecodeConfig(**ck.get("decode", {}))
        return cls(model, a["extra_channels"], device, cfg, name=a.get("name", "learned"))

    def with_decode(self, **kw) -> "LearnedPredictor":
        p = LearnedPredictor(self.model, self.extra, self.device, replace(self.cfg, **kw), self.name)
        p._cache = self._cache
        return p

    @torch.no_grad()
    def raw(self, image: np.ndarray, pair: Pair | None) -> dict:
        key = pair.key if pair is not None else None
        if key is not None and key in self._cache:
            return self._cache[key]
        x = torch.from_numpy(to_input(image, self.extra))[None].to(self.device)
        with torch.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
            out = self.model(x)
        out = {k: v[0].float().cpu() for k, v in out.items()}
        if key is not None:
            self._cache[key] = out
        return out

    def clear_cache(self) -> None:
        self._cache.clear()

    def predict(self, image: np.ndarray, pair: Pair) -> list[Wedge]:
        wedges, self.last_scores = decode(self.raw(image, pair), self.cfg)
        return wedges
