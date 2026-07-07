"""Predictors: the pluggable stage that replaces SZ's Lorenzo/spline prediction.

Both predictors share the interface
    predict(recon, known) -> pred
with recon float32 (C, T, T) in original data units, known bool (T, T);
returns float32 (C, T, T) predictions for the whole tile (only hole positions
are consumed by the codec). Predictions must be a pure function of
(recon * known, known) so the decoder can reproduce them exactly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .bitstream import FLAG_MOCK


class MockPredictor:
    """Nearest-known-pixel fill + box smoothing. Deterministic, torch-free,
    any tile size. Used by fast tests and the --mock CLI flag."""

    stream_flag = FLAG_MOCK

    def __init__(self, tile_size: int = 64):
        self.tile_size = tile_size
        self.checkpoint_hash = b"\0" * 16

    def predict(self, recon: np.ndarray, known: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt, uniform_filter

        if not known.any():
            return np.zeros_like(recon)
        _, (ii, jj) = distance_transform_edt(~known, return_indices=True)
        filled = recon[:, ii, jj]
        smooth = uniform_filter(filled, size=(1, 3, 3), mode="nearest")
        # keep exact values at known pixels, smooth only the filled region
        return np.where(known[None], filled, smooth).astype(np.float32)


class MATPredictor:
    """MAT (Mask-Aware Transformer, CVPR 2022) inpainting as the SZ predictor.

    Determinism requirements (decoder must bitwise-reproduce encoder
    predictions on the same platform):
      - model.z is drawn unseeded in MAT.__init__ -> overwritten here with an
        RNG seeded from the header seed;
      - the synthesis network calls F.dropout(training=True) at inference ->
        torch.manual_seed(seed) immediately before every forward.
    """

    tile_size = 512

    def __init__(self, checkpoint_path: str | Path, seed: int,
                 vmin: float, vmax: float):
        import spandrel
        import spandrel_extra_arches
        import torch

        self._torch = torch
        self.seed = int(seed)
        self.vmin = float(vmin)
        self.vmax = float(vmax)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        spandrel_extra_arches.install(ignore_duplicates=True)
        desc = spandrel.ModelLoader(device=self.device).load_from_file(str(checkpoint_path))
        if desc.purpose != "Inpainting":
            raise ValueError(f"checkpoint is not an inpainting model: {desc}")
        self.model = desc.model.float().eval()
        self.model.z = torch.from_numpy(
            np.random.RandomState(self.seed).randn(1, 512)).float().to(self.device)

        self.checkpoint_hash = hashlib.sha256(
            Path(checkpoint_path).read_bytes()).digest()[:16]

    def predict(self, recon: np.ndarray, known: np.ndarray) -> np.ndarray:
        torch = self._torch
        c, h, w = recon.shape
        if (h, w) != (self.tile_size, self.tile_size):
            raise ValueError(f"MAT requires {self.tile_size}x{self.tile_size} tiles, got {h}x{w}")

        span = self.vmax - self.vmin
        norm = (np.clip(recon, self.vmin, self.vmax) - self.vmin) / span
        norm = np.where(known[None], norm, 0.5).astype(np.float32)
        if c == 1:
            norm = np.repeat(norm, 3, axis=0)

        x = torch.from_numpy(norm[None]).to(self.device)
        m = torch.from_numpy((~known).astype(np.float32)[None, None]).to(self.device)
        torch.manual_seed(self.seed)
        with torch.inference_mode():
            y = self.model(x, m)
        pred = y[0].cpu().numpy()
        if c == 1:
            pred = pred.mean(axis=0, keepdims=True)
        pred = pred * span + self.vmin
        return np.clip(pred, self.vmin, self.vmax).astype(np.float32)
