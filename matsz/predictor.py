"""Predictors: the pluggable stage that replaces SZ's Lorenzo/spline prediction.

Both predictors share the interface
    predict(recon, known) -> pred
with recon float32 (C, T, T) in original data units, known bool (T, T);
returns float32 (C, T, T) predictions for the whole tile (only hole positions
are consumed by the codec). Predictions must be a pure function of
(recon * known, known) so the decoder can reproduce them exactly.
"""

from __future__ import annotations

import numpy as np

from .bitstream import FLAG_CUBIC, FLAG_INTERP, FLAG_MOCK, FLAG_NOTILE


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


def _bcast(mask1d: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = mask1d.shape[0]
    return mask1d.reshape(shape)


def _interp_axis(V: np.ndarray, axis: int, s: int, order: str) -> np.ndarray:
    """Predict the odd-stride midpoints along ``axis`` from the even-stride
    (known) samples: SZ3's 1D interpolation — cubic weights [-1, 9, 9, -1]/16
    over the four nearest same-line samples, dropping to linear then to an edge
    copy where the ±3s / ±s neighbours fall outside the tile."""
    n = V.shape[axis]

    def gather(off):
        j = np.arange(n) + off
        return np.take(V, np.clip(j, 0, n - 1), axis=axis), (j >= 0) & (j < n)

    Lm1, vm1 = gather(-s)
    Lp1, vp1 = gather(+s)
    pred = 0.5 * (Lm1 + Lp1)
    if order == "cubic":
        Lm3, vm3 = gather(-3 * s)
        Lp3, vp3 = gather(+3 * s)
        cub = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
        pred = np.where(_bcast(vm3 & vp3, axis, V.ndim), cub, pred)
    both = _bcast(vm1 & vp1, axis, V.ndim)
    only_left = _bcast(vm1 & ~vp1, axis, V.ndim)
    return np.where(both, pred, np.where(only_left, Lm1, Lp1))


class InterpPredictor:
    """SZ3-style interpolation baseline dropped into MAT-SZ's closed loop, so
    MAT/GNN vs. classical interpolation is isolated to the predictor (identical
    quantizer + Huffman/zstd stage, matching SZ3's own pipeline). Torch- and
    checkpoint-free, so streams decode without a model.

    Each dyadic level is split into three codec sub-stages, run antidiagonally
    so every stage predicts from the maximum of already-reconstructed priors:
    horizontal midpoints (interpolate along x), then vertical midpoints (along
    y), then the diagonal centers. Because they are separate stages the codec
    quantizes each into ``recon`` before the next reads it — SZ3's interleaved
    quantize/predict order. By the time the centers run, both their vertical
    neighbours (h-midpoints) and horizontal neighbours (v-midpoints) are
    reconstructed, so a center predicts from 4 reconstructed priors (averaged
    x/y interpolation) instead of the 2 it would see in a single y-pass. The
    predictor supplies its own ``stage_masks``; the decoder rebuilds the
    identical schedule from the header dims alone.

    Tile-free: SZ3's interpolation has no fixed input size (unlike MAT), so the
    codec runs it over the whole image as a single region — no padding, no
    prediction seam.
    """

    tile_free = True  # codec compresses the whole image as one region

    def __init__(self, tile_size: int = 512, order: str = "cubic",
                 levels: int = 4, anchor_stride: int = 16, anchor_block: int = 4):
        if order not in ("linear", "cubic"):
            raise ValueError("order must be 'linear' or 'cubic'")
        self.order = order
        self.levels = levels
        self.anchor_stride = anchor_stride
        self.anchor_block = anchor_block
        self.stream_flag = FLAG_INTERP | (FLAG_CUBIC if order == "cubic" else 0)
        self.checkpoint_hash = b"\0" * 16
        self._cache: dict[tuple[int, int], tuple[list, dict]] = {}

    def _build(self, h: int, w: int) -> tuple[list, dict]:
        """Return (masks, schedule) for an (h, w) region. ``masks`` is the split
        stage list [anchor, lvl1-h, lvl1-v, lvl1-d, lvl2-h, ...]; ``schedule``
        maps |known so far| -> (stride, phase) so ``predict`` knows which
        sub-pass to run. Cached per shape (the codec reuses one region)."""
        key = (h, w)
        if key in self._cache:
            return self._cache[key]
        ih, iw = np.arange(h), np.arange(w)
        covered = np.zeros((h, w), bool)
        anchor = np.zeros((h, w), bool)
        for di in range(self.anchor_block):
            for dj in range(self.anchor_block):
                anchor[di::self.anchor_stride, dj::self.anchor_stride] = True
        masks = [anchor]
        covered |= anchor
        schedule: dict[int, tuple[int, str]] = {}
        for k in range(1, self.levels + 1):
            s = max(self.anchor_stride >> k, 1)
            coarse_h = (ih % (2 * s)) == 0
            mid_h = ((ih % s) == 0) & ~coarse_h
            coarse_w = (iw % (2 * s)) == 0
            mid_w = ((iw % s) == 0) & ~coarse_w
            m_h = (coarse_h[:, None] & mid_w[None, :]) & ~covered    # horizontal
            m_v = (mid_h[:, None] & coarse_w[None, :]) & ~covered    # vertical
            m_d = (mid_h[:, None] & mid_w[None, :]) & ~covered        # diagonal
            if k == self.levels:  # last level: absorb any remainder (small tiles)
                m_d |= ~covered & ~m_h & ~m_v
            for mask, phase in ((m_h, "h"), (m_v, "v"), (m_d, "d")):
                if mask.any():
                    schedule[int(covered.sum())] = (s, phase)
                masks.append(mask)
                covered |= mask
        self._cache[key] = (masks, schedule)
        return self._cache[key]

    def stage_masks(self, h, w, levels, anchor_stride, anchor_block) -> list:
        return self._build(h, w)[0]

    def predict(self, recon: np.ndarray, known: np.ndarray) -> np.ndarray:
        _, h, w = recon.shape
        entry = self._build(h, w)[1].get(int(known.sum()))
        if entry is None:
            raise ValueError("known mask does not match the interp schedule")
        s, phase = entry
        W = recon.astype(np.float64)
        # 'h': horizontal midpoints from coarse columns (along x, axis 2).
        # 'v': vertical midpoints from coarse rows (along y, axis 1).
        # 'd': diagonal centers — average the x and y interpolations; both axes
        # read midpoints already reconstructed this level (4 priors, not 2).
        if phase == "d":
            out = 0.5 * (_interp_axis(W, 1, s, self.order)
                         + _interp_axis(W, 2, s, self.order))
        else:
            out = _interp_axis(W, 2 if phase == "h" else 1, s, self.order)
        return out.astype(np.float32)

