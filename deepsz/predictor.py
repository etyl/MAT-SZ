"""Predictors: the pluggable stage that replaces SZ's Lorenzo/spline prediction.

Both predictors share the interface
    predict(recon, known, pos=None) -> pred
with recon float32 (C, T, T) in original data units, known bool (T, T), and
pos an optional bool (T, T) mask of the points actually being predicted this
stage. With pos given, pred is the compact float32 (C, pos.sum()) prediction
at just those points; with pos=None, pred covers the whole tile. Predictions
must be a pure function of (recon * known, known) so the decoder can
reproduce them exactly.
"""

from __future__ import annotations

import numpy as np

from .bitstream import FLAG_CUBIC, FLAG_INTERP, FLAG_MOCK
from .levels import stage_plan

_INTERP_BUILD_CACHE: dict[tuple, tuple[list, dict, dict, list]] = {}


class MockPredictor:
    """Nearest-known-pixel fill + box smoothing. Deterministic, torch-free,
    any tile size. Used by fast tests and the --mock CLI flag."""

    stream_flag = FLAG_MOCK

    def __init__(self, tile_size: int = 64):
        self.tile_size = tile_size
        self.checkpoint_hash = b"\0" * 16

    def predict(self, recon: np.ndarray, known: np.ndarray,
               pos: np.ndarray | None = None) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt, uniform_filter

        if not known.any():
            out = np.zeros_like(recon)
            return out if pos is None else out[:, pos]
        _, (ii, jj) = distance_transform_edt(~known, return_indices=True)
        filled = recon[:, ii, jj]
        smooth = uniform_filter(filled, size=(1, 3, 3), mode="nearest")
        # keep exact values at known pixels, smooth only the filled region
        out = np.where(known[None], filled, smooth).astype(np.float32)
        return out if pos is None else out[:, pos]


def default_interp_center(ndim: int) -> int:
    """Best fast-mode ``center`` for a spatial rank. Averaging the per-axis
    interpolation (``center=0``) helps in 2-D/3-D but degrades as rank grows —
    a 4-D cell-centre blends 4 already-quantized axis-estimates and loses to
    SZ3's single-direction scheme. Empirically ``center=1`` (interpolate along
    one axis, like SZ3) wins from 4-D up; ``center=0`` wins at/below 3-D."""
    return 1 if ndim >= 4 else 0


def _interp_axis_at(W, geom, order):
    """SZ3's 1D interpolation at precomputed flattened neighbour indices.

    Cubic weights are
    [-1, 9, 9, -1]/16 over the four nearest same-line samples, dropping to linear
    then to an edge copy where the ±3s / ±s neighbours fall outside the tile.
    Operates only on the M query points (``W`` is flattened (C, N) float64), so it never
    materializes a whole-grid temporary — but is bit-identical to evaluating the
    old full-grid form and slicing these positions out."""
    im1, ip1, im3, ip3, vm1, vp1, vm3, vp3 = geom
    Lm1, Lp1 = W[:, im1], W[:, ip1]
    pred = 0.5 * (Lm1 + Lp1)
    if order == "cubic":
        Lm3, Lp3 = W[:, im3], W[:, ip3]
        cub = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
        pred = np.where((vm3 & vp3)[None], cub, pred)
    both = (vm1 & vp1)[None]
    only_left = (vm1 & ~vp1)[None]
    return np.where(both, pred, np.where(only_left, Lm1, Lp1))     # (C, M)


class InterpPredictor:
    """SZ3-style interpolation baseline dropped into DeepSZ's closed loop, so
    MAT/GNN vs. classical interpolation is isolated to the predictor (identical
    quantizer plus an empirical ANS/zstd stage). Torch- and
    checkpoint-free, so streams decode without a model.

    Each dyadic level is split into codec sub-stages ordered by how many axes a
    point straddles as a midpoint (``levels.stage_plan``): the one-odd-axis
    edge-midpoints first, then the multi-odd-axis centres. Because they are
    separate stages the codec quantizes each into ``recon`` before the next
    reads it — SZ3's interleaved quantize/predict order — so a point's ±stride
    neighbours along each of its odd axes are already reconstructed, and it is
    predicted by averaging the single-axis interpolation over exactly those
    axes (2 neighbours for an edge-midpoint, 4 for a 2-D cell centre). The
    predictor supplies its own ``stage_masks``; the decoder rebuilds the
    identical schedule from the header dims alone.

    Tile-free: SZ3's interpolation has no fixed input size (unlike MAT), so the
    codec runs it over the whole image as a single region — no padding, no
    prediction seam.
    """

    tile_free = True  # codec compresses the whole image as one region
    tunable = True    # encoder sweeps (eb_ratio, center) and keeps the smallest
    fast_eb_ratio = 0.9  # single-encode (tune=fast) default; see codec.encode

    def __init__(self, tile_size: int = 512, order: str = "cubic",
                 levels: int = 4, anchor_stride: int = 16, anchor_block: int = 4,
                 center: int | None = None):
        if order not in ("linear", "cubic"):
            raise ValueError("order must be 'linear' or 'cubic'")
        self.order = order
        self.levels = levels
        self.anchor_stride = anchor_stride
        self.anchor_block = anchor_block
        # multi-odd-axis ("centre") prediction: 0 avg all odd axes, 1/2
        # interpolate along the first/last odd axis only (SZ3's single-direction
        # centre). None = pick by spatial rank (``default_interp_center``): 0 for
        # <=3-D, 1 from 4-D up, where averaging over many axes starts to lose.
        # The codec resolves None to a concrete value once the region rank is
        # known; decode always overrides from the header.
        self.center = center
        self.stream_flag = FLAG_INTERP | (FLAG_CUBIC if order == "cubic" else 0)
        self.checkpoint_hash = b"\0" * 16
        self._cache: dict[tuple[int, ...], tuple[list, dict, dict, list]] = {}
        # The codec reveals one stage at a time.  Keep a float64 mirror and
        # update only the newly reconstructed points instead of converting the
        # complete field on every prediction call.
        self._work: np.ndarray | None = None
        self._stage_idx = -1
        self._last_coords: tuple[np.ndarray, ...] | None = None

    def _build(self, shape: tuple[int, ...]) -> tuple[list, dict, dict, list]:
        """Return (masks, schedule) for a region of the given spatial shape,
        from the shared ``levels.stage_plan`` (so the GNN codec/trainer use the
        identical schedule). ``masks`` is the per-axis split stage list [anchor,
        l1-axis0, l1-axis1, ..., l2-axis0, ...]; ``schedule`` maps |known so far|
        -> (stride, axis) so ``predict`` knows which axis to interpolate along.
        Cached per shape."""
        key = tuple(int(n) for n in shape)
        if key in self._cache:
            return self._cache[key]
        global_key = (key, self.levels, self.anchor_stride, self.anchor_block)
        if global_key in _INTERP_BUILD_CACHE:
            self._cache[key] = _INTERP_BUILD_CACHE[global_key]
            return self._cache[key]
        masks: list = []
        schedule: dict[int, tuple[int, tuple[int, ...]]] = {}
        by_pos: dict[int, tuple] = {}
        indices: list[np.ndarray] = []
        covered = 0
        for stage_idx, (mask, s, axes) in enumerate(stage_plan(
                key, self.levels, self.anchor_stride, self.anchor_block)):
            n = int(mask.sum())
            flat = np.flatnonzero(mask)
            indices.append(flat)
            if axes and n:  # non-anchor sub-stage; predict() is keyed by prior |known|
                # np.nonzero() is surprisingly visible for large fields and is
                # invariant across encode candidates and decode.
                coords = np.unravel_index(flat, key)
                geoms = {}
                for axis in axes:
                    geoms[axis] = _axis_geometry(
                        flat, coords, axis, s, key)
                entry = (stage_idx, s, axes, coords, geoms)
                schedule[covered] = entry
                by_pos[id(mask)] = entry
            masks.append(mask)
            covered += n
        self._cache[key] = (masks, schedule, by_pos, indices)
        if len(_INTERP_BUILD_CACHE) >= 16:
            _INTERP_BUILD_CACHE.pop(next(iter(_INTERP_BUILD_CACHE)))
        _INTERP_BUILD_CACHE[global_key] = self._cache[key]
        return self._cache[key]

    def stage_masks(self, shape, levels, anchor_stride, anchor_block) -> list:
        return self._build(shape)[0]

    def stage_indices(self, shape) -> list[np.ndarray]:
        """C-order flat indices aligned with :meth:`stage_masks`."""
        return self._build(shape)[3]

    def predict(self, recon: np.ndarray, known: np.ndarray,
               pos: np.ndarray | None = None) -> np.ndarray:
        built = self._build(recon.shape[1:])
        entry = built[2].get(id(pos))
        if entry is None:
            entry = built[1].get(int(known.sum()))
        if entry is None:
            raise ValueError("known mask does not match the interp schedule")
        stage_idx, _s, axes, coords, geoms = entry
        shape = recon.shape[1:]
        reset = (self._work is None or self._work.shape != recon.shape
                 or stage_idx <= self._stage_idx)
        if reset:
            self._work = np.zeros(recon.shape, np.float64)
            self._work[:, known] = recon[:, known]
        elif self._last_coords is not None:
            self._work[(slice(None), *self._last_coords)] = recon[
                (slice(None), *self._last_coords)]
        self._stage_idx = stage_idx
        self._last_coords = coords
        W = self._work.reshape(recon.shape[0], -1)
        if pos is None:
            coords = np.indices(shape).reshape(len(shape), -1)
            flat = np.arange(np.prod(shape), dtype=np.int64)
            geoms = {a: _axis_geometry(flat, coords, a, _s, shape)
                     for a in axes}
        # Interpolate each odd axis from its ±stride neighbours (already
        # reconstructed in an earlier, lower-weight sub-stage of the same level):
        # an edge-midpoint (one odd axis) reads 2 priors, a 2-D centre (two odd
        # axes) reads 4. ``center`` picks how a multi-odd-axis point combines
        # them: averaged (0) or single-direction (1/2).
        center = (self.center if self.center is not None
                  else default_interp_center(len(shape)))
        if center == 0 or len(axes) == 1:
            out = sum(_interp_axis_at(W, geoms[a], self.order)
                      for a in axes) / len(axes)
        else:
            a = axes[0] if center == 1 else axes[-1]
            out = _interp_axis_at(W, geoms[a], self.order)
        out = out.astype(np.float32)                              # (C, M)
        return out if pos is not None else out.reshape(recon.shape)


def _axis_geometry(flat, coords, axis, stride, shape):
    """Flattened neighbour indices and boundary-validity masks for one axis."""
    ca = coords[axis]
    axis_step = int(np.prod(shape[axis + 1:], dtype=np.int64))

    def index(off):
        shifted = np.clip(ca + off, 0, shape[axis] - 1)
        return flat + (shifted - ca) * axis_step

    return (
        index(-stride), index(stride), index(-3 * stride), index(3 * stride),
        ca - stride >= 0, ca + stride < shape[axis],
        ca - 3 * stride >= 0, ca + 3 * stride < shape[axis],
    )
