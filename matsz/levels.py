"""Progressive stage schedule for the MAT-SZ closed loop.

Stage 0 (anchors): ``anchor_block^ndim`` pixel blocks whose corners lie on the
``anchor_stride`` grid — coded by direct quantization. Each dyadic level
(stride = anchor_stride / 2^k) is then densified one axis at a time, but
*sequentially by how many axes a point is a midpoint on*: a point whose
coordinate is an odd multiple of the level stride on the axes in some set ``A``
(and on the coarse grid elsewhere) is filled only after every point with fewer
odd axes. So its ±stride neighbours along each axis in ``A`` have already been
decoded, and it is interpolated along each of those axes from them — the more
axes a point straddles, the more reconstructed neighbours it sees. In 2-D this
recovers the classic order: horizontal/vertical edge-midpoints (one odd axis,
two coarse neighbours) then the cell centre (two odd axes, four neighbours).

The rule is dimension-agnostic — the same weight ordering drives 2-D images and
n-D grids (matching the rank-generic GNN predictor), yielding ``2^ndim - 1``
sub-stages per level — and the disjoint masks union to the full grid. Encoder,
decoder, and the GNN trainer all derive it from the header parameters alone.
"""

from __future__ import annotations

import itertools

import numpy as np


def _axis_mask(mask1d: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    """Broadcast a per-index 1-D selector along ``axis`` of an ndim grid."""
    shape = [1] * ndim
    shape[axis] = mask1d.shape[0]
    return mask1d.reshape(shape)


def stage_plan(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[tuple[np.ndarray, int, tuple[int, ...]]]:
    """Ordered sub-stages as (mask, stride, axes) for a grid of arbitrary rank.
    ``axes`` is the tuple of axes on which the sub-stage's points are midpoints
    (an odd multiple of ``stride``); the interp predictor averages the
    single-axis interpolation over exactly those axes. The anchor stage has
    ``axes == ()``. ``stage_masks`` drops the metadata."""
    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    if ndim < 1:
        raise ValueError("shape must have at least one axis")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if anchor_stride < 2 or anchor_stride & (anchor_stride - 1):
        raise ValueError("anchor_stride must be a power of two >= 2")
    if not 1 <= anchor_block <= anchor_stride:
        raise ValueError("anchor_block must be in [1, anchor_stride]")

    coords = [np.arange(n) for n in shape]
    covered = np.zeros(shape, bool)

    anchor = np.zeros(shape, bool)
    for offs in itertools.product(range(anchor_block), repeat=ndim):
        anchor[tuple(slice(o, None, anchor_stride) for o in offs)] = True
    plan: list[tuple[np.ndarray, int, tuple[int, ...]]] = [(anchor, anchor_stride, ())]
    covered |= anchor

    for k in range(1, levels + 1):
        s = max(anchor_stride >> k, 1)
        # weight w = number of axes a point is a midpoint on; low -> high so a
        # point's ±s neighbours along its odd axes are already decoded.
        for w in range(1, ndim + 1):
            for axes in itertools.combinations(range(ndim), w):
                mask = np.ones(shape, bool)
                for j, cj in enumerate(coords):
                    if j in axes:       # midpoint (odd multiple of s) on this axis
                        sel = ((cj % s) == 0) & ((cj % (2 * s)) != 0)
                    else:               # on the coarse (stride 2s) grid
                        sel = (cj % (2 * s)) == 0
                    mask &= _axis_mask(sel, j, ndim)
                mask &= ~covered
                if k == levels and w == ndim:  # final sub-stage absorbs remainder
                    mask |= ~covered
                plan.append((mask, s, axes))
                covered |= mask

    return plan


def stage_masks(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[np.ndarray]:
    return [mask for mask, _, _ in stage_plan(shape, levels, anchor_stride, anchor_block)]
