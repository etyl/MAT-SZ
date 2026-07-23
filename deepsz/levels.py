"""Progressive stage schedule for the DeepSZ closed loop.

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
    # The dyadic levels must reach stride 1, i.e. levels >= log2(anchor_stride);
    # otherwise the finest pixels never become midpoints and get dumped into the
    # remainder clause below as one huge, badly-predicted stage (60-90% bloat).
    if (1 << levels) < anchor_stride:
        raise ValueError(f"levels={levels} too small for anchor_stride="
                         f"{anchor_stride}: need levels >= log2(anchor_stride) = "
                         f"{anchor_stride.bit_length() - 1} to densify to stride 1")

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


def stage_strides(ndim: int, levels: int, anchor_stride: int) -> list[int]:
    """Per-stage lattice stride, aligned with ``stage_plan`` order, computed in
    closed form from ``(ndim, levels, anchor_stride)`` alone — no masks.

    The stride sequence is a pure function of the schedule shape: stage 0 (the
    anchors) has stride ``anchor_stride``; each dyadic level ``k`` contributes
    ``2^ndim - 1`` sub-stages all at stride ``max(anchor_stride >> k, 1)``. It is
    independent of the grid extent, so this reproduces ``[stride for _, stride,
    _ in stage_plan(shape, ...)]`` for any ``shape`` of rank ``ndim`` without
    materialising a single stage mask (the mask build is O(levels * n_points),
    catastrophic on the large representative grids ``stage_ebs`` is handed in
    high dimensions)."""
    if ndim < 1:
        raise ValueError("shape must have at least one axis")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if anchor_stride < 2 or anchor_stride & (anchor_stride - 1):
        raise ValueError("anchor_stride must be a power of two >= 2")
    if (1 << levels) < anchor_stride:
        raise ValueError(f"levels={levels} too small for anchor_stride="
                         f"{anchor_stride}: need levels >= log2(anchor_stride) = "
                         f"{anchor_stride.bit_length() - 1} to densify to stride 1")
    per_level = (1 << ndim) - 1
    strides = [anchor_stride]
    for k in range(1, levels + 1):
        strides += [max(anchor_stride >> k, 1)] * per_level
    return strides


def point_levels(
    coords: "list[np.ndarray] | tuple[np.ndarray, ...]",
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> np.ndarray:
    """Dyadic level at which each point is revealed, from coordinate residues.

    ``coords`` is one integer array per axis (equal shapes, broadcast not
    required). Level 0 = anchor pattern (every coordinate ``% anchor_stride <
    anchor_block``); otherwise the smallest ``k >= 1`` whose lattice
    ``stride >> k`` contains the point on every axis. Points off every dyadic
    lattice (the schedule's remainder clause) land on the finest level. This is
    exactly the level of the ``stage_plan`` stage that reveals the point, and
    the chunked codec uses it to pick a halo point's per-level coarse
    embedding without materialising any full-shape stage mask."""
    coords = [np.asarray(c, np.int64) for c in coords]
    out = np.full(coords[0].shape, levels, np.int8)
    anchor = np.ones(coords[0].shape, bool)
    for c in coords:
        anchor &= (c % anchor_stride) < anchor_block
    for k in range(levels - 1, 0, -1):  # coarse levels overwrite finer ones
        s = max(anchor_stride >> k, 1)
        on = np.ones(coords[0].shape, bool)
        for c in coords:
            on &= (c % s) == 0
        out[on] = k
    out[anchor] = 0
    return out


def stage_ebs(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int,
    eb: float,
    eb_ratio: float,
) -> list[float]:
    """Per-stage absolute error bound, aligned with ``stage_plan`` order.

    Coarser (larger-stride) levels get a tighter bound ``eb * eb_ratio**depth``,
    ``depth`` = log2(stride / finest stride), so their quantization error
    propagates less into the finer levels interpolated from them (QoZ-style
    level-wise error budgeting). The finest level keeps the full ``eb``, so the
    global ``|x - recon| <= eb`` bound still holds unconditionally. ``eb_ratio``
    1.0 -> flat ``eb`` everywhere (classic SZ).

    Depends on ``shape`` only through its rank: the per-stage strides are
    closed-form (see ``stage_strides``), so no stage masks are built. This keeps
    the call cheap even on the large same-rank representative grids the chunked
    codec evaluates it on (a 4-D ``(2*stride)^4`` grid is ~17M points — building
    its masks cost seconds per compress)."""
    strides = stage_strides(len(shape), levels, anchor_stride)
    if not 1 <= anchor_block <= anchor_stride:
        raise ValueError("anchor_block must be in [1, anchor_stride]")
    finest = min(strides)
    return [eb * eb_ratio ** np.log2(stride / finest) for stride in strides]
