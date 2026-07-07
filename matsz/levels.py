"""Progressive stage schedule for the MAT-SZ closed loop.

Stage 0 (anchors): anchor_block x anchor_block pixel blocks whose top-left
corners lie on the anchor_stride grid — coded by direct quantization.
Each dyadic level (stride = anchor_stride / 2^k) is then split into three
antidiagonal sub-stages — horizontal midpoints, vertical midpoints, then the
diagonal centers — so a stage always predicts from the maximum of
already-reconstructed neighbours (the centers see all four orthogonal
midpoints, not two). The last level's diagonal stage absorbs any remainder, so
the masks are disjoint and their union covers the full (h, w) tile. Encoder,
decoder, and the GNN trainer all derive them from the header parameters alone.
"""

from __future__ import annotations

import numpy as np


def stage_plan(
    h: int,
    w: int,
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[tuple[np.ndarray, int, str]]:
    """Ordered sub-stages as (mask, stride, phase); phase in
    {"anchor", "h", "v", "d"}. ``stage_masks`` drops the metadata; the interp
    predictor keeps stride/phase to pick which axis to interpolate."""
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if anchor_stride < 2 or anchor_stride & (anchor_stride - 1):
        raise ValueError("anchor_stride must be a power of two >= 2")
    if not 1 <= anchor_block <= anchor_stride:
        raise ValueError("anchor_block must be in [1, anchor_stride]")

    ih, iw = np.arange(h), np.arange(w)
    covered = np.zeros((h, w), bool)

    anchor = np.zeros((h, w), bool)
    for di in range(anchor_block):
        for dj in range(anchor_block):
            anchor[di::anchor_stride, dj::anchor_stride] = True
    plan: list[tuple[np.ndarray, int, str]] = [(anchor, anchor_stride, "anchor")]
    covered |= anchor

    for k in range(1, levels + 1):
        s = max(anchor_stride >> k, 1)
        coarse_h = (ih % (2 * s)) == 0
        mid_h = ((ih % s) == 0) & ~coarse_h
        coarse_w = (iw % (2 * s)) == 0
        mid_w = ((iw % s) == 0) & ~coarse_w
        m_h = (coarse_h[:, None] & mid_w[None, :]) & ~covered  # horizontal
        m_v = (mid_h[:, None] & coarse_w[None, :]) & ~covered  # vertical
        m_d = (mid_h[:, None] & mid_w[None, :]) & ~covered      # diagonal
        if k == levels:  # last level: absorb any remainder (small tiles)
            m_d |= ~covered & ~m_h & ~m_v
        for mask, phase in ((m_h, "h"), (m_v, "v"), (m_d, "d")):
            plan.append((mask, s, phase))
            covered |= mask

    return plan


def stage_masks(
    h: int,
    w: int,
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[np.ndarray]:
    return [mask for mask, _, _ in stage_plan(h, w, levels, anchor_stride, anchor_block)]
