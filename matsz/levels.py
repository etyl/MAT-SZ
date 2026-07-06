"""Progressive stage schedule for the MAT-SZ closed loop.

Stage 0 (anchors): anchor_block x anchor_block pixel blocks whose top-left
corners lie on the anchor_stride grid — coded by direct quantization.
Stages 1..levels: grid stride halves each stage (anchor_stride / 2^k), taking
only positions not covered by earlier stages; the last stage (stride 1) covers
every remaining pixel. The returned masks are disjoint and their union covers
the full (h, w) tile — encoder and decoder derive them independently from the
header parameters alone.
"""

from __future__ import annotations

import numpy as np


def stage_masks(
    h: int,
    w: int,
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[np.ndarray]:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if anchor_stride < 2 or anchor_stride & (anchor_stride - 1):
        raise ValueError("anchor_stride must be a power of two >= 2")
    if not 1 <= anchor_block <= anchor_stride:
        raise ValueError("anchor_block must be in [1, anchor_stride]")

    covered = np.zeros((h, w), bool)
    masks: list[np.ndarray] = []

    anchor = np.zeros((h, w), bool)
    for di in range(anchor_block):
        for dj in range(anchor_block):
            anchor[di::anchor_stride, dj::anchor_stride] = True
    masks.append(anchor)
    covered |= anchor

    for k in range(1, levels + 1):
        stride = max(anchor_stride >> k, 1)
        grid = np.zeros((h, w), bool)
        grid[::stride, ::stride] = True
        if k == levels:
            grid[:] = True  # final stage covers everything left
        mask = grid & ~covered
        masks.append(mask)
        covered |= mask

    return masks
