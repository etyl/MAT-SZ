import numpy as np
import pytest

from matsz.levels import stage_masks


# all levels == log2(stride) so the schedule densifies to stride 1 (guard)
@pytest.mark.parametrize("shape,levels,stride,block", [
    ((64, 64), 3, 8, 1),
    ((64, 64), 3, 8, 4),
    ((512, 512), 3, 8, 4),
    ((512, 512), 4, 16, 8),
    ((64, 48), 2, 4, 1),
    ((33, 65), 4, 16, 2),
    ((16, 16, 16), 3, 8, 1),   # 3-D: schedule is dimension-agnostic
    ((24,), 3, 8, 1),          # 1-D
])
def test_partition(shape, levels, stride, block):
    masks = stage_masks(shape, levels, stride, block)
    ndim = len(shape)
    # anchor + (2^ndim - 1) sub-stages per level (one per non-empty odd-axis set)
    assert len(masks) == 1 + (2 ** ndim - 1) * levels
    total = np.zeros(shape, int)
    for m in masks:
        assert m.shape == shape
        total += m.astype(int)
    assert (total == 1).all()  # disjoint and exhaustive


def test_anchor_geometry():
    masks = stage_masks((32, 32), 3, 8, 2)  # levels==log2(stride)
    anchor = masks[0]
    assert anchor[0, 0] and anchor[0, 1] and anchor[1, 0] and anchor[1, 1]
    assert not anchor[0, 2] and not anchor[2, 2]
    assert anchor[8, 8] and anchor[9, 9]


def test_deterministic():
    a = stage_masks((64, 64), 3, 8, 4)
    b = stage_masks((64, 64), 3, 8, 4)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_validation():
    with pytest.raises(ValueError):
        stage_masks((64, 64), 0, 8)
    with pytest.raises(ValueError):
        stage_masks((64, 64), 3, 7)
    with pytest.raises(ValueError):
        stage_masks((64, 64), 3, 8, 9)
    with pytest.raises(ValueError):  # levels < log2(stride): can't reach stride 1
        stage_masks((64, 64), 2, 16)
