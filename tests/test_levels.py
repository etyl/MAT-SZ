import numpy as np
import pytest

from matsz.levels import stage_masks


@pytest.mark.parametrize("h,w,levels,stride,block", [
    (64, 64, 3, 8, 1),
    (64, 64, 3, 8, 4),
    (512, 512, 3, 8, 4),
    (512, 512, 2, 16, 8),
    (64, 48, 1, 4, 1),
    (33, 65, 4, 16, 2),
])
def test_partition(h, w, levels, stride, block):
    masks = stage_masks(h, w, levels, stride, block)
    assert len(masks) == levels + 1
    total = np.zeros((h, w), int)
    for m in masks:
        total += m.astype(int)
    assert (total == 1).all()  # disjoint and exhaustive


def test_anchor_geometry():
    masks = stage_masks(32, 32, 2, 8, 2)
    anchor = masks[0]
    assert anchor[0, 0] and anchor[0, 1] and anchor[1, 0] and anchor[1, 1]
    assert not anchor[0, 2] and not anchor[2, 2]
    assert anchor[8, 8] and anchor[9, 9]


def test_deterministic():
    a = stage_masks(64, 64, 3, 8, 4)
    b = stage_masks(64, 64, 3, 8, 4)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_validation():
    with pytest.raises(ValueError):
        stage_masks(64, 64, 0, 8)
    with pytest.raises(ValueError):
        stage_masks(64, 64, 3, 7)
    with pytest.raises(ValueError):
        stage_masks(64, 64, 3, 8, 9)
