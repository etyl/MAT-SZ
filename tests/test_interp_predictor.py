"""SZ-style interpolation baseline: bound holds and streams self-decode."""

import numpy as np
import pytest

from deepsz.codec import compress, decompress
from deepsz.levels import stage_plan
from deepsz.predictor import InterpPredictor
from tests.test_codec_mock import smooth_image


def _interp_axis_full(V, axis, s, order):
    """Pre-refactor whole-grid 1-D interpolation (reference for the compact
    per-point gather now in InterpPredictor)."""
    n = V.shape[axis]

    def gather(off):
        j = np.arange(n) + off
        return np.take(V, np.clip(j, 0, n - 1), axis=axis), (j >= 0) & (j < n)

    def bc(m):
        sh = [1] * V.ndim; sh[axis] = m.shape[0]; return m.reshape(sh)

    Lm1, vm1 = gather(-s)
    Lp1, vp1 = gather(+s)
    pred = 0.5 * (Lm1 + Lp1)
    if order == "cubic":
        Lm3, vm3 = gather(-3 * s)
        Lp3, vp3 = gather(+3 * s)
        cub = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
        pred = np.where(bc(vm3 & vp3), cub, pred)
    return np.where(bc(vm1 & vp1), pred, np.where(bc(vm1 & ~vp1), Lm1, Lp1))


@pytest.mark.parametrize("order", ["linear", "cubic"])
@pytest.mark.parametrize("center", [0, 1, 2])
def test_compact_gather_matches_full_grid(order, center):
    """The compact per-point predict is bit-identical to evaluating the old
    whole-grid interpolation and slicing the stage's points out — for every
    sub-stage of a non-multiple-of-stride, multi-channel region."""
    shape, levels, stride, block = (70, 90), 3, 8, 1
    recon = (np.random.RandomState(0).rand(2, *shape).astype(np.float32) * 50)
    pred = InterpPredictor(64, order, levels, stride, block, center=center)
    known = np.zeros(shape, bool)
    for mask, s, axes in stage_plan(shape, levels, stride, block):
        if axes and mask.any():
            got = pred.predict(recon, known, mask)
            W = recon.astype(np.float64)
            if center == 0 or len(axes) == 1:
                ref = sum(_interp_axis_full(W, a + 1, s, order) for a in axes) / len(axes)
            else:
                a = axes[0] if center == 1 else axes[-1]
                ref = _interp_axis_full(W, a + 1, s, order)
            assert np.array_equal(got, ref.astype(np.float32)[:, mask])
        known |= mask


@pytest.mark.parametrize("order", ["linear", "cubic"])
@pytest.mark.parametrize("shape,eb", [((70, 90, 3), 2.0), ((80, 80, 1), 4.0)])
def test_roundtrip_bound(order, shape, eb):
    h, w, c = shape
    img = smooth_image(h, w, c)
    if c == 1:
        img = img[..., 0]
    # predictor schedule must match compress()'s levels/anchor_stride/block
    predictor = InterpPredictor(64, order, levels=4, anchor_stride=8, anchor_block=4)
    stream, stats = compress(img, eb, predictor, levels=4, anchor_stride=8,
                             anchor_block=4)
    rec = decompress(stream)  # no factory: torch-free, decodes from flags alone
    assert rec.shape == img.shape and rec.dtype == img.dtype
    assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
    assert np.array_equal(stats["recon"], rec)  # encoder recon == decoder output


@pytest.mark.parametrize("eb_ratio", [1.0, 0.8, 0.5])
@pytest.mark.parametrize("center", [0, 1, 2])
def test_per_level_eb_and_center(eb_ratio, center):
    """Fixed per-level eb decay + any centre mode: the finest level keeps the
    full eb, so the global bound holds, and the tuned params round-trip via the
    header (encoder recon == decoder output)."""
    img = smooth_image(80, 96, 1)[..., 0]
    eb = 3.0
    pred = InterpPredictor(64, "cubic", levels=4, anchor_stride=8,
                           anchor_block=4, center=center)
    stream, stats = compress(img, eb, pred, levels=4, anchor_stride=8,
                             anchor_block=4, eb_ratio=eb_ratio)
    rec = decompress(stream)
    assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
    assert np.array_equal(stats["recon"], rec)


def test_autotune_improves_distortion_with_bounded_size_slack():
    """Auto-tuning may spend a small bounded size slack to reduce propagated
    coarse-level quantization error."""
    img = smooth_image(96, 96, 1)[..., 0]
    eb = 3.0
    kw = dict(levels=4, anchor_stride=8, anchor_block=4)
    tuned, tuned_stats = compress(img, eb, InterpPredictor(64, "cubic", **kw),
                                  tune="rd", **kw)
    flat, flat_stats = compress(img, eb, InterpPredictor(64, "cubic", **kw),
                                eb_ratio=1.0, **kw)
    assert len(tuned) <= 1.05 * len(flat)
    assert tuned_stats["recon_sse"] <= flat_stats["recon_sse"]


def test_fast_tune_is_single_flat_candidate():
    img = smooth_image(96, 96, 1)[..., 0]
    kw = dict(levels=4, anchor_stride=8, anchor_block=4)
    _, stats = compress(img, 3.0, InterpPredictor(64, "cubic", **kw),
                        tune="fast", **kw)
    assert stats["eb_ratio"] == 1.0
    assert stats["interp_center"] == 0
