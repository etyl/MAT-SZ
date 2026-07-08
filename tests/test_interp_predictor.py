"""SZ-style interpolation baseline: bound holds and streams self-decode."""

import numpy as np
import pytest

from matsz.codec import compress, decompress
from matsz.predictor import InterpPredictor
from tests.test_codec_mock import smooth_image


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


def test_autotune_matches_or_beats_flat():
    """Auto-tuning (eb_ratio=None) never loses to flat eb on size, since flat
    (1.0) is one of the candidates it sweeps."""
    img = smooth_image(96, 96, 1)[..., 0]
    eb = 3.0
    kw = dict(levels=4, anchor_stride=8, anchor_block=4)
    tuned, _ = compress(img, eb, InterpPredictor(64, "cubic", **kw), **kw)
    flat, _ = compress(img, eb, InterpPredictor(64, "cubic", **kw), eb_ratio=1.0, **kw)
    assert len(tuned) <= len(flat)
