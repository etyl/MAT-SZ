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
