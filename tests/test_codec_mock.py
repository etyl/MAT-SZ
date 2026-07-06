"""Full-pipeline integration tests with the torch-free MockPredictor."""

import numpy as np
import pytest

from matsz.codec import compress, decompress
from matsz.predictor import MockPredictor


def smooth_image(h, w, c, seed=0):
    """Smooth-ish synthetic image so prediction has something to work with."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([
        128 + 100 * np.sin(xx / (10 + 5 * k)) * np.cos(yy / (13 + 3 * k))
        + rng.randn(h, w) * 2
        for k in range(c)], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("shape,eb", [
    ((130, 70, 3), 0.5),
    ((130, 70, 3), 2.0),
    ((100, 100, 1), 8.0),
    ((64, 64, 3), 2.0),
])
def test_roundtrip_bound(shape, eb):
    h, w, c = shape
    img = smooth_image(h, w, c)
    if c == 1:
        img = img[..., 0]
    stream, stats = compress(img, eb, MockPredictor(64))
    rec = decompress(stream)
    assert rec.shape == img.shape and rec.dtype == img.dtype
    assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
    assert stats["ratio"] > 1.0


def test_deterministic_streams_and_output():
    img = smooth_image(100, 130, 3)
    s1, _ = compress(img, 2.0, MockPredictor(64))
    s2, _ = compress(img, 2.0, MockPredictor(64))
    assert s1 == s2
    r1 = decompress(s1)
    r2 = decompress(s1)
    assert np.array_equal(r1, r2)


def test_encoder_recon_matches_decoder_output():
    img = smooth_image(64, 64, 3, seed=5)
    stream, stats = compress(img, 1.0, MockPredictor(64))
    rec = decompress(stream)
    assert np.array_equal(stats["recon"], rec)


def test_float_input():
    rng = np.random.RandomState(7)
    img = (rng.rand(70, 90).astype(np.float32) * 4).round(2)
    stream, _ = compress(img, 0.01, MockPredictor(64))
    rec = decompress(stream)
    assert rec.dtype == np.float32
    assert np.abs(img - rec).max() <= 0.01


def test_real_predictor_stream_requires_factory():
    img = smooth_image(64, 64, 3)
    stream, _ = compress(img, 2.0, MockPredictor(64))
    # corrupt the mock flag to simulate a real-predictor stream
    from matsz.bitstream import Header, read_stream, write_stream
    hdr, payloads = read_stream(stream)
    hdr.flags &= ~1
    with pytest.raises(ValueError, match="checkpoint"):
        decompress(write_stream(hdr, payloads))
