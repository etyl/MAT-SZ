"""End-to-end test with the real MAT checkpoint. Slow (CPU-only: ~1 min per
512x512 MAT forward). Skipped automatically when the checkpoint is absent.

Run with: pytest -m slow tests/test_e2e_real.py
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "MAT_Places512_G_fp16.safetensors"
IMG = ROOT / "tests" / "data" / "kodim23.png"

pytestmark = pytest.mark.slow

needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not downloaded")


@needs_ckpt
def test_real_roundtrip():
    from PIL import Image

    from matsz.codec import compress, decompress
    from matsz.predictor import MATPredictor

    img = np.asarray(Image.open(IMG).convert("RGB"))[:512, :512]
    eb = 2.0
    seed = 1234
    predictor = MATPredictor(CKPT, seed, float(img.min()), float(img.max()))

    stream1, stats = compress(img, eb, predictor, levels=2, seed=seed)
    stream2, _ = compress(img, eb, predictor, levels=2, seed=seed)
    assert stream1 == stream2, "compression is not deterministic"

    # fresh predictor instance = decoder side
    rec = decompress(
        stream1,
        lambda hdr: MATPredictor(CKPT, hdr.seed, hdr.vmin, hdr.vmax))
    assert rec.shape == img.shape
    assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
    assert np.array_equal(stats["recon"], rec), \
        "decoder reconstruction differs from encoder simulation"
    assert stats["ratio"] > 1.0
