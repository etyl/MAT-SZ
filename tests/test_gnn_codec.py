"""Public GNN tensor codec API tests."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepsz import GNNCompressorCodec
from deepsz.gnn_predictor import CKPT_VERSION, build_model


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_model(d=8).eval()
    path = tmp_path / "gnn.pt"
    torch.save({
        "d": model.d,
        "agg_level": 2,
        "state_dict": model.state_dict(),
        "version": CKPT_VERSION,
    }, path)
    return path


def _codec(path, eb=1e-3):
    return GNNCompressorCodec(
        path,
        error_bound=eb,
        levels=2,
        anchor_stride=4,
        anchor_block=1,
        max_radius=4,
        chunk_size=0,
        fp16=False,
        compile=False,
    )


def test_defaults_match_eval_tensor(tiny_checkpoint):
    codec = GNNCompressorCodec(tiny_checkpoint)

    assert codec.error_bound == 0.01
    assert codec.levels == 5
    assert codec.anchor_stride == 32
    assert codec.anchor_block == 1
    assert codec.chunk_size is None
    assert codec.chunk_batch is None
    assert codec.fp16 is False
    assert codec.compile is True


def test_numpy_nd_tensor_roundtrip(tiny_checkpoint):
    rng = np.random.RandomState(0)
    x = rng.rand(5, 6, 4).astype(np.float32)
    codec = _codec(tiny_checkpoint, eb=0.01)

    stream = codec.compress(x)
    y = codec.uncompress(stream)

    assert isinstance(stream, bytes)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == x.shape
    assert y.dtype == torch.float32
    assert torch.max(torch.abs(y - torch.from_numpy(x))) <= 0.01


def test_torch_tensor_input_roundtrip(tiny_checkpoint):
    x = torch.linspace(-1.0, 1.0, 35, dtype=torch.float32).reshape(7, 5)
    codec = _codec(tiny_checkpoint, eb=0.005)

    y = codec.uncompress(codec.compress(x))

    assert tuple(y.shape) == tuple(x.shape)
    assert y.dtype == x.dtype
    assert torch.max(torch.abs(y - x)) <= 0.005


def test_scalar_shape_roundtrip(tiny_checkpoint):
    x = np.asarray(3.25, dtype=np.float32)
    codec = _codec(tiny_checkpoint, eb=0.001)

    y = codec.uncompress(codec.compress(x))

    assert tuple(y.shape) == ()
    assert y.dtype == torch.float32
    assert torch.abs(y - torch.tensor(3.25)) <= 0.001
