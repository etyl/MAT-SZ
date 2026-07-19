"""Public GNN tensor codec API tests."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepsz import GNNCompressorCodec
from deepsz.bitstream import read_stream as read_generic_stream
from deepsz.codec import compress, decompress
from deepsz.gnn_codec import _read_stream
from deepsz.gnn_predictor import CKPT_VERSION, GNNPredictor, build_model


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_model(d=8).eval()
    path = tmp_path / "gnn.pt"
    torch.save({
        "d": model.d,
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
    assert codec.agg_level == 2
    assert codec.prune_invalid_lines is False
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


@pytest.mark.parametrize("chunk_size", [0, 4])
def test_level1_optimized_path_is_stored_and_roundtrips(
        tiny_checkpoint, chunk_size):
    rng = np.random.RandomState(4)
    x = rng.rand(5, 6, 4).astype(np.float32)
    codec = GNNCompressorCodec(
        tiny_checkpoint,
        error_bound=0.01,
        levels=2,
        anchor_stride=4,
        anchor_block=1,
        max_radius=4,
        agg_level=1,
        chunk_size=chunk_size,
        compile=False,
    )

    stream = codec.compress(x)
    meta, _ = _read_stream(stream)
    y = codec.uncompress(stream)

    assert meta["agg_level"] == 1
    assert meta["prune_invalid_lines"] is True
    assert torch.max(torch.abs(y - torch.from_numpy(x))) <= 0.01


def test_level1_legacy_geometry_stream_overrides_decoder_default(tiny_checkpoint):
    rng = np.random.RandomState(6)
    x = rng.rand(5, 6, 4).astype(np.float32)
    encoder = GNNCompressorCodec(
        tiny_checkpoint, error_bound=0.01, levels=2, anchor_stride=4,
        anchor_block=1, max_radius=4, agg_level=1, chunk_size=0,
        compile=False, prune_invalid_lines=False)
    decoder = GNNCompressorCodec(
        tiny_checkpoint, error_bound=0.01, levels=2, anchor_stride=4,
        anchor_block=1, max_radius=4, agg_level=1, chunk_size=0,
        compile=False)

    stream = encoder.compress(x)
    meta, _ = _read_stream(stream)
    y = decoder.uncompress(stream)

    assert meta["prune_invalid_lines"] is False
    assert decoder.prune_invalid_lines is True
    assert torch.max(torch.abs(y - torch.from_numpy(x))) <= 0.01


def test_generic_stream_stores_level1_execution_path(tiny_checkpoint):
    rng = np.random.RandomState(5)
    x = rng.rand(8, 7).astype(np.float32)
    encoder = GNNPredictor(
        tiny_checkpoint, float(x.min()), float(x.max()), tile_size=0,
        max_radius=4, levels=2, anchor_stride=4, anchor_block=1,
        agg_level=1)

    stream, _ = compress(
        x, 0.01, encoder, levels=2, anchor_stride=4, anchor_block=1,
        tune="fast")
    header, _ = read_generic_stream(stream)

    assert header.agg_level == 1
    assert header.gnn_prune_invalid is True

    def decoder(header):
        return GNNPredictor(
            tiny_checkpoint, header.vmin, header.vmax, tile_size=0,
            max_radius=4, levels=header.levels,
            anchor_stride=header.anchor_stride,
            anchor_block=header.anchor_block, agg_level=header.agg_level,
            prune_invalid_lines=header.gnn_prune_invalid)

    y = decompress(stream, decoder)
    assert np.max(np.abs(y - x)) <= np.float32(0.01)


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
