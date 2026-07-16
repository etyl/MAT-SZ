"""Skeleton codec: global anchor-grid lines (SZ cubic/linear) + per-chunk GNN
interiors.

The error bound is the quantizer's guarantee (predictor-independent), so a tiny
random checkpoint suffices — same as the chunked-codec suite. These cover the
line-pass + interior-split roundtrip, integer sources, encoder determinism,
stream tagging, and that line points end up within the bound.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("constriction")  # rANS backend

from deepsz.gnn_predictor import CKPT_VERSION, build_model
from deepsz.skel_codec import SkeletonGNNCodec, _VERSION_SKEL
from deepsz.gnn_codec import _read_stream

STRIDE = 4
LEVELS = 2


@pytest.fixture()
def v5_ckpt(tmp_path):
    torch.manual_seed(0)
    model = build_model(d=8).eval()
    path = tmp_path / "gnn_v5.pt"
    torch.save({"d": model.d, "state_dict": model.state_dict(),
                "version": CKPT_VERSION}, path)
    return path


def _codec(path, *, eb=1e-2, chunk_size=STRIDE, order="cubic", interfaces=False):
    return SkeletonGNNCodec(
        path, error_bound=eb, levels=LEVELS, anchor_stride=STRIDE,
        anchor_block=1, max_radius=4, chunk_size=chunk_size,
        line_order=order, interfaces=interfaces, fp16=False, compile=False)


def _maxerr(y, x):
    return float(torch.max(torch.abs(y.float() - torch.as_tensor(x).float())))


def _field(shape):
    x = np.zeros(shape, np.float32)
    for k, s in enumerate(shape):
        wave = np.cos(np.linspace(0, 2 * np.pi, s, dtype=np.float32))
        x = x + wave.reshape([-1 if i == k else 1 for i in range(len(shape))])
    return x


@pytest.mark.parametrize("shape,chunk_size", [
    ((16, 16), STRIDE),        # 2D, one cell per chunk
    ((16, 16), 8),             # 2D, 2x2 cells per chunk
    ((20, 16), 8),             # 2D, ragged along axis 0
    ((16, 16, 16), 8),         # 3D
])
@pytest.mark.parametrize("order", ["cubic", "linear"])
def test_skel_roundtrip_float(v5_ckpt, shape, chunk_size, order):
    rng = np.random.RandomState(len(shape))
    x = _field(shape) + rng.rand(*shape).astype(np.float32) * 0.05
    codec = _codec(v5_ckpt, eb=0.02, chunk_size=chunk_size, order=order)

    stream = codec.compress(x)
    y = codec.uncompress(stream)

    assert tuple(y.shape) == shape
    assert _maxerr(y, x) <= 0.02


def test_skel_roundtrip_integer(v5_ckpt):
    rng = np.random.RandomState(7)
    x = (rng.rand(16, 16) * 50).astype(np.int32)
    codec = _codec(v5_ckpt, eb=1.0, chunk_size=8)

    y = codec.uncompress(codec.compress(x))

    assert np.issubdtype(np.dtype(y.numpy().dtype), np.integer)
    assert tuple(y.shape) == x.shape
    assert _maxerr(y, x) <= 1.0


def test_skel_encoder_deterministic(v5_ckpt):
    rng = np.random.RandomState(3)
    x = rng.rand(16, 16).astype(np.float32)
    codec = _codec(v5_ckpt, chunk_size=8)

    assert codec.compress(x) == codec.compress(x)   # byte-identical closed loop


def test_skel_stream_tagged(v5_ckpt):
    x = _field((16, 16))
    codec = _codec(v5_ckpt, order="linear")
    stream = codec.compress(x)

    meta, _ = _read_stream(bytes(stream))
    assert meta.get("skeleton") is True
    assert meta.get("line_order") == "linear"
    # version 4 is written in the prefix
    import struct
    from deepsz.gnn_codec import _PREFIX
    _, version, _ = struct.unpack_from(_PREFIX, bytes(stream), 0)
    assert version == _VERSION_SKEL


def test_line_points_within_bound(v5_ckpt):
    """Every point on an anchor-grid line (exactly one off-grid coord) is coded
    globally by the SZ line pass and must land within eb."""
    shape = (24, 24)
    x = _field(shape).astype(np.float32)
    eb = 0.02
    codec = _codec(v5_ckpt, eb=eb, chunk_size=8)

    y = codec.uncompress(codec.compress(x)).numpy()

    coords = np.indices(shape)
    off = sum((coords[k] % STRIDE != 0).astype(int) for k in range(2))
    line_mask = off == 1
    assert np.abs(y[line_mask] - x[line_mask]).max() <= eb + 1e-6


def test_non_skeleton_stream_rejected(v5_ckpt):
    """A stream from the plain chunked codec is not a skeleton stream."""
    from deepsz import GNNCompressorCodec

    x = _field((16, 16))
    plain = GNNCompressorCodec(
        v5_ckpt, error_bound=0.02, levels=LEVELS, anchor_stride=STRIDE,
        anchor_block=1, max_radius=4, chunk_size=8, fp16=False, compile=False)
    stream = plain.compress(x)

    with pytest.raises(ValueError):
        _codec(v5_ckpt, chunk_size=8).uncompress(stream)


@pytest.mark.parametrize("shape,chunk_size", [
    ((16, 16, 16), 8),         # 3D, 2x2x2 chunks -> interior chunk faces exist
    ((24, 16, 16), 8),         # 3D, ragged along axis 0
    ((16, 16, 16, 16), 8),     # 4D
])
@pytest.mark.parametrize("order", ["cubic", "linear"])
def test_skel_interfaces_roundtrip(v5_ckpt, shape, chunk_size, order):
    """Milestone B: chunk-boundary interiors (interfaces) coded in a global
    phase 1, strict interiors in phase 2 with the reconstructed faces as halo.
    Must still hold the error bound and roundtrip exactly."""
    rng = np.random.RandomState(len(shape))
    x = _field(shape) + rng.rand(*shape).astype(np.float32) * 0.05
    codec = _codec(v5_ckpt, eb=0.02, chunk_size=chunk_size, order=order,
                   interfaces=True)

    stream = codec.compress(x)
    y = codec.uncompress(stream)

    assert tuple(y.shape) == shape
    assert _maxerr(y, x) <= 0.02
    meta, _ = _read_stream(bytes(stream))
    assert meta.get("interfaces") is True


def test_skel_interfaces_deterministic(v5_ckpt):
    rng = np.random.RandomState(5)
    x = (_field((16, 16, 16)) + rng.rand(16, 16, 16).astype(np.float32) * 0.05)
    codec = _codec(v5_ckpt, chunk_size=8, interfaces=True)
    assert codec.compress(x) == codec.compress(x)   # byte-identical closed loop


def test_skel_chunks_are_halo_free(v5_ckpt):
    """The core task-4 invariant: a chunk's stage predictions depend only on the
    global skeleton (anchors + lines), never on another chunk's interior. So
    perturbing a *different* chunk's interior cells in the recon must leave this
    chunk's predictions bit-identical -> chunks are independent / parallel."""
    from deepsz.gnn_codec import (_anchor_axes, _code_anchor_stage,
                                  _chunk_stage_ebs)
    from deepsz.skel_codec import _code_line_pass, _interior_split
    from deepsz.levels import stage_masks

    shape, eb = (16, 16), 0.02
    x = _field(shape).astype(np.float32)
    codec = _codec(v5_ckpt, eb=eb, chunk_size=8)   # 2x2 chunks of 8x8
    vals = x[None].astype(np.float32)
    ebs = _chunk_stage_ebs(shape, LEVELS, STRIDE, 1, eb, 0.8)

    # recon holding only the global skeleton (interiors left at zero)
    skel = np.zeros_like(vals)
    _code_anchor_stage(vals, skel, _anchor_axes(shape, STRIDE, 1), ebs[0],
                       codec.radius, False)
    _code_line_pass(vals, skel, shape, STRIDE, 1, eb, 0.8, codec.radius,
                    "cubic", False)

    def chunk0_preds(seed):
        pred = codec._chunked_predictor(float(vals.min()), float(vals.max()))
        pred.begin(shape, codec._skel_edges(shape), channels=1)
        rec = seed.copy()
        sls = pred.chunk_slices(0)
        cshape = tuple(s.stop - s.start for s in sls)
        cm = stage_masks(cshape, pred.levels, STRIDE, 1)
        counts = [int(p.sum()) for p in cm]
        split = _interior_split(cshape, cm, STRIDE)
        pred.start_chunk(0, rec)
        outs = []
        for s in range(1, len(cm)):
            if counts[s] == 0:
                continue
            p, _ = pred.predict_stage(s, rec, ebs[s])
            outs.append(p.copy())
            pos_int, _, n_int = split[s]
            if n_int:   # advance the chain with truth, as coding would
                rec[(slice(None), *sls)][:, pos_int] = \
                    vals[(slice(None), *sls)][:, pos_int]
        return np.concatenate([o.ravel() for o in outs])

    # corrupt interior (>=2 off-grid) cells that lie OUTSIDE chunk 0, leaving the
    # skeleton intact; chunk 0 must not notice.
    coords = np.indices(shape)
    off = sum((coords[k] % STRIDE != 0).astype(int) for k in range(2))
    interior = off >= 2
    outside0 = np.ones(shape, bool)
    outside0[0:8, 0:8] = False
    garbage = skel.copy()
    rng = np.random.RandomState(0)
    m = interior & outside0
    garbage[0][m] += rng.rand(int(m.sum())).astype(np.float32) * 5.0

    assert np.array_equal(chunk0_preds(skel), chunk0_preds(garbage))
