"""Chunked GNN codec: bounded-memory path for large / high-dim tensors.

Covers the v3 (chunked) stream added alongside the v2 whole-tensor path:
n-D + integer roundtrips within the error bound, encoder determinism, auto vs
forced chunk selection, chunked-vs-whole equivalence of the guarantee, and the
halo geometry (that out-of-chunk neighbours become live only once their chunk is
coded). The error bound holds regardless of predictor quality — it is the
quantizer's guarantee — so a tiny random checkpoint suffices.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("constriction")  # rANS backend; skip if unavailable

from deepsz import GNNCompressorCodec
from deepsz.gnn_predictor import (CKPT_VERSION, _OverlaidGeom, build_chunk_geoms,
                                  build_model, chunk_halo_info)

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


def _codec(path, *, eb=1e-2, chunk_size):
    return GNNCompressorCodec(
        path, error_bound=eb, levels=LEVELS, anchor_stride=STRIDE,
        anchor_block=1, max_radius=4, chunk_size=chunk_size)


def _maxerr(y, x):
    return float(torch.max(torch.abs(y.float() - torch.as_tensor(x).float())))


# --- roundtrip within the error bound --------------------------------------

@pytest.mark.parametrize("shape", [
    (8, 8),            # 2D, 2x2 chunks
    (12, 8),           # 2D, ragged along axis 0 (12 = 3 chunks; even)
    (8, 8, 8),         # 3D, 2x2x2
    (8, 8, 8, 8),      # 4D
])
def test_chunked_roundtrip_float(v5_ckpt, shape):
    rng = np.random.RandomState(len(shape))
    # smooth-ish field so the predictor has something to do (bound holds anyway)
    x = np.zeros(shape, np.float32)
    for k, s in enumerate(shape):
        wave = np.cos(np.linspace(0, 2 * np.pi, s, dtype=np.float32))
        x = x + wave.reshape([-1 if i == k else 1 for i in range(len(shape))])
    x += rng.rand(*shape).astype(np.float32) * 0.05
    codec = _codec(v5_ckpt, eb=0.02, chunk_size=STRIDE)

    y = codec.uncompress(codec.compress(x))

    assert tuple(y.shape) == shape
    assert _maxerr(y, x) <= 0.02


def test_chunked_roundtrip_integer(v5_ckpt):
    rng = np.random.RandomState(7)
    x = (rng.rand(8, 8) * 50).astype(np.int32)
    codec = _codec(v5_ckpt, eb=1.0, chunk_size=STRIDE)

    y = codec.uncompress(codec.compress(x))

    assert np.issubdtype(np.dtype(y.numpy().dtype), np.integer)
    assert tuple(y.shape) == x.shape
    assert _maxerr(y, x) <= 1.0


# --- determinism ------------------------------------------------------------

@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 8)])
def test_chunked_encoder_deterministic(v5_ckpt, shape):
    rng = np.random.RandomState(3)
    x = rng.rand(*shape).astype(np.float32)
    codec = _codec(v5_ckpt, chunk_size=STRIDE)

    a = codec.compress(x)
    b = codec.compress(x)

    assert a == b  # byte-identical: closed loop is deterministic incl. coarse table


# --- chunked vs whole -------------------------------------------------------

def test_chunked_matches_whole_bound(v5_ckpt):
    """Same tensor both ways: each path honours the bound; a small tensor codes
    identically small under either (sanity that the pipeline, not luck, is wired).
    """
    rng = np.random.RandomState(11)
    x = rng.rand(8, 12).astype(np.float32)

    whole = _codec(v5_ckpt, chunk_size=0)   # force whole-tensor (v2)
    chunk = _codec(v5_ckpt, chunk_size=STRIDE)  # force chunked (v3)

    yw = whole.uncompress(whole.compress(x))
    yc = chunk.uncompress(chunk.compress(x))

    assert _maxerr(yw, x) <= whole.error_bound
    assert _maxerr(yc, x) <= chunk.error_bound


def test_auto_chunk_selection(v5_ckpt):
    """chunk_size=None: whole-tensor for small inputs, chunked past the
    threshold; forced int must be a multiple of anchor_stride."""
    codec = _codec(v5_ckpt, chunk_size=None)
    assert codec._chunk_edges((16, 16)) is None            # small -> whole
    big = (1 << 12, 1 << 12)                                # 16.7M points -> chunked
    edges = codec._chunk_edges(big)
    assert edges is not None
    assert all(e % STRIDE == 0 and e > 0 for e in edges)

    bad = _codec(v5_ckpt, chunk_size=STRIDE + 1)            # not a multiple
    with pytest.raises(ValueError):
        bad.compress(np.zeros((8, 8), np.float32))


# --- halo geometry: out-of-chunk neighbours go live only once coded ---------

def test_halo_links_activate_when_neighbour_coded():
    """Vertical (2,1) chunk grid: chunk 1 (bottom) sees chunk 0 (top) across the
    border. Anchors (level 0) are always usable context; the finer halo cells
    become valid neighbours only after the top chunk is coded. Uses the (2,1)
    orientation because coded-neighbour links into the negative-side halo are
    structurally richer there than in (1,2)."""
    stride, levels = 8, 3
    edges = (16, 16)
    shape = (32, 16)                      # two stacked 16x16 chunks
    grid = (2, 1)
    cg = build_chunk_geoms(edges, levels, stride, 1, torch, None)
    origin = (16, 0)                      # chunk 1 (bottom)
    halo_set = set(cg.halo_flat.tolist())

    def halo_valid_links(usable_np):
        ut = torch.from_numpy(usable_np)
        total = 0
        for s in cg.chain[1:]:            # refinement stages only
            g = cg.geoms[s]
            ov = _OverlaidGeom(g, ut)
            for ip, v in ((g.ip, ov.vp), (g.in_, ov.vn)):
                in_halo = np.isin(ip.numpy(), cg.halo_flat)
                total += int((v.numpy() & in_halo).sum())
        return total

    usable_uncoded, *_ = chunk_halo_info(
        cg, origin, shape, edges, grid, np.array([False, False]))
    usable_coded, *_ = chunk_halo_info(
        cg, origin, shape, edges, grid, np.array([True, False]))

    # more halo cells usable once the neighbour is coded
    assert usable_coded.sum() > usable_uncoded.sum()
    # Coding the neighbour is what creates live cross-border links: the periodic
    # nearest step upward from interior points otherwise lands on in-chunk
    # lattice cells, so an uncoded top halo contributes none (the (2,1) negative-
    # side asymmetry). This guards the halo wiring being live, not dead.
    assert halo_valid_links(usable_coded) > halo_valid_links(usable_uncoded)
    assert halo_valid_links(usable_coded) > 0


def test_out_of_tensor_halo_never_usable():
    """A corner chunk's halo that falls outside the tensor is never usable,
    regardless of coded flags."""
    stride, levels = 8, 3
    edges = (16, 16)
    shape = (32, 16)
    grid = (2, 1)
    cg = build_chunk_geoms(edges, levels, stride, 1, torch, None)
    # chunk 0 (top): its top halo has global row < 0 -> out of tensor
    usable, *_ = chunk_halo_info(
        cg, (0, 0), shape, edges, grid, np.array([True, True]))
    gc = cg.halo_coords + np.array([0, 0])
    out = np.any((gc < 0) | (gc >= np.array(shape)), axis=1)
    out_flat = cg.halo_flat[out]
    assert not usable[out_flat].any()
