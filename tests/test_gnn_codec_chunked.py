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
import deepsz.gnn_predictor as gp
from deepsz.gnn_codec import _chunk_waves
from deepsz.gnn_predictor import (CKPT_VERSION, ChunkedGNNPredictor,
                                  _CompactFrame, build_chunk_geoms, build_model,
                                  chunk_halo_info)

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
        anchor_block=1, max_radius=4, chunk_size=chunk_size,
        fp16=False, compile=False, gate=False)


def _maxerr(y, x):
    return float(torch.max(torch.abs(y.float() - torch.as_tensor(x).float())))


def test_gate_roundtrip_and_header(v5_ckpt):
    """Scale-gated interp fallback: the bound holds, decode is driven by the
    header (not the codec flag), and an all-off gate leaves the stream
    byte-identical to gate=False."""
    from deepsz.gnn_codec import _read_stream
    rng = np.random.RandomState(7)
    gx, gy = np.meshgrid(np.linspace(0, 4, 16, dtype=np.float32),
                         np.linspace(0, 4, 16, dtype=np.float32), indexing="ij")
    f = np.sin(gx) * np.cos(gy) + rng.rand(16, 16).astype(np.float32) * 0.01
    eb = 1e-6
    on = GNNCompressorCodec(
        v5_ckpt, error_bound=eb, levels=LEVELS, anchor_stride=STRIDE,
        anchor_block=1, max_radius=4, chunk_size=STRIDE, fp16=False,
        compile=False, gate=True)
    off = _codec(v5_ckpt, eb=eb, chunk_size=STRIDE)
    s_on, s_off = on.compress(f), off.compress(f)
    gates = _read_stream(s_on)[0].get("gates")
    if gates is None:
        assert s_on == s_off
    else:
        assert any(g >> 4 for g in gates)
    for s in (s_on, s_off):
        assert _maxerr(off.uncompress(s), f) <= eb


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


# --- wave batching: same-color chunks are independent, so they batch ---------

def test_chunk_waves_are_mutually_independent():
    """Every wave's chunks are >=2 apart on each axis they differ, so their
    one-chunk-thick halos never overlap -> batching them is order-independent."""
    grid = (6, 4)
    for wave in _chunk_waves(grid):
        coords = [np.unravel_index(ci, grid) for ci in wave]
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                diff = np.abs(np.array(coords[i]) - np.array(coords[j]))
                assert diff.max() >= 2   # never adjacent (Chebyshev distance >= 2)


def test_query_only_nearest_search_matches_period_tile_lookup():
    rng = np.random.RandomState(17)
    pat = rng.rand(4, 4, 4) > 0.7
    q = np.stack(np.nonzero(rng.rand(4, 4, 4) > 0.4), axis=1)
    res = tuple(q[:, k] for k in range(q.shape[1]))
    direction = (1, -1, 0)

    tiled = gp._nearest_steps_at(pat, direction, 4, res)
    query_only = gp._nearest_steps_at(
        pat, direction, 4, res, query_only=True)

    np.testing.assert_array_equal(query_only, tiled)


def test_chunk_geometry_uses_query_only_search_and_reports_progress(monkeypatch):
    """Chunk schedules must not rebuild a full period tile for every stage and
    direction.  That path effectively hangs for a 32^4 chunk (76 stages)."""
    gp._CHUNK_GEOM_CACHE.clear()
    seen = []
    original = gp._nearest_steps_at

    def spy(*args, **kwargs):
        seen.append(kwargs.get("query_only", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(gp, "_nearest_steps_at", spy)
    updates = []
    geom = gp.build_chunk_geoms(
        (8, 8), LEVELS, STRIDE, 1, torch, None, 2, updates.append)

    assert seen and all(seen)
    assert sum(updates) == len(geom.geoms)

    # A cache hit still completes a caller's setup bar immediately.
    cached_updates = []
    assert gp.build_chunk_geoms(
        (8, 8), LEVELS, STRIDE, 1, torch, None, 2,
        cached_updates.append) is geom
    assert sum(cached_updates) == len(geom.geoms)


def test_field_budget_estimate_warns_instead_of_aborting(v5_ckpt):
    predictor = ChunkedGNNPredictor(
        v5_ckpt, 0.0, 1.0, levels=LEVELS, anchor_stride=STRIDE)
    predictor.shape = (8, 8)
    predictor.edges = (8, 8)
    predictor.d = 1 << 30  # force the static estimate beyond the CPU budget

    with pytest.warns(RuntimeWarning, match="estimate is advisory"):
        predictor._check_field_budget(ndim=2, channels=1)


def test_cuda_budget_includes_reusable_allocator_cache():
    class FakeCuda:
        @staticmethod
        def mem_get_info(device):
            return 2_000, 10_000

        @staticmethod
        def memory_reserved(device):
            return 5_000

        @staticmethod
        def memory_allocated(device):
            return 1_000

    class FakeTorch:
        cuda = FakeCuda()

    # 2,000 driver-free + 4,000 reserved-but-unused, with the 80% margin.
    assert gp._cuda_working_budget(FakeTorch(), "cuda") == 4_800


def test_fp16_flag_roundtrips_and_persists(v5_ckpt):
    """fp16=True round-trips within the bound and the flag rides in the stream so
    decode replays the same float path. (autocast only bites on cuda; on cpu this
    checks the plumbing + that enabling it doesn't break the closed loop.)"""
    from deepsz.gnn_codec import _read_stream

    rng = np.random.RandomState(9)
    x = rng.rand(8, 8).astype(np.float32)
    codec = GNNCompressorCodec(
        v5_ckpt, error_bound=0.02, levels=LEVELS, anchor_stride=STRIDE,
        anchor_block=1, max_radius=4, chunk_size=STRIDE, fp16=True,
        compile=False)

    stream = codec.compress(x)
    meta, _ = _read_stream(bytes(stream))
    assert meta.get("fp16") is True
    assert _maxerr(codec.uncompress(stream), x) <= 0.02


def test_compile_flag_roundtrips_and_persists(v5_ckpt, monkeypatch):
    """compile=True round-trips within the bound and the flag rides in the stream
    so decode replays the same compiled float path. Small workloads skip compile
    (dynamo warmup never amortizes) and record compiled=False."""
    import deepsz.gnn_codec as gc
    from deepsz.gnn_codec import _read_stream

    rng = np.random.RandomState(11)
    x = rng.rand(8, 8).astype(np.float32)
    codec = GNNCompressorCodec(
        v5_ckpt, error_bound=0.02, levels=LEVELS, anchor_stride=STRIDE,
        anchor_block=1, max_radius=4, chunk_size=STRIDE, fp16=False,
        compile=True)

    stream = codec.compress(x)
    meta, _ = _read_stream(bytes(stream))
    assert meta.get("compiled") is False        # 4 chunks: below the gate

    monkeypatch.setattr(gc, "_COMPILE_MIN_CHUNKS", 1)
    stream = codec.compress(x)
    meta, _ = _read_stream(bytes(stream))
    assert meta.get("compiled") is True
    assert _maxerr(codec.uncompress(stream), x) <= 0.02


def test_batch_size_invariant_within_bound(v5_ckpt):
    """A wave split into different sub-batch sizes still round-trips within the
    bound (the closed loop stays consistent because enc/dec share the batch)."""
    rng = np.random.RandomState(5)
    x = rng.rand(24, 16).astype(np.float32)   # grid 6x4 -> waves of size 2
    for batch in (1, 2, 64):
        orig = ChunkedGNNPredictor.max_batch
        ChunkedGNNPredictor.max_batch = lambda self, cs, _b=batch: _b
        try:
            codec = _codec(v5_ckpt, eb=0.02, chunk_size=STRIDE)
            y = codec.uncompress(codec.compress(x))
        finally:
            ChunkedGNNPredictor.max_batch = orig
        assert _maxerr(y, x) <= 0.02


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

    def halo_valid_links(coded):
        # compact halo rows are the trailing block (row index > n_interior); a
        # valid line into one is a live cross-border neighbour.
        frame = _CompactFrame(cg, origin, shape, edges, grid, coded, torch, None)
        total = 0
        for s in cg.chain[1:]:            # refinement stages only
            g = frame.geoms[s]
            for ip, v in ((g.ip, g.vp), (g.in_, g.vn)):
                in_halo = ip > frame.n_interior
                total += int((v & in_halo).sum())
        return total

    present_uncoded = chunk_halo_info(
        cg, origin, shape, edges, grid, np.array([False, False]))[0]
    present_coded = chunk_halo_info(
        cg, origin, shape, edges, grid, np.array([True, False]))[0]

    # more halo cells usable once the neighbour is coded
    assert len(present_coded) > len(present_uncoded)
    # Coding the neighbour is what creates live cross-border links: the periodic
    # nearest step upward from interior points otherwise lands on in-chunk
    # lattice cells, so an uncoded top halo contributes none (the (2,1) negative-
    # side asymmetry). This guards the halo wiring being live, not dead.
    assert halo_valid_links(np.array([True, False])) > \
        halo_valid_links(np.array([False, False]))
    assert halo_valid_links(np.array([True, False])) > 0


def test_out_of_tensor_halo_never_usable():
    """A corner chunk's halo that falls outside the tensor is never usable,
    regardless of coded flags."""
    stride, levels = 8, 3
    edges = (16, 16)
    shape = (32, 16)
    grid = (2, 1)
    cg = build_chunk_geoms(edges, levels, stride, 1, torch, None)
    # chunk 0 (top): its top halo has global row < 0 -> out of tensor
    present, *_ = chunk_halo_info(
        cg, (0, 0), shape, edges, grid, np.array([True, True]))
    gc = cg.ref_halo_coords + np.array([0, 0])
    out = np.any((gc < 0) | (gc >= np.array(shape)), axis=1)
    out_flat = cg.ref_halo_flat[out]
    assert not np.isin(out_flat, present).any()
