"""Checkpoint-backed, tensor-shaped GNN compressor codec."""

from __future__ import annotations

import json
import os
import queue
import struct
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import zstandard

from .codec import _compress_tile
from . import gnn_predictor as _gp
from .gnn_predictor import ChunkedGNNPredictor, GNNPredictor
from .levels import stage_ebs, stage_masks
from .quantizer import dequantize, quantize
from .bitstream import pack_stage, unpack_stage
from .rans import build_laplace_tables, scale_to_level


_MAGIC = b"MATSZGNN"
_VERSION = 2          # whole-tensor streams
_VERSION_CHUNKED = 3  # chunk-major streams (global anchors + per-chunk stages)
_PREFIX = "<8sII"
_PREFIX_SIZE = struct.calcsize(_PREFIX)

# auto mode: whole-tensor below this many points, chunked above (whole-tensor
# memory is ~30*L*K*d bytes/point in transients — ~2^21 points is a few GB)
_AUTO_CHUNK_THRESHOLD = 1 << 21
_AUTO_CHUNK_POINTS = 1 << 18  # target points per chunk
# torch.compile only pays past this many chunks (dynamo warmup is seconds; the
# fused embed saves ~ms per wave). ponytail: rough amortization cutoff, tune if
# compile cost or per-wave savings change materially.
_COMPILE_MIN_CHUNKS = 64


def _log(msg):
    # ponytail: env-gated so tests/CLI stay quiet; set DEEPSZ_PROGRESS=1 to see it
    if not os.environ.get("DEEPSZ_PROGRESS"):
        return
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _cuda_peak(predictor):
    """Peak GPU bytes since the last call (resets the counter), or None on CPU."""
    torch = getattr(predictor, "_torch", None)
    dev = getattr(predictor, "device", None)
    if torch is None or dev is None or dev.type != "cuda":
        return None
    peak = torch.cuda.max_memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    return peak


def _progress_bar(tag, n):
    # env-gated (DEEPSZ_PROGRESS) so tests/CLI stay quiet; disabled bar is a no-op.
    from tqdm import tqdm
    return tqdm(total=n, desc=tag, unit="wave", file=sys.stderr,
                disable=not os.environ.get("DEEPSZ_PROGRESS"))


def _as_numpy(x: Any) -> np.ndarray:
    """Accept numpy arrays and torch tensors without importing torch eagerly."""
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    cpu = getattr(x, "cpu", None)
    numpy = getattr(x, "numpy", None)
    if detach is not None and cpu is not None:
        return x.detach().cpu().numpy()
    if numpy is not None:
        return x.numpy()
    return np.asarray(x)


def _meta_agg_level(meta: dict[str, Any]) -> int | None:
    """Neighbourhood aggregation level recorded in a stream, or None (full) for
    streams written before the option existed."""
    v = meta.get("agg_level")
    return None if v is None else int(v)


def _dtype_meta(dtype: np.dtype) -> dict[str, Any]:
    dtype = np.dtype(dtype)
    return {
        "str": dtype.str,
        "kind": dtype.kind,
        "itemsize": dtype.itemsize,
    }


def _restore_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    elif dtype.kind == "b":
        values = values >= 0.5
    return values.astype(dtype, copy=False)


def _write_stream(meta: dict[str, Any], payload: bytes, zstd_level: int,
                  version: int = _VERSION) -> bytes:
    header = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = zstandard.ZstdCompressor(level=zstd_level).compress(payload)
    return struct.pack(_PREFIX, _MAGIC, version, len(header)) + header + body


def _read_stream(stream: bytes) -> tuple[dict[str, Any], bytes]:
    if len(stream) < _PREFIX_SIZE:
        raise ValueError("not a DeepSZ GNN stream")
    magic, version, header_len = struct.unpack_from(_PREFIX, stream, 0)
    if magic != _MAGIC:
        raise ValueError(f"not a DeepSZ GNN stream (bad magic {magic!r})")
    if version not in (_VERSION, _VERSION_CHUNKED):
        raise ValueError(f"unsupported DeepSZ GNN stream version {version}")
    off = _PREFIX_SIZE
    meta = json.loads(stream[off:off + header_len].decode("utf-8"))
    payload = zstandard.ZstdDecompressor().decompress(stream[off + header_len:])
    return meta, payload


def _empty_stats(n_stages: int) -> dict[str, Any]:
    return {
        "predict_s": 0.0,
        "quantize_s": 0.0,
        "entropy_s": 0.0,
        "outliers": 0,
        "stage_codes": [0] * n_stages,
        "stage_outliers": [0] * n_stages,
        "stage_payload_bytes": [0] * n_stages,
        "stage_model_bits": [0.0] * n_stages,
        "stage_pred_sae": [0.0] * n_stages,
        "stage_pred_sse": [0.0] * n_stages,
        "stage_recon_sae": [0.0] * n_stages,
        "stage_recon_sse": [0.0] * n_stages,
        "stage_recon_max": [0.0] * n_stages,
    }


def _decompress_region(
    payload: bytes,
    shape: tuple[int, ...],
    masks: list[np.ndarray],
    ebs: list[float],
    radius: int,
    predictor: GNNPredictor,
    use_rans: bool,
) -> np.ndarray:
    recon = np.zeros((1, *shape), np.float32)
    known = np.zeros(shape, bool)
    off = 0
    for stage_idx, pos in enumerate(masks):
        n = int(pos.sum())
        if n == 0:
            if use_rans:
                tables = build_laplace_tables(ebs[stage_idx], radius)
                codes, outliers, off = unpack_stage(
                    payload, off, rans_levels=np.zeros(0, np.uint8),
                    rans_tables=tables)
            else:
                codes, outliers, off = unpack_stage(payload, off)
            continue
        if stage_idx == 0:
            pred = np.zeros((1, n), np.float32)
            scale = np.full((1, n), ebs[stage_idx], np.float32)
        else:
            if use_rans:
                pred, scale = predictor.predict(recon, known, pos,
                                                eb=ebs[stage_idx])
            else:
                got = predictor.predict(recon, known, pos, eb=ebs[stage_idx])
                pred = got[0] if isinstance(got, tuple) else got
                scale = None
        if use_rans:
            tables = build_laplace_tables(ebs[stage_idx], radius)
            levels64 = scale_to_level(scale, ebs[stage_idx]).reshape(-1)
            codes, outliers, off = unpack_stage(
                payload, off, rans_levels=levels64, rans_tables=tables)
        else:
            codes, outliers, off = unpack_stage(payload, off)
        recon[:, pos] = dequantize(pred, codes, outliers, ebs[stage_idx],
                                   radius).reshape(1, n)
        known |= pos
    if off != len(payload):
        raise ValueError("trailing bytes in DeepSZ GNN payload")
    return recon[0]


def _chunk_stage_ebs(shape, levels, stride, block, eb, eb_ratio) -> list[float]:
    """Per-stage error bounds for the chunked path. The stage strides depend
    only on (rank, levels, stride), so evaluate ``stage_ebs`` on a tiny
    same-rank shape — never on the full tensor (that would materialise
    full-shape stage masks, the memory bug this path removes)."""
    return stage_ebs((2 * stride,) * len(shape), levels, stride, block, eb,
                     eb_ratio)


def _anchor_axes(shape: tuple[int, ...], stride: int, block: int) -> list[np.ndarray]:
    """Per-axis anchor coordinates. The anchor set is separable (every
    coordinate has residue < block mod stride), so the global anchor pass can
    index it with np.ix_ and never materialise a full-shape mask."""
    axes = []
    for n in shape:
        c = np.arange(n)
        axes.append(c[(c % stride) < block])
    return axes


def _auto_chunk_edges(shape: tuple[int, ...], stride: int) -> tuple[int, ...]:
    """Uniform chunk edge targeting ~_AUTO_CHUNK_POINTS points per chunk,
    rounded down to a multiple of the anchor stride (>= one stride)."""
    target = _AUTO_CHUNK_POINTS ** (1.0 / len(shape))
    edge = max(stride, int(target) // stride * stride)
    return (edge,) * len(shape)


def _code_anchor_stage(values, recon, axes, eb0, radius, round_output):
    """Encoder side of the global anchor pass: quantize anchors against pred 0,
    write their recon, return the packed stage."""
    c = values.shape[0]
    sub = (slice(None), *np.ix_(*axes))
    avals = values[sub].reshape(c, -1)
    n = avals.shape[1]
    pred = np.zeros((c, n), np.float32)
    codes, outliers = quantize(avals, pred, eb0, radius,
                               round_output=round_output)
    recon[sub] = dequantize(pred, codes, outliers, eb0, radius).reshape(
        recon[sub].shape)
    tables = build_laplace_tables(eb0, radius)
    levels64 = scale_to_level(np.full((c, n), eb0, np.float32), eb0).reshape(-1)
    return pack_stage(codes, outliers, rans_levels=levels64, rans_tables=tables)


def _decode_anchor_stage(payload, off, recon, axes, eb0, radius):
    c = recon.shape[0]
    sub = (slice(None), *np.ix_(*axes))
    n = int(np.prod([len(a) for a in axes]))
    tables = build_laplace_tables(eb0, radius)
    levels64 = scale_to_level(np.full((c, n), eb0, np.float32), eb0).reshape(-1)
    codes, outliers, off = unpack_stage(payload, off, rans_levels=levels64,
                                        rans_tables=tables)
    pred = np.zeros((c, n), np.float32)
    recon[sub] = dequantize(pred, codes, outliers, eb0, radius).reshape(
        recon[sub].shape)
    return off


def _chunk_waves(grid: tuple[int, ...]) -> list[list[int]]:
    """Group chunk ids into color waves. A wave = chunks with the same per-axis
    parity ("color") and the same tensor-boundary signature. Same-color chunks
    are >=2 apart on every axis they differ, so their halos (thickness one chunk)
    never overlap -> they are mutually independent and, given the color ordering,
    share one coded-neighbour pattern -> identical stage geometry, so they batch
    in the model's B dim. Ordered by color so a wave's cross-color neighbours in
    earlier colors are already coded. Correctness (the error bound) holds for any
    order; only which context is available, hence the ratio, shifts."""
    ndim = len(grid)
    groups: dict = {}
    for ci in range(int(np.prod(grid))):
        cidx = np.unravel_index(ci, grid)
        color = tuple(int(i) % 2 for i in cidx)
        bsig = tuple((int(i) == 0, int(i) == g - 1) for i, g in zip(cidx, grid))
        groups.setdefault((color, bsig), []).append(ci)

    def rank(key):
        color, bsig = key
        return (sum(b << k for k, b in enumerate(color)), bsig)

    return [groups[k] for k in sorted(groups, key=rank)]


def _compress_chunked(
    values: np.ndarray,
    ebs: list[float],
    radius: int,
    round_output: bool,
    predictor: ChunkedGNNPredictor,
    edges: tuple[int, ...],
    batch_cap: int | None = None,
    overlap: bool = False,
) -> bytes:
    """Wave-batched encode: global anchor pass, then chunks coded in color waves
    (see `_chunk_waves`), each wave split into memory-bounded sub-batches run
    together in the model's B dim. Stream order is wave -> sub-batch -> stage ->
    chunk, mirrored bitwise by the decoder. Peak memory is O(batch * chunk).

    ``overlap``: run the per-stage rANS packing on a background thread. Only the
    quantize+dequantize (which writes recon that the next forward reads) stays on
    the critical path; pack_stage does not feed back, so it can in principle hide
    behind the next stage's GPU forward. Output bytes are identical -- each pack
    writes into its reserved slot, flattened in order at the end.

    Caveat (measured): constriction's rANS holds the GIL, so it does not actually
    overlap the main thread's Python-driven launch loop on an eager/latency-bound
    GPU -- it comes out ~neutral here. It is likeliest to pay off where the GPU
    forward is fused/long (``--compile`` on Volta+) so the main thread sits in
    GIL-releasing CUDA syncs the worker can drain into. Opt-in for that reason."""
    c = values.shape[0]
    shape = values.shape[1:]
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros_like(values)
    axes = _anchor_axes(shape, stride, block)
    _log(f"encode: shape={shape} edges={edges} device={predictor.device} "
         f"coding anchors...")
    parts = [_code_anchor_stage(values, recon, axes, ebs[0], radius,
                                round_output)]
    predictor.begin(shape, edges, channels=c)
    predictor.anchor_coarse(recon)
    B_cap = predictor.max_batch(tuple(min(e, n) for e, n in zip(edges, shape)))
    if batch_cap is not None:                        # user cap (never above safe)
        B_cap = max(1, min(B_cap, int(batch_cap)))
    predictor.chunk_batch = B_cap                    # surfaced into stream meta
    waves = _chunk_waves(predictor.grid)
    n_sub = sum(-(-len(g) // B_cap) for g in waves)
    _log(f"encode: anchors done, {predictor.n_chunks} chunks, batch={B_cap}, "
         f"{n_sub} model-waves")
    # Optional background rANS: pack_stage runs on a worker while the main thread
    # drives the next GPU forward. Each emit reserves an ordered slot the worker
    # fills, so the joined byte stream is identical to the synchronous path.
    task_q: queue.Queue | None = None
    worker: threading.Thread | None = None
    worker_err: list[BaseException] = []
    if overlap:
        # Unbounded on purpose: a bounded queue back-pressures the main thread on
        # put() whenever the worker can't drain, and constriction's rANS holds
        # the GIL, so the worker *is* starved during the main thread's launch
        # loop -- a small cap turns neutral into a large regression (measured
        # -24% at cap 64). Trades RAM (buffered stage codes) for that safety.
        task_q = queue.Queue()

        def _rans_worker():
            while True:
                item = task_q.get()
                try:
                    if item is None:
                        return
                    slot, cd, ol, lv, tb = item
                    slot[0] = pack_stage(cd, ol, rans_levels=lv, rans_tables=tb)
                except BaseException as exc:                    # surface to main
                    worker_err.append(exc)
                    return
                finally:
                    task_q.task_done()

        worker = threading.Thread(target=_rans_worker, daemon=True)
        worker.start()

    def emit(codes, outliers, levels, tables):
        if overlap:
            if worker_err:
                raise worker_err[0]
            slot: list = [None]
            parts.append(slot)
            task_q.put((slot, codes, outliers, levels, tables))
        else:
            parts.append(pack_stage(codes, outliers, rans_levels=levels,
                                    rans_tables=tables))

    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    mask_cache: dict = {}    # cshape -> (stage masks, per-stage counts)
    bar = _progress_bar("encode", n_sub)
    for group in waves:
        for i in range(0, len(group), B_cap):
            ids = group[i:i + B_cap]
            cshape = tuple(sl.stop - sl.start
                           for sl in predictor.chunk_slices(ids[0]))
            if cshape not in mask_cache:
                cm = stage_masks(cshape, predictor.levels, stride, block)
                mask_cache[cshape] = (cm, [int(p.sum()) for p in cm])
            cmasks, counts = mask_cache[cshape]
            predictor.start_wave(ids, recon)
            for s in range(1, len(cmasks)):
                pos = cmasks[s]
                n = counts[s]
                tables = stage_tables[s]
                if n == 0:
                    for _ in ids:
                        emit(np.zeros(0, np.uint32), np.zeros(0, np.float32),
                             np.zeros(0, np.uint8), tables)
                    continue
                pred, scale = predictor.predict_wave_stage(s, recon, ebs[s])
                for bi, ci in enumerate(ids):
                    sls = predictor.chunk_slices(ci)
                    cvals = values[(slice(None), *sls)][:, pos]
                    p = pred[bi][None, :]
                    codes, outliers = quantize(cvals, p, ebs[s], radius,
                                               round_output=round_output)
                    recon[(slice(None), *sls)][:, pos] = dequantize(
                        p, codes, outliers, ebs[s], radius).reshape(c, n)
                    emit(codes, outliers,
                         scale_to_level(scale[bi][None, :], ebs[s]).reshape(-1),
                         tables)
            predictor.finish_wave(recon)
            peak = _cuda_peak(predictor)
            if peak:
                bar.set_postfix_str(f"peak {peak / 1e9:.2f}GB")
            bar.update(1)
    bar.close()
    if overlap:
        task_q.put(None)
        worker.join()
        if worker_err:
            raise worker_err[0]
        parts = [p[0] if isinstance(p, list) else p for p in parts]
    return b"".join(parts)


def _decompress_chunked(
    payload: bytes,
    shape: tuple[int, ...],
    ebs: list[float],
    radius: int,
    predictor: ChunkedGNNPredictor,
    edges: tuple[int, ...],
    batch: int,
) -> np.ndarray:
    c = 1
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros((c, *shape), np.float32)
    axes = _anchor_axes(shape, stride, block)
    _log(f"decode: shape={shape} edges={edges} decoding anchors...")
    off = _decode_anchor_stage(payload, 0, recon, axes, ebs[0], radius)
    predictor.begin(shape, edges, channels=c)
    predictor.anchor_coarse(recon)
    B_cap = max(1, int(batch))
    waves = _chunk_waves(predictor.grid)
    n_sub = sum(-(-len(g) // B_cap) for g in waves)
    _log(f"decode: anchors done, {predictor.n_chunks} chunks, batch={B_cap}, "
         f"{n_sub} model-waves")
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    mask_cache: dict = {}    # cshape -> (stage masks, per-stage counts)
    bar = _progress_bar("decode", n_sub)
    for group in waves:
        for i in range(0, len(group), B_cap):
            ids = group[i:i + B_cap]
            cshape = tuple(sl.stop - sl.start
                           for sl in predictor.chunk_slices(ids[0]))
            if cshape not in mask_cache:
                cm = stage_masks(cshape, predictor.levels, stride, block)
                mask_cache[cshape] = (cm, [int(p.sum()) for p in cm])
            cmasks, counts = mask_cache[cshape]
            predictor.start_wave(ids, recon)
            for s in range(1, len(cmasks)):
                pos = cmasks[s]
                n = counts[s]
                tables = stage_tables[s]
                if n == 0:
                    for _ in ids:
                        _c, _o, off = unpack_stage(
                            payload, off, rans_levels=np.zeros(0, np.uint8),
                            rans_tables=tables)
                    continue
                pred, scale = predictor.predict_wave_stage(s, recon, ebs[s])
                for bi, ci in enumerate(ids):
                    levels64 = scale_to_level(
                        scale[bi][None, :], ebs[s]).reshape(-1)
                    codes, outliers, off = unpack_stage(
                        payload, off, rans_levels=levels64, rans_tables=tables)
                    sls = predictor.chunk_slices(ci)
                    recon[(slice(None), *sls)][:, pos] = dequantize(
                        pred[bi][None, :], codes, outliers, ebs[s],
                        radius).reshape(c, n)
            predictor.finish_wave(recon)
            peak = _cuda_peak(predictor)
            if peak:
                bar.set_postfix_str(f"peak {peak / 1e9:.2f}GB")
            bar.update(1)
    bar.close()
    if off != len(payload):
        raise ValueError("trailing bytes in DeepSZ GNN payload")
    return recon[0]


class GNNCompressorCodec:
    """Usable Python codec for GNN-backed DeepSZ tensor compression.

    The codec is initialized from a GNN checkpoint path. ``compress`` accepts a
    numpy array or torch tensor of any rank and returns bytes. ``uncompress``
    accepts those bytes and returns a torch tensor with the original shape and
    dtype.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        error_bound: float = 1e-2,
        *,
        levels: int = 5,
        anchor_stride: int = 32,
        anchor_block: int = 1,
        radius: int = 1 << 15,
        max_radius: int = 64,
        agg_level: int | None = 2,
        device: str | None = None,   # None -> cuda if available, else cpu
        zstd_level: int = 9,
        eb_ratio: float | None = None,  # None = auto: fast -> 0.8, size -> sweep
        tune: str = "fast",
        strict_checkpoint: bool = True,
        chunk_size: int | tuple[int, ...] | None = None,
        chunk_batch: int | None = None,
        fp16: bool = False,
        compile: bool = True,
        overlap: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GNN checkpoint not found: {self.checkpoint_path}")
        if error_bound <= 0:
            raise ValueError("error_bound must be > 0")
        if tune not in ("fast", "size"):
            raise ValueError("tune must be 'fast' or 'size'")

        self.error_bound = float(error_bound)
        self.levels = int(levels)
        self.anchor_stride = int(anchor_stride)
        self.anchor_block = int(anchor_block)
        self.radius = int(radius)
        self.max_radius = int(max_radius)
        # Neighbourhood aggregation level: cap on the L1 length of the GNN's
        # neighbour lines (None = full neighbourhood). Smaller = fewer directions
        # per point = faster inference, most impactful in high dimensions. Frozen
        # into the stream so decode reproduces the encoder's prediction bitwise.
        self.agg_level = None if agg_level is None else int(agg_level)
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.zstd_level = int(zstd_level)
        self.eb_ratio = eb_ratio
        self.tune = tune
        self.strict_checkpoint = bool(strict_checkpoint)
        # chunk_size: None = auto (whole-tensor for small inputs, chunked
        # above _AUTO_CHUNK_THRESHOLD points); 0 = force whole-tensor; an int
        # or per-axis tuple forces chunked with those edges (multiples of
        # anchor_stride).
        self.chunk_size = chunk_size
        # chunk_batch: None = auto (as many same-geometry chunks as fit the encode
        # GPU); an int caps it (capped again at the memory-safe auto value). The
        # value used is frozen into the stream so decode replays it — set it to
        # what the smallest decode device can hold.
        self.chunk_batch = None if chunk_batch is None else max(1, int(chunk_batch))
        # fp16: run the message-pass matmuls in fp16 autocast (cuda only; the
        # readout stays fp32). ~2x on the GNN forward, may cost a little ratio at
        # small eb -> opt-in. Stored in meta so decode uses the same float path.
        self.fp16 = bool(fp16)
        # compile: torch.compile the message-pass embed (fuses the elementwise
        # ops that aren't in the GEMMs). Stored in meta so decode uses the same
        # compiled float path. First encode pays a one-off compilation cost.
        self.compile = bool(compile)
        # overlap: run per-stage rANS packing on a background thread so it hides
        # behind the next stage's GPU forward. Encode-only; the output bytes are
        # identical, so nothing about the stream or decode changes.
        self.overlap = bool(overlap)
        self.checkpoint_hash = self._checkpoint_hash()

    def _chunk_edges(self, shape: tuple[int, ...]) -> tuple[int, ...] | None:
        """Chunk edges for this shape, or None for the whole-tensor path."""
        cs = self.chunk_size
        if cs == 0:
            return None
        if cs is None:
            if int(np.prod(shape)) <= _AUTO_CHUNK_THRESHOLD:
                return None
            return _auto_chunk_edges(shape, self.anchor_stride)
        edges = ((int(cs),) * len(shape) if np.isscalar(cs)
                 else tuple(int(e) for e in cs))
        if len(edges) != len(shape):
            raise ValueError("chunk_size must be scalar or one entry per axis")
        for e in edges:
            if e < self.anchor_stride or e % self.anchor_stride:
                raise ValueError("chunk_size must be a positive multiple of "
                                 "anchor_stride")
        return edges

    def compress(self, x: Any, error_bound: float | None = None) -> bytes:
        """Compress a numpy array or torch tensor of any rank into bytes."""
        arr = np.asarray(_as_numpy(x))
        if arr.size == 0:
            raise ValueError("cannot compress an empty tensor")
        if arr.dtype.kind not in "biuf":
            raise TypeError(f"unsupported dtype {arr.dtype}; expected numeric data")

        dtype = np.dtype(arr.dtype)
        original_shape = tuple(int(n) for n in arr.shape)
        shape = original_shape if original_shape else (1,)
        values = arr.reshape(shape).astype(np.float32, copy=False)
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        eb = self.error_bound if error_bound is None else float(error_bound)
        if eb <= 0:
            raise ValueError("error_bound must be > 0")

        ratio_candidates = (
            [float(self.eb_ratio)] if self.eb_ratio is not None
            else ([1.0, 0.9, 0.8, 0.7] if self.tune == "size" else [0.8])
        )
        edges = self._chunk_edges(shape)
        # torch.compile costs seconds of dynamo warmup per process; only worth
        # it when there are enough chunk waves to amortize. Frozen into the
        # stream meta so decode replays the same float path.
        use_compile = self.compile and edges is not None and int(np.prod(
            [-(-n // e) for n, e in zip(shape, edges)])) >= _COMPILE_MIN_CHUNKS
        candidates: list[tuple[int, bytes]] = []
        for ratio in ratio_candidates:
            chunk_batch = None
            if edges is None:
                payload = self._compress_payload(values, dtype, eb, vmin, vmax,
                                                 ratio)
            else:
                payload, chunk_batch = self._compress_chunked_payload(
                    values, dtype, eb, vmin, vmax, ratio, edges, use_compile)
            meta = {
                "codec": "deepsz.gnn",
                "shape": list(original_shape),
                "coded_shape": list(shape),
                "dtype": _dtype_meta(dtype),
                "error_bound": eb,
                "levels": self.levels,
                "anchor_stride": self.anchor_stride,
                "anchor_block": self.anchor_block,
                "radius": self.radius,
                "max_radius": self.max_radius,
                "agg_level": self.agg_level,
                "vmin": vmin,
                "vmax": vmax,
                "eb_ratio": ratio,
                "entropy_coder": "rans",
                "checkpoint_hash": self.checkpoint_hash.hex(),
            }
            if edges is not None:
                meta["chunks"] = list(edges)
                meta["chunk_batch"] = int(chunk_batch)
                meta["m_tile"] = int(_gp._M_TILE)   # replay the exact float path
                meta["fp16"] = bool(self.fp16)
                meta["compiled"] = bool(use_compile)
            stream = _write_stream(meta, payload, self.zstd_level,
                                   _VERSION if edges is None
                                   else _VERSION_CHUNKED)
            candidates.append((len(stream), stream))
        return min(candidates, key=lambda item: item[0])[1]

    def uncompress(self, stream: bytes | bytearray | memoryview):
        """Decompress bytes from ``compress`` and return a torch tensor."""
        import torch

        meta, payload = _read_stream(bytes(stream))
        if meta.get("codec") != "deepsz.gnn":
            raise ValueError("not a DeepSZ GNN tensor stream")
        got_hash = meta.get("checkpoint_hash")
        if self.strict_checkpoint and got_hash != self.checkpoint_hash.hex():
            raise ValueError("checkpoint hash differs from the stream metadata")

        shape = tuple(int(n) for n in meta["coded_shape"])
        original_shape = tuple(int(n) for n in meta["shape"])
        dtype = np.dtype(meta["dtype"]["str"])
        vmin = float(meta["vmin"])
        vmax = float(meta["vmax"])
        if vmax <= vmin:
            vmax = vmin + 1.0

        if "chunks" in meta:
            edges = tuple(int(e) for e in meta["chunks"])
            predictor = self._chunked_predictor(vmin, vmax, meta)
            ebs = _chunk_stage_ebs(shape, int(meta["levels"]),
                                   int(meta["anchor_stride"]),
                                   int(meta["anchor_block"]),
                                   float(meta["error_bound"]),
                                   float(meta["eb_ratio"]))
            saved_tile = _gp._M_TILE
            _gp._M_TILE = int(meta.get("m_tile", saved_tile))  # match encode path
            try:
                values = _decompress_chunked(payload, shape, ebs,
                                             int(meta["radius"]), predictor, edges,
                                             int(meta.get("chunk_batch", 1)))
            finally:
                _gp._M_TILE = saved_tile
            out = _restore_dtype(values.reshape(original_shape), dtype)
            return torch.as_tensor(out)

        predictor = self._predictor(vmin, vmax, meta)
        masks = stage_masks(shape, int(meta["levels"]), int(meta["anchor_stride"]),
                            int(meta["anchor_block"]))
        ebs = stage_ebs(shape, int(meta["levels"]), int(meta["anchor_stride"]),
                        int(meta["anchor_block"]), float(meta["error_bound"]),
                        float(meta["eb_ratio"]))
        use_rans = meta.get("entropy_coder", "huffman") == "rans"
        values = _decompress_region(payload, shape, masks, ebs, int(meta["radius"]),
                                    predictor, use_rans)
        out = _restore_dtype(values.reshape(original_shape), dtype)
        return torch.as_tensor(out)

    decompress = uncompress

    def _compress_payload(
        self,
        values: np.ndarray,
        dtype: np.dtype,
        eb: float,
        vmin: float,
        vmax: float,
        eb_ratio: float,
    ) -> bytes:
        predictor = self._predictor(vmin, vmax)
        masks = stage_masks(values.shape, self.levels, self.anchor_stride,
                            self.anchor_block)
        ebs = stage_ebs(values.shape, self.levels, self.anchor_stride,
                        self.anchor_block, eb, eb_ratio)
        stats = _empty_stats(len(masks))
        payload, _ = _compress_tile(values[None, ...], masks, ebs, predictor,
                                    self.radius, dtype.kind in "bi", stats)
        return payload

    def _compress_chunked_payload(
        self,
        values: np.ndarray,
        dtype: np.dtype,
        eb: float,
        vmin: float,
        vmax: float,
        eb_ratio: float,
        edges: tuple[int, ...],
        use_compile: bool,
    ) -> bytes:
        predictor = self._chunked_predictor(vmin, vmax)
        predictor.compile = bool(use_compile)
        ebs = _chunk_stage_ebs(values.shape, self.levels, self.anchor_stride,
                               self.anchor_block, eb, eb_ratio)
        payload = _compress_chunked(values[None, ...], ebs, self.radius,
                                    dtype.kind in "bi", predictor, edges,
                                    self.chunk_batch, overlap=self.overlap)
        return payload, int(predictor.chunk_batch)

    def _chunked_predictor(
        self,
        vmin: float,
        vmax: float,
        meta: dict[str, Any] | None = None,
    ) -> ChunkedGNNPredictor:
        levels = self.levels if meta is None else int(meta["levels"])
        anchor_stride = (self.anchor_stride if meta is None
                         else int(meta["anchor_stride"]))
        anchor_block = (self.anchor_block if meta is None
                        else int(meta["anchor_block"]))
        agg_level = self.agg_level if meta is None else _meta_agg_level(meta)
        predictor = ChunkedGNNPredictor(
            self.checkpoint_path,
            vmin,
            vmax,
            device=self.device,
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=anchor_block,
            agg_level=agg_level,
        )
        # encode: from the codec flag; decode: replay the stream's float path
        predictor.fp16 = (self.fp16 if meta is None
                          else bool(meta.get("fp16", False)))
        predictor.compile = (self.compile if meta is None
                             else bool(meta.get("compiled", False)))
        return predictor

    def _predictor(
        self,
        vmin: float,
        vmax: float,
        meta: dict[str, Any] | None = None,
    ) -> GNNPredictor:
        levels = self.levels if meta is None else int(meta["levels"])
        anchor_stride = self.anchor_stride if meta is None else int(meta["anchor_stride"])
        anchor_block = self.anchor_block if meta is None else int(meta["anchor_block"])
        max_radius = self.max_radius if meta is None else int(meta["max_radius"])
        agg_level = self.agg_level if meta is None else _meta_agg_level(meta)
        return GNNPredictor(
            self.checkpoint_path,
            vmin,
            vmax,
            tile_size=0,
            max_radius=max_radius,
            device=self.device,
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=anchor_block,
            agg_level=agg_level,
        )

    def _checkpoint_hash(self) -> bytes:
        import hashlib

        return hashlib.sha256(self.checkpoint_path.read_bytes()).digest()[:16]


GNNCodec = GNNCompressorCodec
