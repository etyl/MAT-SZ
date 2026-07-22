"""Detailed end-to-end profile of the *chunked* GNN codec on an n-D tensor.

Runs the real `GNNCompressorCodec` closed-loop compress on a `.npy`/`.pt`
tensor with the same flags as `eval_tensor.sh`. It warms up over the first
`--warmup` chunk-encoding steps, then profiles a *single* chunk step (one
model-wave: start_wave..finish_wave) under `torch.profiler` and aborts the rest.

It prints the kineto `key_averages` table (kernels + record_function ranges,
sorted by CUDA time) and writes a chrome trace. Unlike the old monkeypatched
per-`forward` timers, kineto observes the CUDA stream, so it does NOT force
graph breaks: with `--compile` the embed pass genuinely fuses and the table
shows the real fused Triton kernels instead of an un-fused per-op breakdown.
The coarse codec phases (predict_wave_stage / start_wave / quantize / ...) are
tagged with `record_function` so they appear as named ranges alongside the
kernels; they live outside the compiled `embed` graph, so labelling them is
safe. After fusion the per-sublayer (`dir`/`bidir`/`rope`) names are gone by
design — use `with_stack` (on by default) + the chrome trace to map a fused
kernel back to its Python source line.

Note: `torch.profiler` (kineto) can segfault on pre-Volta GPUs (TITAN Xp,
cap 6.1). This targets Volta+ (V100), which is also what `--compile`/Triton
needs anyway.

    python scripts/profile_chunked_tensor.py data/rti_normal.npy \
        --gnn-checkpoint checkpoints/d32.pt --eb 0.01 --levels 4 \
        --anchor-stride 16 --chunk-size 16 --anchor-block 1 --agg-level 2 \
        --chunk-batch 1 --fp16 --compile

Same knobs the codec takes (--fp16, --compile, --overlap, --chunk-size, ...).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_tensor(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".pt", ".pth")):
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj.detach().cpu().numpy()
    raise ValueError(f"unsupported extension: {path} (use .npy/.pt/.pth)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="tensor file (.npy or .pt/.pth)")
    ap.add_argument("--gnn-checkpoint", required=True)
    ap.add_argument("--eb", type=float, default=0.01)
    ap.add_argument("--rel", action="store_true",
                    help="scale eb by the value range (max-min)")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--anchor-stride", type=int, default=16)
    ap.add_argument("--anchor-block", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="chunk edge (multiple of anchor-stride); 0 = whole-tensor")
    ap.add_argument("--chunk-batch", type=int, default=None)
    ap.add_argument("--radius", type=int, default=1 << 15)
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--eb-ratio", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--warmup", type=int, default=10,
                    help="chunk-encoding steps to warm up on before the timed step")
    ap.add_argument("--trace", default="chunk_step_trace.json",
                    help="chrome trace output path for the profiled step")
    ap.add_argument("--row-limit", type=int, default=30,
                    help="rows in the key_averages table")
    ap.add_argument("--no-stack", action="store_true",
                    help="disable with_stack (lower overhead, no source attribution)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    # M-tiling off by default so the profile matches eval_tensor.sh; caller may
    # override in the environment before launching.
    os.environ.setdefault("DEEPSZ_M_TILE", str(1 << 30))
    os.environ["DEEPSZ_PROGRESS"] = "0"    # bars would swamp the tables

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from deepsz.gnn_codec import GNNCompressorCodec
    from deepsz.gnn_predictor import ChunkedGNNPredictor

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = torch.device(args.device).type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    arr = load_tensor(args.input)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    eb = args.eb * max(float(arr.max()) - float(arr.min()), 1.0) if args.rel else args.eb

    def make():
        return GNNCompressorCodec(
            args.gnn_checkpoint, error_bound=eb, levels=args.levels,
            anchor_stride=args.anchor_stride, anchor_block=args.anchor_block,
            radius=args.radius,
            zstd_level=args.zstd_level, eb_ratio=args.eb_ratio, tune="fast",
            chunk_size=args.chunk_size, chunk_batch=args.chunk_batch,
            fp16=args.fp16, compile=args.compile, overlap=args.overlap,
            device=args.device)

    # Tag the coarse codec phases with record_function so they show up as named
    # ranges in the trace/table. These call sites are *outside* the compiled
    # embed graph, so wrapping them does not break fusion (the whole point of
    # moving to kineto). We deliberately do NOT wrap dir/bidir/rope/line_pool —
    # those live inside embed and must stay fusible.
    def label(obj, name):
        orig = getattr(obj, name)

        def w(*a, **k):
            with record_function(name):
                return orig(*a, **k)
        setattr(obj, name, w)

    import deepsz.gnn_codec as gc
    for fn in ("quantize", "dequantize", "pack_stage", "unpack_stage",
               "build_laplace_tables", "scale_to_level", "_code_anchor_stage",
               "_decode_anchor_stage"):
        if hasattr(gc, fn):
            label(gc, fn)
    for m in ("predict_wave_stage", "start_wave", "finish_wave",
              "anchor_coarse", "begin"):
        label(ChunkedGNNPredictor, m)

    # Step-level warm-up + single-step profiling. A step is one model-wave
    # (start_wave..finish_wave) = one chunk at chunk-batch 1. Warm up over the
    # first `warmup` steps (this also warms the CUDA context and the compiled
    # graph), start kineto on the (warmup+1)-th step, stop + abort after it.
    class _Stop(Exception):
        pass

    prof = profile(
        activities=[ProfilerActivity.CPU]
        + ([ProfilerActivity.CUDA] if is_cuda else []),
        with_stack=not args.no_stack, record_shapes=False)

    step = [0]
    timed = {}
    timed_start = ChunkedGNNPredictor.start_wave
    timed_finish = ChunkedGNNPredictor.finish_wave

    def start_ctrl(*a, **k):
        step[0] += 1
        if step[0] == args.warmup + 1:            # entering the timed step
            sync(); prof.start(); timed["t0"] = time.perf_counter()
        return timed_start(*a, **k)

    def finish_ctrl(*a, **k):
        r = timed_finish(*a, **k)
        if step[0] == args.warmup + 1:            # timed step done -> stop
            sync(); timed["t_step"] = time.perf_counter() - timed["t0"]
            prof.stop()
            raise _Stop
        return r

    ChunkedGNNPredictor.start_wave = start_ctrl
    ChunkedGNNPredictor.finish_wave = finish_ctrl

    # One compress: the anchor pass + `warmup` chunk steps warm the caches, the
    # (warmup+1)-th step is profiled, then _Stop aborts before the rest encode.
    c = make()
    try:
        c.compress(arr)
    except _Stop:
        pass
    if "t_step" not in timed:
        print(f"tensor has only {step[0]} chunk-steps; need > --warmup "
              f"({args.warmup})", file=sys.stderr)
        return

    total = timed["t_step"]
    print(f"\ntensor {args.input} {arr.shape} {arr.dtype}  eb={eb}")
    print(f"chunk-size={args.chunk_size} chunk-batch={args.chunk_batch} "
          f"fp16={args.fp16} compile={args.compile} "
          f"overlap={args.overlap} device={args.device}")
    print(f"single chunk-step (after {args.warmup} warm-up steps): "
          f"{1000*total:.2f}ms  (wall; kineto adds overhead below)")

    sort_key = "cuda_time_total" if is_cuda else "cpu_time_total"
    print()
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.row_limit))

    trace_path = os.path.abspath(args.trace)
    prof.export_chrome_trace(trace_path)
    print(f"\nchrome trace -> {trace_path}  (open in chrome://tracing or "
          f"https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
