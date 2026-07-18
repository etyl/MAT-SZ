"""Microbenchmark the chunked GNN forward to see if batching helps, and project
the full-tensor encode time.

The encode is ~92% GNN forward, so this times one worst-case chunk's full stage
chain (start -> finest -> finalize) at each --chunk-batch, filling the model B dim
properly (which a small end-to-end synthetic can't, since colours don't fill).
ms/chunk vs B is the whole answer: flat => the GPU is already saturated at B=1 and
batching won't help; dropping => raise the batch.

    python scripts/profile_gnn.py --gnn-checkpoint CKPT \
        --levels 4 \
        --batches 1,2,4,8,16 --target-shape 119,128,128,128

Send the printed table back and I'll tell you which knob to turn.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import deepsz.gnn_predictor as gp  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_worst_chunk(ckpt, args, device):
    """Load the model and build the compact geometry of a worst-case interior
    chunk (all neighbours coded -> full referenced halo). Returns
    (model, geoms, chain, n_compact, d)."""
    d, model, _ = gp._load_inference_model(ckpt, torch, device)
    edge = args.anchor_stride
    edges = (edge,) * args.ndim
    grid = (5,) * args.ndim                    # >=5 so a 2*edge origin is interior
    shape = tuple(5 * edge for _ in range(args.ndim))
    cg = gp.build_chunk_geoms(edges, args.levels, edge, args.anchor_block,
                              torch, device)
    origin = np.array([2 * edge] * args.ndim, np.int64)
    coded = np.ones(int(np.prod(grid)), bool)  # worst case: every neighbour live
    frame = gp._CompactFrame(cg, origin, shape, edges, grid, coded, torch, device)
    return model, frame.geoms, cg.chain, frame.n_compact, d


def _make_wave(model, geoms, chain, N, ndim, d, B, device, fp16=False):
    """Return a callable that runs one full chunk's stage chain (start -> finest
    -> finalize) at batch B — the pure GNN forward, no quantize/host."""
    import contextlib

    def amp():
        if fp16 and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def one_wave():
        E = torch.zeros(B, N, ndim, d, device=device)
        ctx = None
        with amp():
            for j in range(1, len(chain)):
                gpv, ghv = geoms[chain[j - 1]], geoms[chain[j]]
                fvals = None if gpv is None else torch.zeros(B, gpv.M, device=device)
                (_v, _lb), E, ctx = gp.stage_forward(
                    model, E, gpv, ghv, fvals, torch, finalize_ctx=ctx, eb=0.01)
        return E
    return one_wave


def _bench_wave(model, geoms, chain, N, ndim, d, B, device, repeats, fp16=False):
    """Time one wave at batch B. ms/chunk vs B isolates whether the GPU is
    underused (ms/chunk drops with B) or saturated (flat)."""
    one_wave = _make_wave(model, geoms, chain, N, ndim, d, B, device, fp16)
    with torch.no_grad():
        one_wave()                              # warmup this B (autotune)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(); t = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            one_wave()
    _sync(); wave = (time.perf_counter() - t) / repeats
    peak = (torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else float("nan"))
    return wave, peak


def _profile_wave(model, geoms, chain, N, ndim, d, B, device, rows, fp16=False):
    """Operator-level trace of one wave: which ops (matmul/softmax/index/copy...)
    dominate GPU time -> what's worth optimizing."""
    from torch.profiler import ProfilerActivity, profile
    one_wave = _make_wave(model, geoms, chain, N, ndim, d, B, device, fp16)
    acts = [ProfilerActivity.CPU]
    cuda = device.type == "cuda"
    if cuda:
        acts.append(ProfilerActivity.CUDA)
    with torch.no_grad():
        one_wave(); _sync()                     # warmup
        with profile(activities=acts) as prof:
            one_wave(); _sync()
    sort = "self_cuda_time_total" if cuda else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=rows))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gnn-checkpoint", required=True)
    ap.add_argument("--ndim", type=int, default=4)
    ap.add_argument("--levels", type=int, default=4,
                    help="dyadic levels; anchor stride (= chunk edge) is 2**levels")
    ap.add_argument("--anchor-block", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model-frac", type=float, default=0.92,
                    help="fraction of encode time in the GNN forward (from the "
                         "end-to-end run); used to project full-tensor time")
    ap.add_argument("--batches", default="1,2,4,8,16",
                    help="comma list of --chunk-batch to sweep")
    ap.add_argument("--target-shape", default="119,128,128,128",
                    help="real tensor shape to project the full encode time for")
    ap.add_argument("--profile", action="store_true",
                    help="operator-level torch.profiler trace instead of the "
                         "batch sweep -> shows which ops dominate the forward")
    ap.add_argument("--profile-batch", type=int, default=1,
                    help="batch to trace under --profile")
    ap.add_argument("--profile-rows", type=int, default=30)
    ap.add_argument("--fp16", action="store_true",
                    help="run the message pass in fp16 autocast (cuda)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the embed pass (fuses the elementwise "
                         "ops; warmup absorbs the one-off compile cost)")
    args = ap.parse_args(argv)
    args.anchor_stride = 1 << args.levels  # stride is 2**levels, not a knob

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge = args.anchor_stride
    tgt = tuple(int(v) for v in args.target_shape.split(","))
    n_chunks = int(np.prod([-(-n // edge) for n in tgt]))

    if device.type != "cuda":
        print("WARNING: no GPU -> timings are ~meaningless for the V100 run")
    print(f"device={device}  edge={edge} levels={args.levels} ndim={args.ndim}")
    model, geoms, chain, N, d = _build_worst_chunk(args.gnn_checkpoint, args, device)
    if args.compile:
        # match the codec: compile embed once, dynamic over M. The warmup wave in
        # _bench_wave / the trace's own warmup absorbs the compilation cost.
        model.embed = torch.compile(model.embed, dynamic=True)
    M = max((g.M for g in geoms if g is not None), default=0)
    print(f"worst-case interior chunk: n_compact={N} finest_M={M} d={d} "
          f"stages={len(chain)}")
    print(f"target={tgt} = {n_chunks} chunks, model-frac={args.model_frac}\n")

    if args.profile:
        print(f"operator trace of one wave at batch={args.profile_batch}:\n")
        _profile_wave(model, geoms, chain, N, args.ndim, d, args.profile_batch,
                      device, args.profile_rows, args.fp16)
        return

    batches = [int(b) for b in args.batches.split(",")]
    hdr = "batch  wave(s)  ms/chunk  speedup  peakGB  proj-full"
    print(hdr); print("-" * len(hdr))
    base = None
    for B in batches:
        print(f"...batch={B}", flush=True)
        try:
            wave, peak = _bench_wave(model, geoms, chain, N, args.ndim, d, B,
                                     device, args.repeats, args.fp16)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"{B:>5}  OOM")
                continue
            raise
        per_chunk = wave / B
        base = base if base is not None else per_chunk
        # full encode ~ every chunk pays per_chunk of model time, / model-frac for
        # the host/anchor tail. Assumes batches fill (true on the 8^4 target grid).
        proj = n_chunks * per_chunk / max(args.model_frac, 1e-6)
        print(f"{B:>5}  {wave:7.3f}  {per_chunk*1e3:7.1f}  {base/per_chunk:6.2f}x"
              f"  {peak:5.2f}  {proj/60:6.1f}min")
    print("\nms/chunk = model forward per chunk. speedup vs first row.")
    print("Flat ms/chunk => GPU saturated at B=1, batching won't help; the model "
          "itself (stages*M*L) is the wall -> fewer levels / smaller d / prune "
          "directions. Dropping ms/chunk => raise --chunk-batch until peakGB nears "
          "VRAM or the speedup plateaus.")


if __name__ == "__main__":
    main()
