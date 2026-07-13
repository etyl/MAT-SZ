"""Detailed end-to-end profile of the *chunked* GNN codec on an n-D tensor.

Runs the real `GNNCompressorCodec` closed-loop compress (and, by default,
decompress) on a `.npy`/`.pt` tensor with the same flags as `eval_tensor.sh`,
and prints two CUDA-synced breakdowns:

  1. phase table  -- predict_wave_stage (GPU forward) vs start_wave (halo) vs
     quantize / pack_stage (rANS) / dequantize / anchors ...
  2. forward table -- inside the forward: model.embed / _line_messages /
     dir / bidir / rope / line_pool / head_of / finalize.

It uses monkeypatched perf-counter timers with a torch.cuda.synchronize() on
each call (real wall-time attribution) instead of torch.profiler, which
segfaults on pre-Volta GPUs (TITAN Xp, cap 6.1). Warm-up runs first, then the
timed run; nested phases overlap, so read each table as a tree.

    python scripts/profile_chunked_tensor.py data/rti_normal.npy \
        --gnn-checkpoint checkpoints/d32.pt --eb 0.01 --levels 4 \
        --anchor-stride 16 --chunk-size 16 --anchor-block 1 --agg-level 2 \
        --chunk-batch 1 --fp16

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
    ap.add_argument("--agg-level", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="chunk edge (multiple of anchor-stride); 0 = whole-tensor")
    ap.add_argument("--chunk-batch", type=int, default=None)
    ap.add_argument("--radius", type=int, default=1 << 15)
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--eb-ratio", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--no-decode", action="store_true",
                    help="profile compress only (skip the decode pass)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    # M-tiling off by default so the profile matches eval_tensor.sh; caller may
    # override in the environment before launching.
    os.environ.setdefault("DEEPSZ_M_TILE", str(1 << 30))
    os.environ["DEEPSZ_PROGRESS"] = "0"    # bars would swamp the tables

    import torch
    import deepsz.gnn_predictor as gp
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
            agg_level=args.agg_level, radius=args.radius,
            zstd_level=args.zstd_level, eb_ratio=args.eb_ratio, tune="fast",
            chunk_size=args.chunk_size, chunk_batch=args.chunk_batch,
            fp16=args.fp16, compile=args.compile, overlap=args.overlap,
            device=args.device)

    # --- timing harness: accumulate wall time per patched call ----------------
    acc: dict[str, float] = {}
    cnt: dict[str, int] = {}

    def rec(name, dt):
        acc[name] = acc.get(name, 0.0) + dt
        cnt[name] = cnt.get(name, 0) + 1

    def wrap(obj, name, label, do_sync):
        orig = getattr(obj, name)

        def w(*a, **k):
            if do_sync:
                sync()
            t0 = time.perf_counter()
            r = orig(*a, **k)
            if do_sync:
                sync()
            rec(label, time.perf_counter() - t0)
            return r
        setattr(obj, name, w)

    # Phase-level (codec) timers: module-level CPU fns + predictor GPU methods.
    import deepsz.gnn_codec as gc
    for fn in ("quantize", "dequantize", "pack_stage", "unpack_stage",
               "build_laplace_tables", "scale_to_level", "_code_anchor_stage",
               "_decode_anchor_stage"):
        if hasattr(gc, fn):
            wrap(gc, fn, fn, do_sync=False)
    for m in ("predict_wave_stage", "start_wave", "finish_wave",
              "anchor_coarse", "begin"):
        wrap(ChunkedGNNPredictor, m, m, do_sync=is_cuda)

    # Forward-sublayer timers: wrap each model instance as it is built.
    orig_build = gp.build_model

    def build_timed(*a, **k):
        m = orig_build(*a, **k)
        for meth in ("embed", "_line_messages", "_embed_block", "head_of",
                     "finalize"):
            if hasattr(m, meth):
                wrap(m, meth, "fwd." + meth, do_sync=is_cuda)
        for sub in ("dir", "bidir", "rope", "line_pool"):
            mod = getattr(m, sub, None)
            if mod is not None and hasattr(mod, "forward"):
                wrap(mod, "forward", "fwd.mlp." + sub, do_sync=is_cuda)
        return m
    gp.build_model = build_timed

    # --- warm up (compile caches / allocator), then reset and time ------------
    c = make()
    s = c.compress(arr)
    if not args.no_decode:
        c.uncompress(s)
    sync()
    acc.clear(); cnt.clear()

    c = make()
    sync(); t0 = time.perf_counter()
    stream = c.compress(arr)
    sync(); t_comp = time.perf_counter() - t0

    t_dec = None
    if not args.no_decode:
        sync(); t0 = time.perf_counter()
        c.uncompress(stream)
        sync(); t_dec = time.perf_counter() - t0

    total = t_comp + (t_dec or 0.0)
    ratio = arr.nbytes / len(stream)
    print(f"\ntensor {args.input} {arr.shape} {arr.dtype}  eb={eb}")
    print(f"chunk-size={args.chunk_size} chunk-batch={args.chunk_batch} "
          f"agg-level={args.agg_level} fp16={args.fp16} compile={args.compile} "
          f"overlap={args.overlap} device={args.device}")
    print(f"compress {t_comp:.2f}s" + (f"  decompress {t_dec:.2f}s" if t_dec else "")
          + f"   {len(stream)} bytes  ratio {ratio:.2f}")

    def table(title, keys):
        rows = [(k, cnt[k], acc[k]) for k in keys if k in acc]
        rows.sort(key=lambda r: -r[2])
        if not rows:
            return
        print(f"\n{title}  (%% of total {total:.2f}s; nested phases overlap)")
        print(f"  {'phase':<26}{'calls':>8}{'total_s':>10}{'%':>7}{'ms/call':>10}")
        for k, n, t in rows:
            print(f"  {k:<26}{n:>8}{t:>10.3f}{100*t/total:>7.1f}{1000*t/n:>10.3f}")

    phase_keys = ["predict_wave_stage", "start_wave", "finish_wave",
                  "anchor_coarse", "begin", "pack_stage", "unpack_stage",
                  "quantize", "dequantize", "build_laplace_tables",
                  "scale_to_level", "_code_anchor_stage", "_decode_anchor_stage"]
    fwd_keys = ["fwd.embed", "fwd._embed_block", "fwd._line_messages",
                "fwd.head_of", "fwd.finalize", "fwd.mlp.dir", "fwd.mlp.bidir",
                "fwd.mlp.rope", "fwd.mlp.line_pool"]
    table("== phase breakdown ==", phase_keys)
    table("== forward sublayers ==", fwd_keys)


if __name__ == "__main__":
    main()
