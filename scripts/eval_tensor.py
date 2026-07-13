"""Roundtrip-eval DeepSZ on an arbitrary tensor loaded from .npy or torch.

    python scripts/eval_tensor.py field.npy --eb 1e-3
    python scripts/eval_tensor.py weights.pt --eb 0.01 --rel

Same flags as `deepsz eval`; the only difference is the input is a raw tensor
(any dtype/shape the codec accepts, i.e. 2-D or 3-D with 1 or 3 channels)
instead of an image decoded by PIL.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Use this worktree's deepsz, not a stale pip-installed copy in site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsz.cli import add_common, build_predictor, run_compress
from deepsz.codec import decompress


def load_tensor(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".pt", ".pth")):
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, torch.Tensor):
            raise ValueError(f"{path} holds {type(obj).__name__}, expected a single tensor")
        return obj.detach().cpu().numpy()
    raise ValueError(f"unsupported extension: {path} (use .npy/.pt/.pth)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="tensor file (.npy or .pt/.pth)")
    add_common(ap)
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="gnn only: force chunk edge (multiple of anchor-stride); "
                         "0 = whole-tensor, omit = auto. Peak field/chunk is "
                         "(edge+2*anchor_stride)^ndim, so keep edge small and "
                         "anchor-stride << edge for large n-D tensors")
    ap.add_argument("--chunk-batch", type=int, default=None,
                    help="gnn only: cap how many chunks are coded together in the "
                         "model batch dim (omit = auto from encode-GPU memory). "
                         "The value is stored in the stream and replayed at decode, "
                         "so set it to what the smallest decode device can hold")
    ap.add_argument("--fp16", action="store_true",
                    help="gnn only: fp16 autocast on the GNN message pass (cuda; "
                         "~2x forward, readout stays fp32). May cost a little ratio "
                         "at small eb -- compare bits/value with and without")
    ap.add_argument("--overlap", action="store_true",
                    help="gnn chunked only: pack the per-stage rANS on a "
                         "background thread so it hides behind the next stage's "
                         "GPU forward. Output bytes are identical; encode-only")
    ap.add_argument("--compile", action="store_true",
                    help="gnn only: torch.compile the message-pass embed (fuses "
                         "the elementwise ~40%%; one-off compile cost on first "
                         "chunk). Same float path replayed at decode")
    args = ap.parse_args(argv)

    arr = load_tensor(args.input)
    orig_bytes = arr.nbytes
    if arr.dtype == np.float64:  # codec pipeline is float32; store as such
        print("note: float64 input cast to float32 (float32-precision reconstruction)")
        arr = arr.astype(np.float32)
    eb = args.eb
    if args.rel:
        eb = args.eb * max(float(arr.max()) - float(arr.min()), 1.0)

    if args.predictor == "gnn":
        # GNNCompressorCodec auto-chunks large tensors; the codec.compress path
        # allocates a dense embedding field over the whole tensor and OOMs.
        os.environ.setdefault("DEEPSZ_PROGRESS", "1")  # per-chunk progress to stderr
        from deepsz.gnn_codec import GNNCompressorCodec
        codec = GNNCompressorCodec(
            args.gnn_checkpoint, error_bound=eb, levels=args.levels,
            anchor_stride=args.anchor_stride, anchor_block=args.anchor_block,
            agg_level=args.agg_level,
            radius=args.radius, zstd_level=args.zstd_level,
            eb_ratio=args.eb_ratio,
            tune=args.tune if args.tune in ("fast", "size") else "fast",
            chunk_size=args.chunk_size, chunk_batch=args.chunk_batch,
            fp16=args.fp16, compile=args.compile, overlap=args.overlap)
        t0 = time.time()
        stream = codec.compress(arr)
        t_comp = time.time() - t0
        stats = {"outliers": 0}  # ponytail: codec doesn't surface outlier count
        t0 = time.time()
        rec = codec.uncompress(stream).numpy()
        t_dec = time.time() - t0
    else:
        t0 = time.time()
        stream, stats = run_compress(arr, args)
        t_comp = time.time() - t0

        t0 = time.time()
        rec = decompress(stream, lambda hdr: build_predictor(args, hdr))
        t_dec = time.time() - t0

    a = arr.astype(np.float64)
    r = rec.reshape(arr.shape).astype(np.float64)
    max_err = float(np.abs(a - r).max())
    mse = float(np.mean((a - r) ** 2))
    peak = 255.0 if arr.dtype == np.uint8 else max(float(arr.max()) - float(arr.min()), 1e-12)
    psnr = 10 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")
    bpv = 8 * len(stream) / arr.size

    bound_ok = max_err <= eb
    ratio = orig_bytes / len(stream)
    print(f"tensor {args.input} {arr.shape} {arr.dtype}, eb={eb}")
    print(f"  compressed:  {len(stream)} bytes, ratio {ratio:.2f}, {bpv:.3f} bits/value")
    print(f"  PSNR:        {psnr:.2f} dB")
    print(f"  max abs err: {max_err} <= eb: {'PASS' if bound_ok else 'FAIL'}")
    print(f"  outliers:    {stats['outliers']} ({100*stats['outliers']/arr.size:.3f}%)")
    print(f"  time:        compress {t_comp:.1f}s, decompress {t_dec:.1f}s")
    if not bound_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
