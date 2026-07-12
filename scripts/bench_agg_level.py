"""Measure the inference speed-up of the neighbourhood aggregation level on a
high-dimensional (default 4-D) tensor.

The GNN visits, per point, one line for every direction in {-1,0,1}^ndim (up to
sign) -- ``(3^ndim - 1)/2`` lines, which is the ``L`` dimension of the message
tensor and the dominant cost in high dimensions (40 lines in 4-D, 121 in 5-D).
The aggregation level caps a line's L1 length (its number of non-zero
components): level 1 keeps only axis-aligned direct neighbours, level 2 adds the
2-axis diagonals, ... level ndim (== full) keeps everything. Fewer lines ->
proportionally less message-pass work.

This drives the real closed-loop predictor (encoder feedback path) at each
level, reports latency + speed-up vs. the full neighbourhood, and roundtrips the
codec once per level to confirm the error bound still holds (encoder and decoder
use the same level, so prediction stays bit-identical).

    python scripts/bench_agg_level.py --shape 32 32 32 32 --eb 1e-2

With a real checkpoint the numbers are representative; with none, a random v5
model is used (inference *time* is a function of the architecture and shapes,
not the weights, so the speed-up measurement is valid either way).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ensure_rans_backend():
    """Real rANS if available, else a bit-exact raw stand-in (sizes meaningless,
    correctness/timing intact) so the roundtrip check runs on constriction-less
    hosts. Mirrors scripts/check_chunked_memory.py."""
    try:
        import constriction  # noqa: F401
        return "constriction"
    except ImportError:
        import deepsz.bitstream as bitstream
        import deepsz.rans as rans

        def fake_encode(codes, levels64, tables):
            return np.asarray(codes, np.uint32).ravel().astype("<u4").tobytes()

        def fake_decode(blob, levels64, tables):
            out = np.frombuffer(blob, dtype="<u4").astype(np.uint32)
            if len(out) != len(np.asarray(levels64).ravel()):
                raise ValueError("fake rANS length mismatch")
            return out

        rans.rans_encode = bitstream.rans_encode = fake_encode
        rans.rans_decode = bitstream.rans_decode = fake_decode
        return "raw stand-in (no constriction; payload sizes not meaningful)"


def synth_field(shape, seed):
    """Smooth-ish separable field + a little noise (compressible, like a real
    scientific field) so the predictor has structure to exploit."""
    rng = np.random.RandomState(seed)
    x = rng.rand(*shape).astype(np.float32) * 0.05
    for k, s in enumerate(shape):
        wave = np.cos(np.linspace(0, 4 * np.pi, s, dtype=np.float32))
        x += wave.reshape([-1 if i == k else 1 for i in range(len(shape))])
    return x


def closed_loop_ms(values, predictor, masks, ebs, radius, device):
    """One encoder prediction/quantization feedback sweep; returns wall-ms of the
    GNN predict calls only (stage 0 anchors are coded directly, not predicted)."""
    import torch

    from deepsz.quantizer import dequantize, quantize

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    recon = np.zeros_like(values)
    known = np.zeros(values.shape[1:], dtype=bool)
    sync()
    t0 = time.perf_counter()
    for stage_idx, (pos, eb) in enumerate(zip(masks, ebs)):
        n = int(pos.sum())
        if not n:
            continue
        if stage_idx == 0:
            pred = np.zeros((values.shape[0], n), np.float32)
        else:
            pred, _scale = predictor.predict(recon, known, pos, eb=eb)
        codes, outliers = quantize(values[:, pos], pred, eb, radius,
                                   round_output=False)
        recon[:, pos] = dequantize(pred, codes, outliers, eb, radius).reshape(
            values.shape[0], n)
        known |= pos
    sync()
    return (time.perf_counter() - t0) * 1e3


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", type=int, nargs="+", default=(32, 32, 32, 32))
    ap.add_argument("--eb", type=float, default=1e-2)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--anchor-stride", type=int, default=16)
    ap.add_argument("--anchor-block", type=int, default=1)
    ap.add_argument("--radius", type=int, default=1 << 15)
    ap.add_argument("--d", type=int, default=32, help="model width (random ckpt)")
    ap.add_argument("--checkpoint", default=None,
                    help="real v5 checkpoint (default: random weights)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--levels-to-test", type=int, nargs="+", default=None,
                    help="aggregation levels to sweep (default 1..ndim)")
    ap.add_argument("--m-tile", type=int, default=16384,
                    help="cap the per-stage query tile so the whole-tensor path "
                         "fits in RAM (set before importing deepsz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-roundtrip", action="store_true",
                    help="skip the codec error-bound check (timing only)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # _M_TILE is read from the environment at import time; set it first so the
    # whole-tensor path tiles the (B, L, M, K, d) message buffers into RAM.
    os.environ["DEEPSZ_M_TILE"] = str(args.m_tile)

    backend = _ensure_rans_backend()
    import torch

    from deepsz.gnn_predictor import (CKPT_VERSION, GNNPredictor, build_model,
                                      half_directions)
    from deepsz.levels import stage_ebs, stage_masks

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    shape = tuple(args.shape)
    ndim = len(shape)
    torch.manual_seed(args.seed)

    ckpt = args.checkpoint
    if ckpt is None:
        model = build_model(d=args.d).eval()
        ckpt = str(Path(tempfile.mkdtemp()) / "gnn_random.pt")
        torch.save({"d": args.d, "state_dict": model.state_dict(),
                    "version": CKPT_VERSION}, ckpt)

    x = synth_field(shape, args.seed)
    values = x[None, ...].astype(np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    masks = stage_masks(shape, args.levels, args.anchor_stride, args.anchor_block)
    ebs = stage_ebs(shape, args.levels, args.anchor_stride, args.anchor_block,
                    args.eb, 1.0)

    tested = args.levels_to_test or list(range(1, ndim + 1))
    full_lines = len(half_directions(ndim))
    print(f"tensor {shape} ({x.size/1e6:.2f}M points) | ndim={ndim} | device={device} "
          f"| eb={args.eb} | rANS: {backend}")
    print(f"full neighbourhood = {full_lines} lines per point "
          f"((3^{ndim}-1)/2); m_tile={args.m_tile}\n")

    rows = []
    for lvl in tested:
        n_lines = len(half_directions(ndim, lvl))
        predictor = GNNPredictor(
            ckpt, vmin, vmax, tile_size=0, max_radius=64, device=device,
            levels=args.levels, anchor_stride=args.anchor_stride,
            anchor_block=args.anchor_block, agg_level=lvl)
        for _ in range(args.warmup):
            closed_loop_ms(values, predictor, masks, ebs, args.radius, device)
        samples = [closed_loop_ms(values, predictor, masks, ebs, args.radius, device)
                   for _ in range(args.repeats)]
        rows.append((lvl, n_lines, statistics.median(samples), min(samples)))

    # Reference: the full neighbourhood (agg_level=None) is the current default.
    ref_predictor = GNNPredictor(
        ckpt, vmin, vmax, tile_size=0, max_radius=64, device=device,
        levels=args.levels, anchor_stride=args.anchor_stride,
        anchor_block=args.anchor_block, agg_level=None)
    for _ in range(args.warmup):
        closed_loop_ms(values, ref_predictor, masks, ebs, args.radius, device)
    ref_samples = [closed_loop_ms(values, ref_predictor, masks, ebs, args.radius, device)
                   for _ in range(args.repeats)]
    ref_ms = statistics.median(ref_samples)

    print(f"{'agg_level':>10} {'lines':>7} {'lines%':>7} {'infer p50':>12} "
          f"{'infer min':>12} {'speedup':>9}")
    for lvl, n_lines, med, mn in rows:
        print(f"{lvl:>10} {n_lines:>7} {100*n_lines/full_lines:>6.0f}% "
              f"{med:>10.1f}ms {mn:>10.1f}ms {ref_ms/med:>8.2f}x")
    print(f"{'full/None':>10} {full_lines:>7} {'100%':>7} "
          f"{ref_ms:>10.1f}ms {min(ref_samples):>10.1f}ms {1.0:>8.2f}x")

    if args.no_roundtrip:
        return

    # Correctness: encode + decode once per level with the tensor codec and check
    # the error bound. Encoder and decoder share agg_level (frozen in the stream),
    # so the reconstruction must satisfy |x - recon| <= eb at every level.
    print("\nCodec roundtrip error-bound check (whole-tensor path):")
    from deepsz.gnn_codec import GNNCompressorCodec

    for lvl in tested + [None]:
        codec = GNNCompressorCodec(
            ckpt, error_bound=args.eb, levels=args.levels,
            anchor_stride=args.anchor_stride, anchor_block=args.anchor_block,
            agg_level=lvl, radius=args.radius, chunk_size=0,
            strict_checkpoint=False, device=device)
        stream = codec.compress(x)
        rec = codec.uncompress(stream).numpy().reshape(x.shape)
        max_err = float(np.abs(x.astype(np.float64) - rec.astype(np.float64)).max())
        tag = "full" if lvl is None else f"level {lvl}"
        print(f"  {tag:>8}: max|err| = {max_err:.3e}  "
              f"({'PASS' if max_err <= args.eb + 1e-6 else 'FAIL'} <= {args.eb})")


if __name__ == "__main__":
    main()
