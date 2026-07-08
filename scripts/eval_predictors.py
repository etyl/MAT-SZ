"""Evaluate DeepSZ predictors (GNN vs. interpolation) against SZ3 on Kodak.

Runs each method through the same closed-loop codec (identical quantizer +
Huffman/zstd stage) so the comparison isolates the predictor, and reports
bit-rate (bpp), PSNR, and error-bound compliance per image / error bound.

Usage:
    python scripts/eval_predictors.py --data data/kodak \
        --gnn-checkpoint data/gnn_predictor.pt --eb 1 2 4
    python scripts/eval_predictors.py --data data/kodak --methods interp sz3
    python scripts/eval_predictors.py --data data/kodak --csv results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepsz.baselines import sz3_roundtrip
from deepsz.bitstream import FLAG_CUBIC, FLAG_GNN, FLAG_INTERP, FLAG_MOCK, Header
from deepsz.codec import compress, decompress
from deepsz.predictor import InterpPredictor, MockPredictor

METHODS = ("gnn", "interp", "interp-linear", "sz3")


def default_error_bounds() -> list[float]:
    """Default ABS bounds for images loaded in normalized [0, 1] units."""
    return [1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0]


def load_image(path: Path) -> np.ndarray:
    """Grayscale, [0,1]-normalised float32 (2D)."""
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, np.float32) / 255.0


def make_predictor(method: str, img: np.ndarray, args):
    """Encoder-side predictor for a DeepSZ closed-loop method."""
    if method == "gnn":
        from deepsz.gnn_predictor import GNNPredictor
        return GNNPredictor(args.gnn_checkpoint, float(img.min()), float(img.max()),
                            tile_size=args.gnn_tile, max_radius=args.max_radius,
                            device=args.device, levels=args.levels,
                            anchor_stride=args.anchor_stride,
                            anchor_block=args.anchor_block)
    order = "linear" if method == "interp-linear" else "cubic"
    return InterpPredictor(args.interp_tile, order, args.levels,
                           args.anchor_stride, args.anchor_block)


def build_predictor_for_decompress(method: str, hdr: Header, args):
    """Decoder-side predictor; params come from the stream header."""
    if hdr.flags & FLAG_GNN:
        from deepsz.gnn_predictor import GNNPredictor
        return GNNPredictor(args.gnn_checkpoint, hdr.vmin, hdr.vmax,
                            tile_size=hdr.tile_size, max_radius=args.max_radius,
                            device=args.device, levels=hdr.levels,
                            anchor_stride=hdr.anchor_stride,
                            anchor_block=hdr.anchor_block)
    if hdr.flags & FLAG_MOCK:
        return MockPredictor(hdr.tile_size)
    return InterpPredictor(hdr.tile_size, "cubic" if hdr.flags & FLAG_CUBIC else "linear",
                           hdr.levels, hdr.anchor_stride, hdr.anchor_block)


def _quality(img: np.ndarray, rec: np.ndarray, n_bytes: int, eb: float) -> dict:
    rec2d = rec if rec.ndim == img.ndim else rec[..., None]
    diff = img.astype(np.float64) - rec2d.astype(np.float64)
    max_err = float(np.abs(diff).max())
    mse = float(np.mean(diff ** 2))
    peak = 255.0 if img.dtype == np.uint8 else max(float(img.max()) - float(img.min()), 1.0)
    psnr = 10 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")
    bpp = 8 * n_bytes / (img.shape[0] * img.shape[1])
    return dict(n_bytes=n_bytes, bpp=bpp, psnr=psnr, max_err=max_err,
                bound_ok=max_err <= eb)


def eval_deepsz(img: np.ndarray, eb: float, method: str, args) -> dict:
    pred = make_predictor(method, img, args)
    t0 = time.time()
    stream, _ = compress(img, eb, pred, levels=args.levels,
                         anchor_stride=args.anchor_stride,
                         anchor_block=args.anchor_block, radius=args.radius,
                         seed=args.seed, zstd_level=args.zstd_level,
                         eb_ratio=args.eb_ratio,
                         tune=args.tune,
                         tune_size_slack=args.tune_size_slack)
    t_comp = time.time() - t0
    del pred  # drop the encoder-side embedding field before decode builds its own
    t0 = time.time()
    rec = decompress(stream, lambda hdr: build_predictor_for_decompress(method, hdr, args))
    t_dec = time.time() - t0
    r = _quality(img, rec, len(stream), eb)
    r.update(t_comp=t_comp, t_dec=t_dec)
    return r


def eval_sz3(img: np.ndarray, eb: float) -> dict | None:
    t0 = time.time()
    result = sz3_roundtrip(img, eb)
    t_comp = time.time() - t0
    if result is None:
        return None
    n_bytes, rec = result
    r = _quality(img, rec, n_bytes, eb)
    r.update(t_comp=t_comp, t_dec=0.0)
    return r


def fmt(v, spec):
    return "—" if (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def print_tables(rows: list[dict], methods: list[str]) -> None:
    ebs = sorted({r["eb"] for r in rows})

    print("\n=== Per-image results ===")
    hdr = (f"{'image':<12} {'eb':>6} {'method':<13} "
           f"{'bpp':>7} {'PSNR':>8} {'maxErr':>9} {'ok':>4} {'tComp':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ok = "PASS" if r["bound_ok"] else "FAIL"
        print(f"{r['image']:<12} {r['eb']:>6g} {r['method']:<13} "
              f"{r['bpp']:>7.3f} {r['psnr']:>8.2f} {r['max_err']:>9.5f} "
              f"{ok:>4} {r['t_comp']:>7.2f}")

    print("\n=== Averages over dataset (per method / eb) ===")
    hdr2 = (f"{'method':<13} {'eb':>6} {'bpp':>7} {'PSNR':>8} "
            f"{'PASS%':>6} {'tComp':>7} {'vs sz3 bpp':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    # sz3 average bpp per eb for the relative column
    sz3_bpp = {}
    for eb in ebs:
        sub = [r for r in rows if r["eb"] == eb and r["method"] == "sz3"]
        sz3_bpp[eb] = np.mean([r["bpp"] for r in sub]) if sub else float("nan")
    for method in methods:
        for eb in ebs:
            sub = [r for r in rows if r["eb"] == eb and r["method"] == method]
            if not sub:
                continue
            avg_bpp = np.mean([r["bpp"] for r in sub])
            avg_psnr = np.mean([r["psnr"] for r in sub if np.isfinite(r["psnr"])])
            pass_pct = 100 * sum(r["bound_ok"] for r in sub) / len(sub)
            avg_tcomp = np.mean([r["t_comp"] for r in sub])
            ref = sz3_bpp.get(eb, float("nan"))
            rel = 100 * (avg_bpp / ref - 1) if ref and np.isfinite(ref) else float("nan")
            rel_str = "—" if method == "sz3" else fmt(rel, "+10.1f")
            print(f"{method:<13} {eb:>6g} {avg_bpp:>7.3f} {avg_psnr:>8.2f} "
                  f"{pass_pct:>6.1f} {avg_tcomp:>7.2f} {rel_str:>11}")


def _rd_points(rows: list[dict], method: str) -> tuple[list[float], list[float], list[float]]:
    """Dataset-averaged (bpp, PSNR) per error bound for one method, sorted by
    bpp so the polyline runs low-rate -> high-rate."""
    ebs = sorted({r["eb"] for r in rows if r["method"] == method})
    pts = []
    for eb in ebs:
        sub = [r for r in rows if r["method"] == method and r["eb"] == eb]
        if not sub:
            continue
        bpp = float(np.mean([r["bpp"] for r in sub]))
        psnr = float(np.mean([r["psnr"] for r in sub if np.isfinite(r["psnr"])]))
        pts.append((bpp, psnr, eb))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def plot_rd(rows: list[dict], methods: list[str], path: str) -> None:
    """Rate-distortion curves (bpp vs PSNR), one polyline per method, points
    annotated with their error bound. Saved to ``path``."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping RD plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        bpp, psnr, ebs = _rd_points(rows, method)
        if not bpp:
            continue
        ax.plot(bpp, psnr, marker="o", label=method)
        for x, y, eb in zip(bpp, psnr, ebs):
            ax.annotate(f"eb={eb:g}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.set_xlabel("rate (bits per pixel)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-distortion on Kodak")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nRD curves saved to {path}")


def save_csv(rows: list[dict], path: str) -> None:
    import csv
    fields = ["image", "method", "eb", "n_bytes", "bpp", "psnr", "max_err",
              "bound_ok", "t_comp", "t_dec"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="directory of Kodak images (kodim*.png)")
    ap.add_argument("--gnn-checkpoint", default=str(ROOT / "data" / "gnn_predictor.pt"),
                    help="GNN checkpoint (.pt) for the gnn method")
    ap.add_argument("--methods", nargs="+", default=["gnn", "interp", "sz3"],
                    choices=METHODS, help="predictors to evaluate")
    ap.add_argument("--eb", type=float, nargs="+",
                    default=default_error_bounds(),
                    help="absolute error bounds to sweep in the normalized "
                         "[0,1] image units (defaults: 1, 2, 4 gray levels)")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--anchor-stride", type=int, default=16)
    ap.add_argument("--anchor-block", type=int, default=1)
    ap.add_argument("--radius", type=int, default=1 << 15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--eb-ratio", type=float, default=None,
                    help="per-level eb decay (coarse tighter); default: interp "
                         "uses --tune, 1.0 forces flat classic SZ")
    ap.add_argument("--tune", choices=("fast", "size", "rd"), default="fast",
                    help="auto search when --eb-ratio is omitted: fast=one "
                         "candidate, size=min bytes, rd=lowest SSE within slack")
    ap.add_argument("--tune-size-slack", type=float, default=1.05,
                    help="for --tune rd, choose the lowest-SSE candidate within "
                         "this factor of the smallest stream")
    ap.add_argument("--gnn-tile", type=int, default=64,
                    help="tile size for the GNN predictor")
    ap.add_argument("--interp-tile", type=int, default=512,
                    help="region/tile size for the interp predictor")
    ap.add_argument("--max-radius", type=int, default=64,
                    help="GNN neighbour search radius")
    ap.add_argument("--device", default="cpu", help="torch device for the GNN")
    ap.add_argument("--images", nargs="*", default=None,
                    help="specific image filenames (default: all in --data)")
    ap.add_argument("--csv", default="eval.csv",
                    help="per-image CSV filename (bare name -> run dir)")
    ap.add_argument("--plot", nargs="?", const="eval_rd.png", default="eval_rd.png",
                    help="RD-curve PNG filename (bare name -> run dir)")
    ap.add_argument("--no-plot", action="store_true", help="disable the RD plot")
    args = ap.parse_args()

    # per-run dir: outputs/eval/<date>-<config-hash>/ holds the CSV, RD plot,
    # and a config.json snapshot, mirroring train_gnn's runs/ layout.
    import hashlib
    import json
    cfg_hash = hashlib.sha1(repr(sorted(vars(args).items())).encode()).hexdigest()[:6]
    run_dir = ROOT / "outputs" / "eval" / f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"run dir: {run_dir}")
    if args.csv and not Path(args.csv).is_absolute():
        args.csv = str(run_dir / args.csv)
    if args.plot and not Path(args.plot).is_absolute():
        args.plot = str(run_dir / args.plot)

    data_dir = Path(args.data)
    images = sorted(data_dir.glob("kodim*.png")) or sorted(data_dir.glob("*.png"))
    if args.images:
        images = [data_dir / n for n in args.images]
    if not images:
        sys.exit(f"No images found in {data_dir}")

    if "gnn" in args.methods and not Path(args.gnn_checkpoint).exists():
        sys.exit(f"GNN checkpoint not found: {args.gnn_checkpoint}")

    print(f"Evaluating {args.methods} on {len(images)} images from {data_dir}")
    print(f"Error bounds: {args.eb}  |  levels={args.levels} "
          f"anchor_stride={args.anchor_stride} anchor_block={args.anchor_block}")
    if "gnn" in args.methods:
        print(f"GNN checkpoint: {args.gnn_checkpoint}  (tile={args.gnn_tile}, "
              f"device={args.device})\n")

    rows: list[dict] = []
    total = len(images) * len(args.eb) * len(args.methods)
    done = 0
    for img_path in images:
        img = load_image(img_path)
        for eb in sorted(args.eb):
            for method in args.methods:
                done += 1
                print(f"[{done:3d}/{total}] {img_path.name} eb={eb:g} {method} ...",
                      end=" ", flush=True)
                try:
                    r = eval_sz3(img, eb) if method == "sz3" else \
                        eval_deepsz(img, eb, method, args)
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    continue
                if r is None:
                    print("skipped (SZ3 unavailable)")
                    continue
                r.update(image=img_path.name, method=method, eb=eb)
                rows.append(r)
                ok = "PASS" if r["bound_ok"] else "FAIL"
                print(f"bpp={r['bpp']:.3f} PSNR={r['psnr']:.2f}dB "
                      f"maxErr={r['max_err']:.1f} [{ok}] {r['t_comp']:.2f}s")
                # GNN runs untiled (whole image = one region), so its embedding
                # field is O(image size) instead of O(tile size); release it
                # and any cached CUDA blocks before the next image/eb.
                if method == "gnn" and args.device.startswith("cuda"):
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()

    if not rows:
        sys.exit("No results — check errors above.")

    present = [m for m in args.methods if any(r["method"] == m for r in rows)]
    print_tables(rows, present)
    if args.csv:
        save_csv(rows, args.csv)
    if not args.no_plot:
        plot_rd(rows, present, args.plot)


if __name__ == "__main__":
    main()
