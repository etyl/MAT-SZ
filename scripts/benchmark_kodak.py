"""Benchmark DeepSZ on the Kodak dataset.

Usage:
    python scripts/benchmark_kodak.py                          # default ebs 1 2 4
    python scripts/benchmark_kodak.py --csv results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepsz.codec import compress, decompress
from deepsz.predictor import InterpPredictor
from deepsz.bitstream import Header
from deepsz.baselines import sz3_roundtrip

KODAK_DIR = ROOT / "data" / "kodak"


def load_image(path: Path) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    return np.asarray(im)


def make_predictor(args, img: np.ndarray):
    return InterpPredictor("cubic", args.levels,
                           args.anchor_stride, args.anchor_block)


def build_predictor_for_decompress(args, hdr: Header):
    return InterpPredictor("cubic", hdr.levels,
                           hdr.anchor_stride, hdr.anchor_block)


def eval_one(img: np.ndarray, eb: float, args) -> dict:
    pred = make_predictor(args, img)

    t0 = time.time()
    stream, stats = compress(
        img, eb, pred,
        levels=args.levels,
        anchor_stride=args.anchor_stride,
        anchor_block=args.anchor_block,
        radius=args.radius,
        zstd_level=args.zstd_level,
    )
    t_comp = time.time() - t0

    t0 = time.time()
    rec = decompress(stream, lambda hdr: build_predictor_for_decompress(args, hdr))
    t_dec = time.time() - t0

    rec2d = rec if rec.ndim == img.ndim else rec[..., None]
    diff = img.astype(np.float64) - rec2d.astype(np.float64)
    max_err = float(np.abs(diff).max())
    mse = float(np.mean(diff ** 2))
    peak = 255.0 if img.dtype == np.uint8 else max(float(img.max()) - float(img.min()), 1.0)
    psnr = 10 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")
    bpp = 8 * len(stream) / (img.shape[0] * img.shape[1])

    sz3 = sz3_roundtrip(img, eb)
    if sz3 is not None:
        sz3_bytes, sz3_rec = sz3
        sz3_diff = img.astype(np.float64) - sz3_rec.astype(np.float64)
        sz3_mse = float(np.mean(sz3_diff ** 2))
        sz3_psnr = 10 * np.log10(peak ** 2 / sz3_mse) if sz3_mse > 0 else float("inf")
        sz3_bpp = 8 * sz3_bytes / (img.shape[0] * img.shape[1])
    else:
        sz3_psnr = sz3_bpp = float("nan")

    return dict(
        eb=eb,
        n_bytes=len(stream),
        ratio=stats["ratio"],
        bpp=bpp,
        psnr=psnr,
        max_err=max_err,
        bound_ok=max_err <= eb,
        outliers=stats["outliers"],
        t_comp=t_comp,
        t_dec=t_dec,
        t_predict=stats["predict_s"],
        sz3_bpp=sz3_bpp,
        sz3_psnr=sz3_psnr,
    )


def fmt(v, fmt_str):
    return "—" if (isinstance(v, float) and np.isnan(v)) else format(v, fmt_str)


def print_table(rows: list[dict], images: list[str]) -> None:
    ebs = sorted({r["eb"] for r in rows})

    # Per-image summary: one row per (image, eb)
    print("\n=== Per-image results ===")
    hdr = (f"{'image':<12} {'eb':>4} "
           f"{'bpp':>7} {'ratio':>6} {'PSNR':>8} {'maxErr':>7} {'ok':>4} "
           f"{'sz3bpp':>7} {'sz3PSNR':>8} "
           f"{'tComp':>7} {'tDec':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bound_str = "PASS" if r["bound_ok"] else "FAIL"
        print(f"{r['image']:<12} {r['eb']:>4.0f} "
              f"{r['bpp']:>7.3f} {r['ratio']:>6.2f} {r['psnr']:>8.2f} "
              f"{r['max_err']:>7.2f} {bound_str:>4} "
              f"{fmt(r['sz3_bpp'], '7.3f')} {fmt(r['sz3_psnr'], '8.2f')} "
              f"{r['t_comp']:>7.1f} {r['t_dec']:>6.1f}")

    # Average over images per eb
    print("\n=== Averages over Kodak (24 images) ===")
    hdr2 = (f"{'eb':>4} "
            f"{'bpp':>7} {'ratio':>6} {'PSNR':>8} "
            f"{'sz3bpp':>7} {'sz3PSNR':>8} "
            f"{'PASS%':>6} {'tComp':>7}")
    print(hdr2)
    print("-" * len(hdr2))
    for eb in ebs:
        sub = [r for r in rows if r["eb"] == eb]
        avg_bpp = np.mean([r["bpp"] for r in sub])
        avg_ratio = np.mean([r["ratio"] for r in sub])
        avg_psnr = np.mean([r["psnr"] for r in sub if np.isfinite(r["psnr"])])
        sz3_bpps = [r["sz3_bpp"] for r in sub if not np.isnan(r["sz3_bpp"])]
        sz3_psnrs = [r["sz3_psnr"] for r in sub if not np.isnan(r["sz3_psnr"])]
        avg_sz3_bpp = np.mean(sz3_bpps) if sz3_bpps else float("nan")
        avg_sz3_psnr = np.mean(sz3_psnrs) if sz3_psnrs else float("nan")
        pass_pct = 100 * sum(r["bound_ok"] for r in sub) / len(sub)
        avg_tcomp = np.mean([r["t_comp"] for r in sub])
        print(f"{eb:>4.0f} "
              f"{avg_bpp:>7.3f} {avg_ratio:>6.2f} {avg_psnr:>8.2f} "
              f"{fmt(avg_sz3_bpp, '7.3f')} {fmt(avg_sz3_psnr, '8.2f')} "
              f"{pass_pct:>6.1f} {avg_tcomp:>7.1f}")


def save_csv(rows: list[dict], path: str) -> None:
    import csv
    fields = ["image", "eb", "n_bytes", "ratio", "bpp", "psnr", "max_err",
              "bound_ok", "outliers", "t_comp", "t_dec", "t_predict",
              "sz3_bpp", "sz3_psnr"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eb", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                    help="error bounds to sweep (default: 1 2 4)")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--anchor-stride", type=int, default=16)
    ap.add_argument("--anchor-block", type=int, default=4)
    ap.add_argument("--radius", type=int, default=1 << 15)
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--csv", default=None, help="save per-image CSV to this path")
    ap.add_argument("--images", nargs="*", default=None,
                    help="specific image filenames to run (default: all 24)")
    args = ap.parse_args()

    images = sorted(KODAK_DIR.glob("kodim*.png"))
    if args.images:
        images = [KODAK_DIR / n for n in args.images]

    if not images:
        sys.exit(f"No images found in {KODAK_DIR}")

    print(f"Benchmarking DeepSZ (interp) on {len(images)} Kodak images")
    print(f"Error bounds: {args.eb}  |  levels={args.levels}")
    print(f"Images: {KODAK_DIR}\n")

    rows: list[dict] = []
    total = len(images) * len(args.eb)
    done = 0

    for img_path in images:
        img = load_image(img_path)
        for eb in sorted(args.eb):
            done += 1
            print(f"[{done:3d}/{total}] {img_path.name}  eb={eb} ...", end=" ", flush=True)
            try:
                r = eval_one(img, eb, args)
            except Exception as exc:
                print(f"ERROR: {exc}")
                continue
            r["image"] = img_path.name
            rows.append(r)
            bound_str = "PASS" if r["bound_ok"] else "FAIL"
            print(f"bpp={r['bpp']:.3f}  PSNR={r['psnr']:.2f} dB  "
                  f"maxErr={r['max_err']:.1f} [{bound_str}]  "
                  f"{r['t_comp']:.1f}s comp")

    if not rows:
        sys.exit("No results — check errors above.")

    print_table(rows, [p.name for p in images])

    if args.csv:
        save_csv(rows, args.csv)


if __name__ == "__main__":
    main()
