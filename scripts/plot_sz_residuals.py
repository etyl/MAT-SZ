"""Visualize the residuals left by SZ3's prediction phase on Kodak images.

SZ3's default predictor is multilevel spline *interpolation*, built as a
top-down dyadic pyramid: a single coarse anchor is kept, then the algorithm
predicts the MIDPOINT and DOUBLES the number of predicted points at every finer
level (stride N/2 -> N/4 -> ... -> 1). Each new point is predicted by 1-D cubic
(or linear) interpolation from its two/four already-defined neighbours along one
dimension. The residual = original - prediction is what SZ then quantizes and
entropy-codes; its magnitude/structure determines the bitrate.

This script runs that interpolation (open loop, i.e. predicting from the
original neighbours) per channel and plots, for one image:
    original | SZ prediction | signed residual heatmap | residual histogram

Usage:
    python scripts/plot_sz_residuals.py                     # kodim23, cubic
    python scripts/plot_sz_residuals.py --image 1 --interp linear
    python scripts/plot_sz_residuals.py --stride 32         # cap coarsest anchor grid
    python scripts/plot_sz_residuals.py --all               # MAE over all 24 images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KODAK_DIR = ROOT / "data" / "kodak"

LUMA = np.array([0.299, 0.587, 0.114])


# ---------------------------------------------------------------------------
# SZ3-style multilevel spline interpolation
# ---------------------------------------------------------------------------

def _pred_block(O: np.ndarray, axis: int, main: np.ndarray, other: np.ndarray,
                s: int, cubic: bool) -> np.ndarray:
    """Predict the grid block (varying `main` along `axis`, fixed `other` on the
    orthogonal axis) by 1-D interpolation from the ORIGINAL field `O` at offsets
    +-s (linear) and +-3s (cubic) along `axis`. Points missing the full cubic
    stencil fall back to linear; points missing the right neighbour copy it."""
    n = O.shape[axis]

    def gather(off: int) -> np.ndarray:
        idx = np.clip(main + off, 0, n - 1)
        return O[np.ix_(idx, other)] if axis == 0 else O[np.ix_(other, idx)]

    def col(mask: np.ndarray) -> np.ndarray:            # broadcast along `main`
        return mask[:, None] if axis == 0 else mask[None, :]

    left = gather(-s)
    lin = 0.5 * (left + gather(+s))
    out = np.where(col(main + s < n), lin, left)        # copy left at far border

    if cubic:
        ok = (main - 3 * s >= 0) & (main + 3 * s < n)
        cub = (-gather(-3 * s) + 9 * gather(-s)
               + 9 * gather(+s) - gather(+3 * s)) / 16.0
        out = np.where(col(ok), cub, out)
    return out


def sz_interp(field: np.ndarray, cubic: bool = True,
              coarsest: int | None = None) -> np.ndarray:
    """SZ3 top-down dyadic spline interpolation of a 2-D field.

    Starting from stride L (the largest power of two below the image, or
    `coarsest`/2 when a coarse anchor grid is requested), each level predicts the
    midpoints then halves the stride, doubling the predicted-point count. Only
    the anchor (single point, or the `coarsest` grid) is kept exactly; every
    other pixel is predicted. Prediction reads ORIGINAL neighbours (open loop)."""
    O = field.astype(np.float64)
    P = O.copy()
    h, w = O.shape

    if coarsest is not None:
        s = coarsest // 2                               # keep the coarse grid exact
    else:
        s = 1
        while s * 2 < max(h, w):
            s *= 2

    while s >= 1:
        two_s = 2 * s
        # dim-0: odd rows, on known columns (multiples of 2s)
        mi = np.arange(s, h, two_s)
        oj = np.arange(0, w, two_s)
        if mi.size and oj.size:
            P[np.ix_(mi, oj)] = _pred_block(O, 0, mi, oj, s, cubic)
        # dim-1: odd columns, on all rows at spacing s (fills the (odd,odd) points)
        mi2 = np.arange(0, h, s)
        oj2 = np.arange(s, w, two_s)
        if mi2.size and oj2.size:
            P[np.ix_(mi2, oj2)] = _pred_block(O, 1, oj2, mi2, s, cubic)
        s //= 2
    return P


def sz_predict_rgb(img: np.ndarray, cubic: bool, coarsest: int | None) -> np.ndarray:
    """Per-channel SZ interpolation. img (H,W,C) -> pred float64."""
    pred = np.empty(img.shape, np.float64)
    for k in range(img.shape[2]):
        pred[..., k] = sz_interp(img[..., k].astype(np.float64), cubic, coarsest)
    return pred


# ---------------------------------------------------------------------------

def load_image(path: Path, gray: bool = False) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    if gray:
        return np.asarray(im.convert("L"))[..., None]   # (H, W, 1)
    return np.asarray(im.convert("RGB"))


def residual_stats(res: np.ndarray) -> dict:
    a = np.abs(res)
    return {
        "mae": float(a.mean()),
        "std": float(res.std()),
        "max": float(a.max()),
        "p_le1": float((a <= 1).mean()) * 100,
        "p_le2": float((a <= 2).mean()) * 100,
        "p_le4": float((a <= 4).mean()) * 100,
    }


def run_all(args):
    images = sorted(KODAK_DIR.glob("kodim*.png"))
    print(f"{'image':<12}{'MAE':>8}{'std':>8}{'max':>8}"
          f"{'%<=1':>8}{'%<=2':>8}{'%<=4':>8}")
    print("-" * 60)
    maes = []
    for p in images:
        img = load_image(p, args.gray).astype(np.float64)
        pred = sz_predict_rgb(img, args.interp == "cubic", args.stride)
        res = img - pred
        s = residual_stats(res)
        maes.append(s["mae"])
        print(f"{p.name:<12}{s['mae']:8.3f}{s['std']:8.3f}{s['max']:8.0f}"
              f"{s['p_le1']:8.1f}{s['p_le2']:8.1f}{s['p_le4']:8.1f}")
    print("-" * 60)
    anchor = f"anchor grid {args.stride}" if args.stride else "full pyramid"
    print(f"{'MEAN':<12}{np.mean(maes):8.3f}   (interp={args.interp}, {anchor})")


def run_one(args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    path = KODAK_DIR / f"kodim{args.image:02d}.png"
    if not path.exists():
        sys.exit(f"not found: {path}")
    img = load_image(path, args.gray).astype(np.float64)
    pred = sz_predict_rgb(img, args.interp == "cubic", args.stride)
    res = img - pred

    if args.gray:
        res_luma = res[..., 0]                       # single channel
        show_img = img[..., 0]
        show_pred = np.clip(pred[..., 0], 0, 255)
        show_kw = dict(cmap="gray", vmin=0, vmax=255)
    else:
        res_luma = res @ LUMA                        # signed luminance residual
        show_img = img.astype(np.uint8)
        show_pred = np.clip(pred, 0, 255).astype(np.uint8)
        show_kw = {}
    stats = residual_stats(res)
    vlim = max(1.0, np.percentile(np.abs(res_luma), 99.5))
    mode = "gray" if args.gray else "RGB"

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].imshow(show_img, **show_kw)
    ax[0, 0].set_title(f"Original  (kodim{args.image:02d}, {mode})")

    anchor = f"anchor grid {args.stride}" if args.stride else "full pyramid"
    ax[0, 1].imshow(show_pred, **show_kw)
    ax[0, 1].set_title(f"SZ prediction  ({args.interp} interp, {anchor})")

    im = ax[1, 0].imshow(res_luma, cmap="RdBu_r",
                         norm=TwoSlopeNorm(0.0, -vlim, vlim))
    ax[1, 0].set_title(f"Residual ({'gray' if args.gray else 'luma'}, signed)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)

    for a in ax.flat[:3]:
        a.set_xticks([]); a.set_yticks([])

    hb = ax[1, 1]
    hb.hist(res.ravel(), bins=201, range=(-40, 40), color="steelblue")
    hb.set_yscale("log")
    hb.set_xlabel("residual (0-255 scale)")
    hb.set_ylabel("count (log)")
    hb.set_title("Residual histogram" + ("" if args.gray else " (all channels)"))
    hb.text(0.02, 0.95,
            f"MAE={stats['mae']:.2f}\nstd={stats['std']:.2f}\n"
            f"max={stats['max']:.0f}\n"
            f"|r|<=1: {stats['p_le1']:.1f}%\n"
            f"|r|<=2: {stats['p_le2']:.1f}%\n"
            f"|r|<=4: {stats['p_le4']:.1f}%",
            transform=hb.transAxes, va="top", family="monospace", fontsize=9)

    fig.suptitle("SZ3 interpolation-predictor residuals", fontsize=14)
    plt.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"Plot saved to {args.output}")
    print("residual stats:", {k: round(v, 3) for k, v in stats.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=int, default=23, help="kodak image index 1-24")
    ap.add_argument("--stride", type=int, default=0,
                    help="cap the coarsest anchor grid at this stride (power of two); "
                         "0 = full top-down pyramid (single anchor, SZ3 default)")
    ap.add_argument("--interp", choices=["cubic", "linear"], default="cubic")
    ap.add_argument("--gray", action="store_true",
                    help="convert to grayscale (single luma channel) before predicting")
    ap.add_argument("--all", action="store_true",
                    help="print residual stats over all 24 images instead of plotting")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = str(ROOT / "data" /
                          ("sz_residuals_gray.png" if args.gray else "sz_residuals.png"))
    if args.stride & (args.stride - 1):
        sys.exit("--stride must be a power of two")
    args.stride = args.stride or None       # 0 -> full pyramid
    run_all(args) if args.all else run_one(args)


if __name__ == "__main__":
    main()
