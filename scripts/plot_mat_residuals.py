"""Visualize the residuals left by MAT used as the SZ predictor.

The SZ analog (scripts/plot_sz_residuals.py) predicts every non-anchor pixel by
spline interpolation from its neighbours. Here we instead give MAT an anchor
grid as *known context* and let it inpaint every unknown pixel in one forward
pass, then plot the residual = original - MAT prediction (zero at the anchors,
which we keep exact). This is the direct predictor-swap: same anchor grid,
MAT inpainting in place of spline interpolation.

MAT works on 512x512 tiles, so the image is padded to a multiple of 512, tiled,
predicted tile-by-tile, and cropped back.

Layout (like the SZ script): original | MAT prediction | residual | histogram.

Usage:
    python scripts/plot_mat_residuals.py                       # kodim23, grid 75% unknown
    python scripts/plot_mat_residuals.py --gray
    python scripts/plot_mat_residuals.py --frac 0.5 --mask-type random
    python scripts/plot_mat_residuals.py --all                 # MAE over all 24 images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_mat_masking import MASK_FNS   # noqa: E402  (mask strategies)

KODAK_DIR = ROOT / "data" / "kodak"
DEFAULT_CKPT = ROOT / "models" / "MAT_Places512_G_fp16.safetensors"
TILE = 512
LUMA = np.array([0.299, 0.587, 0.114])


def load_image(path: Path, gray: bool) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    if gray:
        return np.asarray(im.convert("L"))[..., None]
    return np.asarray(im.convert("RGB"))


def mat_predict(img: np.ndarray, known: np.ndarray, predictor) -> np.ndarray:
    """Inpaint unknown pixels of `img` (H,W,C) given the `known` mask (H,W).
    Tiles to 512x512, predicts each tile, returns prediction (H,W,C) float64.
    Known pixels are kept exact; only holes take MAT's output."""
    h, w, c = img.shape
    ph = -(-h // TILE) * TILE
    pw = -(-w // TILE) * TILE
    canvas = np.pad(img, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    kcanvas = np.pad(known, ((0, ph - h), (0, pw - w)), constant_values=False)
    chw = canvas.transpose(2, 0, 1).astype(np.float32)          # (C, pH, pW)
    out = chw.copy()

    for i in range(ph // TILE):
        for j in range(pw // TILE):
            ys, xs = slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE)
            tile = chw[:, ys, xs]
            ktile = kcanvas[ys, xs]
            recon = np.where(ktile[None], tile, 0.0).astype(np.float32)
            pred = predictor.predict(recon, ktile)              # (C, T, T)
            out[:, ys, xs] = np.where(ktile[None], tile, pred)

    pred_hwc = out.transpose(1, 2, 0)[:h, :w]
    return pred_hwc.astype(np.float64)


def mat_predict_progressive(img: np.ndarray, coarsest: int, predictor):
    """Hierarchical MAT prediction over SZ's dyadic schedule.

    Known context starts as the coarse anchor grid (spacing `coarsest`). For
    stride s = coarsest/2, ..., 1 the points on the spacing-s grid that are not
    yet known are predicted by ONE MAT pass per tile (given the current known
    grid), then revealed with their ORIGINAL values (open loop) for the next,
    finer level. Returns (pred HWC float64, predicted-mask HWC bool, level_map).
    """
    h, w, c = img.shape
    ph = -(-h // TILE) * TILE
    pw = -(-w // TILE) * TILE
    canvas = np.pad(img, ((0, ph - h), (0, pw - w), (0, 0)),
                    mode="edge").transpose(2, 0, 1).astype(np.float32)   # (C,pH,pW)
    pH, pW = canvas.shape[1:]

    known = np.zeros((pH, pW), bool)
    known[::coarsest, ::coarsest] = True
    pred = canvas.copy()
    level_map = np.full((pH, pW), -1, np.int16)                          # -1 = anchor

    s, lvl = coarsest // 2, 0
    while s >= 1:
        grid = np.zeros((pH, pW), bool)
        grid[::s, ::s] = True
        newpos = grid & ~known
        for i in range(pH // TILE):
            for j in range(pW // TILE):
                ys, xs = slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE)
                npos = newpos[ys, xs]
                if not npos.any():
                    continue
                ktile = known[ys, xs]
                tile = canvas[:, ys, xs]
                recon = np.where(ktile[None], tile, 0.0).astype(np.float32)
                mp = predictor.predict(recon, ktile)
                pred[:, ys, xs] = np.where(npos[None], mp, pred[:, ys, xs])
        level_map[newpos] = lvl
        known |= newpos
        s //= 2
        lvl += 1

    pred_hwc = pred.transpose(1, 2, 0)[:h, :w].astype(np.float64)
    lmap = level_map[:h, :w]
    return pred_hwc, lmap >= 0, lmap


def make_mask(h: int, w: int, args, rng) -> np.ndarray:
    """Unknown-pixel mask (True = MAT must predict). Anchors (known) follow the
    same grid/random convention as the masking study."""
    return MASK_FNS[args.mask_type](h, w, args.frac, rng, block_size=args.block_size)


def residual_stats(res: np.ndarray, unknown: np.ndarray) -> dict:
    a = np.abs(res[unknown]) if unknown.any() else np.abs(res)
    return {
        "mae": float(a.mean()),
        "std": float(a.std()),
        "max": float(a.max()),
        "p_le1": float((a <= 1).mean()) * 100,
        "p_le2": float((a <= 2).mean()) * 100,
        "p_le4": float((a <= 4).mean()) * 100,
    }


def make_predictor(args):
    from matsz.predictor import MATPredictor, MockPredictor
    if args.mock:
        return MockPredictor(TILE)
    return MATPredictor(str(args.checkpoint), args.seed, 0.0, 255.0)


def run_all(args):
    predictor = make_predictor(args)
    images = sorted(KODAK_DIR.glob("kodim*.png"))
    print(f"{'image':<12}{'MAE':>8}{'std':>8}{'max':>8}"
          f"{'%<=1':>8}{'%<=2':>8}{'%<=4':>8}")
    print("-" * 60)
    maes = []
    for p in images:
        img = load_image(p, args.gray).astype(np.float64)
        h, w, _ = img.shape
        if not args.mock:
            predictor.vmin, predictor.vmax = float(img.min()), float(img.max())
        if args.progressive:
            pred, predicted, _ = mat_predict_progressive(img, args.coarsest, predictor)
            unk3 = predicted[..., None] & np.ones_like(img, bool)
        else:
            rng = np.random.default_rng(args.seed)
            unknown = make_mask(h, w, args, rng)
            pred = mat_predict(img, ~unknown, predictor)
            unk3 = unknown[..., None] & np.ones_like(img, bool)
        s = residual_stats(img - pred, unk3)
        maes.append(s["mae"])
        print(f"{p.name:<12}{s['mae']:8.3f}{s['std']:8.3f}{s['max']:8.0f}"
              f"{s['p_le1']:8.1f}{s['p_le2']:8.1f}{s['p_le4']:8.1f}")
    print("-" * 60)
    cfg = (f"progressive, coarsest={args.coarsest}" if args.progressive
           else f"mask={args.mask_type}, frac={args.frac}")
    print(f"{'MEAN':<12}{np.mean(maes):8.3f}   ({cfg})")


def run_one(args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    path = KODAK_DIR / f"kodim{args.image:02d}.png"
    if not path.exists():
        sys.exit(f"not found: {path}")
    predictor = make_predictor(args)
    img = load_image(path, args.gray).astype(np.float64)
    h, w, _ = img.shape
    if not args.mock:
        predictor.vmin, predictor.vmax = float(img.min()), float(img.max())

    lmap = None
    if args.progressive:
        pred, predicted, lmap = mat_predict_progressive(img, args.coarsest, predictor)
        unknown = predicted
    else:
        rng = np.random.default_rng(args.seed)
        unknown = make_mask(h, w, args, rng)
        pred = mat_predict(img, ~unknown, predictor)
    res = img - pred

    unk3 = unknown[..., None] & np.ones_like(img, bool)
    stats = residual_stats(res, unk3)

    if lmap is not None:                    # per-level MAE (luma / channel-mean)
        resc = np.abs(res).mean(axis=2)
        print("per-level residual (stride : count : MAE):")
        for lvl in range(lmap.max() + 1):
            m = lmap == lvl
            if m.any():
                print(f"  s={args.coarsest >> (lvl + 1):<3d} "
                      f"n={int(m.sum()):7d}  MAE={resc[m].mean():.3f}")

    if args.gray:
        res_luma = res[..., 0]
        show_img = img[..., 0]
        show_pred = np.clip(pred[..., 0], 0, 255)
        show_kw = dict(cmap="gray", vmin=0, vmax=255)
    else:
        res_luma = res @ LUMA
        show_img = img.astype(np.uint8)
        show_pred = np.clip(pred, 0, 255).astype(np.uint8)
        show_kw = {}
    vlim = max(1.0, np.percentile(np.abs(res_luma), 99.5))
    mode = "gray" if args.gray else "RGB"

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].imshow(show_img, **show_kw)
    ax[0, 0].set_title(f"Original  (kodim{args.image:02d}, {mode})")

    if args.progressive:
        sub = f"progressive, coarsest {args.coarsest}, {lmap.max() + 1} levels"
    else:
        sub = f"{args.mask_type} anchors, {100 * (~unknown).mean():.0f}% known"
    ax[0, 1].imshow(show_pred, **show_kw)
    ax[0, 1].set_title(f"MAT prediction  ({sub})")

    im = ax[1, 0].imshow(res_luma, cmap="RdBu_r",
                         norm=TwoSlopeNorm(0.0, -vlim, vlim))
    ax[1, 0].set_title(f"Residual ({'gray' if args.gray else 'luma'}, signed)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)

    for a in ax.flat[:3]:
        a.set_xticks([]); a.set_yticks([])

    hb = ax[1, 1]
    hb.hist(res[unk3].ravel(), bins=201, range=(-40, 40), color="indianred")
    hb.set_yscale("log")
    hb.set_xlabel("residual on predicted pixels (0-255 scale)")
    hb.set_ylabel("count (log)")
    hb.set_title("Residual histogram" + ("" if args.gray else " (all channels)"))
    hb.text(0.02, 0.95,
            f"MAE={stats['mae']:.2f}\nstd={stats['std']:.2f}\n"
            f"max={stats['max']:.0f}\n"
            f"|r|<=1: {stats['p_le1']:.1f}%\n"
            f"|r|<=2: {stats['p_le2']:.1f}%\n"
            f"|r|<=4: {stats['p_le4']:.1f}%",
            transform=hb.transAxes, va="top", family="monospace", fontsize=9)

    fig.suptitle("MAT-as-predictor residuals", fontsize=14)
    plt.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"Plot saved to {args.output}")
    print("residual stats:", {k: round(v, 3) for k, v in stats.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=int, default=23, help="kodak image index 1-24")
    ap.add_argument("--frac", type=float, default=0.75,
                    help="fraction of pixels MAT must predict (unknown); "
                         "the rest are known anchors")
    ap.add_argument("--mask-type", choices=list(MASK_FNS), default="grid")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--progressive", action="store_true",
                    help="hierarchical MAT over SZ's dyadic schedule (one MAT pass "
                         "per level, revealing coarser levels as context)")
    ap.add_argument("--coarsest", type=int, default=16,
                    help="coarsest anchor stride for --progressive (power of two)")
    ap.add_argument("--gray", action="store_true",
                    help="convert to grayscale before predicting")
    ap.add_argument("--mock", action="store_true",
                    help="use the fast nearest-neighbour predictor instead of MAT")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true",
                    help="print residual stats over all 24 images instead of plotting")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.coarsest & (args.coarsest - 1):
        sys.exit("--coarsest must be a power of two")
    if args.output is None:
        tag = "_prog" if args.progressive else ""
        tag += "_gray" if args.gray else ""
        args.output = str(ROOT / "data" / f"mat_residuals{tag}.png")
    run_all(args) if args.all else run_one(args)


if __name__ == "__main__":
    main()
