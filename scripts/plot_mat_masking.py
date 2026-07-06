"""Plot MAT prediction PSNR vs mask fraction on the Kodak dataset.

For each image and masking fraction, the image is partially masked and MAT
inpaints the unknown pixels. PSNR is measured only on the masked (unknown)
pixels. The plot shows median + 0.2/0.8 quantile band across the 24 images.

Mask types (--mask-type):
  random   uniform random pixel mask                         (default)
  grid     regular N×N grid of unknown pixels
  block    random axis-aligned rectangular blocks

Usage:
    python scripts/plot_mat_masking.py
    python scripts/plot_mat_masking.py --mask-type block --block-size 32
    python scripts/plot_mat_masking.py --fractions 0.1 0.3 0.5 0.7 0.9
    python scripts/plot_mat_masking.py --mock        # fast NN predictor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KODAK_DIR = ROOT / "data" / "kodak"
DEFAULT_CKPT = ROOT / "models" / "MAT_Places512_G_fp16.safetensors"
TILE = 512


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------

def mask_random(h: int, w: int, frac_unknown: float, rng: np.random.Generator,
                **kw) -> np.ndarray:
    """Uniform random pixel mask. Returns bool (h, w): True = unknown."""
    flat = rng.random(h * w) < frac_unknown
    return flat.reshape(h, w)


def mask_grid(h: int, w: int, frac_unknown: float, rng: np.random.Generator,
              **kw) -> np.ndarray:
    """SZ-style anchor grid: known (anchor) pixels sit on an evenly-spaced
    lattice; everything else is unknown. Returns bool (h, w): True = unknown.

    To hit an arbitrary fraction we use a real-valued stride s = 1/sqrt(1-frac)
    and round anchor positions to integer pixels, so the anchors stay evenly
    spaced for any requested density (unlike a single integer stride, which can
    only realize the discrete densities 1 - 1/S²)."""
    frac_known = min(1.0, max(1e-9, 1.0 - frac_unknown))
    s = 1.0 / frac_known ** 0.5                 # target spacing (>= 1)
    rows = np.unique(np.round(np.arange(0, h, s)).astype(int))
    cols = np.unique(np.round(np.arange(0, w, s)).astype(int))
    rows = rows[rows < h]
    cols = cols[cols < w]
    m = np.ones((h, w), bool)                   # all unknown
    m[np.ix_(rows, cols)] = False               # anchor pixels are known
    return m


def mask_block(h: int, w: int, frac_unknown: float, rng: np.random.Generator,
               block_size: int = 32, **kw) -> np.ndarray:
    """Random axis-aligned blocks. Blocks are added until the masked fraction
    is at least frac_unknown."""
    m = np.zeros((h, w), bool)
    target = int(frac_unknown * h * w)
    bs = block_size
    while m.sum() < target:
        y = rng.integers(0, h)
        x = rng.integers(0, w)
        m[y:y + bs, x:x + bs] = True
    return m


MASK_FNS = {"random": mask_random, "grid": mask_grid, "block": mask_block}


# ---------------------------------------------------------------------------
# Per-tile prediction PSNR
# ---------------------------------------------------------------------------

def tile_prediction_psnr(
    tile: np.ndarray,          # (C, T, T) float32, values in original units
    unknown_mask: np.ndarray,  # (T, T) bool — pixels MAT must predict
    predictor,
    vmin: float,
    vmax: float,
) -> float | None:
    """Run predictor on `tile` with `~unknown_mask` as context; return PSNR
    of predictions at the unknown positions (dB, peak=255 for uint8)."""
    if not unknown_mask.any() or unknown_mask.all():
        return None
    known = ~unknown_mask
    recon = np.where(known[None], tile, 0.0)
    pred = predictor.predict(recon, known)        # (C, T, T)
    diff = tile[:, unknown_mask].astype(np.float64) - pred[:, unknown_mask].astype(np.float64)
    mse = float(np.mean(diff ** 2))
    if mse == 0:
        return float("inf")
    peak = max(vmax - vmin, 1.0)
    return 10 * np.log10(peak ** 2 / mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    return np.asarray(im.convert("RGB"))


def run(args):
    from matsz.predictor import MATPredictor, MockPredictor

    fractions = sorted(args.fractions)

    if args.cache and Path(args.cache).exists():
        raw = json.loads(Path(args.cache).read_text())
        results = {float(k): v for k, v in raw.items()}
        print(f"Loaded cached results from {args.cache}")
        _plot(results, fractions, args)
        return
    mask_fn = MASK_FNS[args.mask_type]
    mask_kw = {"block_size": args.block_size}

    images = sorted(KODAK_DIR.glob("kodim*.png"))
    if not images:
        sys.exit(f"No images found in {KODAK_DIR}")

    if args.mock:
        predictor = MockPredictor(TILE)
    else:
        predictor = MATPredictor(str(args.checkpoint), args.seed, 0.0, 255.0)

    # results[frac] = list of mean-PSNR values, one per image
    results: dict[float, list[float]] = {f: [] for f in fractions}

    n_total = len(images) * len(fractions) * args.n_masks
    done = 0

    for img_path in images:
        img = load_image(img_path)
        h, w, c = img.shape
        fimg = img.astype(np.float32)
        vmin, vmax = float(fimg.min()), float(fimg.max())

        # Pad to multiple of TILE
        ph = -(-h // TILE) * TILE
        pw = -(-w // TILE) * TILE
        canvas = np.pad(fimg, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
        canvas = canvas.transpose(2, 0, 1)  # (C, pH, pW)
        ty, tx = ph // TILE, pw // TILE

        # Update predictor vmin/vmax for this image
        if not args.mock:
            predictor.vmin = vmin
            predictor.vmax = vmax

        for frac in fractions:
            psnrs_this_image = []
            for rep in range(args.n_masks):
                rng = np.random.default_rng(args.seed + hash((img_path.name, frac, rep)) % (2**31))
                # one mask per tile (same fraction, independent draws)
                for i in range(ty):
                    for j in range(tx):
                        tile = canvas[:, i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE]
                        unknown = mask_fn(TILE, TILE, frac, rng, **mask_kw)
                        # clip to image boundary to avoid measuring on padding
                        yi0, yi1 = i*TILE, min((i+1)*TILE, h)
                        xj0, xj1 = j*TILE, min((j+1)*TILE, w)
                        # only evaluate on real (non-padded) pixels
                        valid = np.zeros((TILE, TILE), bool)
                        valid[:yi1-yi0, :xj1-xj0] = True
                        unknown_valid = unknown & valid
                        p = tile_prediction_psnr(tile, unknown_valid, predictor, vmin, vmax)
                        if p is not None:
                            psnrs_this_image.append(p)
                done += 1
                pct = 100 * done / n_total
                print(f"[{pct:5.1f}%] {img_path.name}  frac={frac:.2f}  rep={rep}  "
                      f"PSNR={np.mean(psnrs_this_image):.2f} dB", flush=True)
            if psnrs_this_image:
                results[frac].append(float(np.mean(psnrs_this_image)))

    if args.cache:
        Path(args.cache).write_text(json.dumps({str(k): v for k, v in results.items()}))
        print(f"Results cached to {args.cache}")

    _plot(results, fractions, args)


def _plot(results: dict, fractions: list, args):
    import matplotlib.pyplot as plt

    fracs = np.array(fractions)
    medians = np.array([np.median(results[f]) for f in fractions])
    q20 = np.array([np.quantile(results[f], 0.2) for f in fractions])
    q80 = np.array([np.quantile(results[f], 0.8) for f in fractions])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(fracs * 100, q20, q80, alpha=0.25, label="Q0.2–Q0.8")
    ax.plot(fracs * 100, medians, marker="o", linewidth=2, label="Median")

    predictor_label = "MockNN" if args.mock else "MAT"
    ax.set_xlabel("Mask fraction (%)")
    ax.set_ylabel("PSNR on masked pixels (dB)")
    ax.set_title(f"MAT inpainting quality vs mask fraction\n"
                 f"Kodak (24 images), mask={args.mask_type}, predictor={predictor_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(args.output)
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    # also print a small table
    print(f"\n{'frac':>6}  {'median':>8}  {'Q0.2':>8}  {'Q0.8':>8}  {'n':>4}")
    print("-" * 42)
    for f in fractions:
        vs = results[f]
        print(f"{f:6.2f}  {np.median(vs):8.2f}  {np.quantile(vs,0.2):8.2f}  "
              f"{np.quantile(vs,0.8):8.2f}  {len(vs):4d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help="mask fractions to sweep")
    ap.add_argument("--mask-type", choices=list(MASK_FNS), default="random",
                    help="masking strategy")
    ap.add_argument("--block-size", type=int, default=32,
                    help="block size for --mask-type block")
    ap.add_argument("--n-masks", type=int, default=1,
                    help="random mask repetitions per (image, fraction)")
    ap.add_argument("--mock", action="store_true",
                    help="use the fast nearest-neighbor predictor instead of MAT")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="mat_masking_psnr.png")
    ap.add_argument("--cache", default=None,
                    help="JSON file to save/load results (skips recompute if present)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
