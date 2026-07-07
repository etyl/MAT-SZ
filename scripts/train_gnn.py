"""Train the lightweight GNN predictor (matsz/gnn_predictor.py) on natural
images. CPU-friendly: the model is ~20k params, a few thousand steps suffice.

    conda run -n nf python scripts/train_gnn.py --data /path/to/images

Each step crops random 128x128 patches, picks a random colour plane
(R / G / B / grayscale luma) so the net trains on grayscale as well as single
colour channels, teacher-forces the hierarchical stage schedule (levels.py)
with true values (+ optional quantization-like noise) and minimises pixel-
weighted MAE over the hole positions of every refinement stage (each pixel
counts once, so the dense stages dominate). Checkpoint -> data/gnn_predictor.pt.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matsz.gnn_predictor import build_model, stage_forward
from matsz.levels import stage_masks

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}


def list_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT]


def load_plane(path, crop):
    """Random `crop`x`crop` patch from a random plane (R/G/B/gray), in [0,1]."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if min(im.size) < crop:
        im = im.resize((max(crop, im.size[0]), max(crop, im.size[1])))
    a = np.asarray(im, np.float32) / 255.0  # (H, W, 3)
    h, w, _ = a.shape
    y = random.randint(0, h - crop)
    x = random.randint(0, w - crop)
    patch = a[y:y + crop, x:x + crop]
    ch = random.randint(0, 3)
    if ch < 3:
        return patch[..., ch]
    return patch @ np.array([0.299, 0.587, 0.114], np.float32)  # luma


def sample_batch(paths, batch, crop):
    planes = [load_plane(random.choice(paths), crop) for _ in range(batch)]
    return torch.from_numpy(np.stack(planes)).reshape(batch, -1)  # (B, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of natural images")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "data" / "gnn_predictor.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.01,
                    help="uniform +/- noise on known values (mimics eb)")
    ap.add_argument("--max-radius", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu", help="cpu | cuda | cuda:N")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    paths = list_images(args.data)
    if not paths:
        raise SystemExit(f"no images found under {args.data}")
    print(f"{len(paths)} images, device={device}")

    model = build_model(args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} params, d={args.d}")

    run_loss = run_base = 0.0
    history = []  # (step, mae, baseline) per step, for the loss curve
    bar = tqdm(range(1, args.steps + 1), desc="train")
    for step in bar:
        x = sample_batch(paths, args.batch, args.crop).to(device)  # (B, N) truth
        levels = random.choice((3, 4, 5))
        stride = random.choice((8, 16, 32))
        masks = stage_masks(args.crop, args.crop, levels, stride, anchor_block=4)

        E = torch.zeros(args.batch, x.shape[1], args.d, device=device)
        known_vals = torch.full_like(x, 0.5)

        def reveal(kv, posf):
            noise = (torch.rand_like(x) * 2 - 1) * args.noise
            return torch.where(posf, (x + noise).clamp(0, 1), kv)

        def maskf(m):  # flat bool mask on the training device
            return torch.from_numpy(m.reshape(-1)).to(device)

        # stage 0: anchors revealed exactly, no prediction
        prev = np.zeros((args.crop, args.crop), bool)
        known = masks[0].copy()
        known_vals = reveal(known_vals, maskf(masks[0]))

        # pixel-weighted MAE: sum absolute error over every hole across all
        # stages, divide by total holes, so each pixel counts once (the dense
        # final stages dominate, matching the L1 the quantizer really pays).
        abs_err = torch.zeros(())
        base_abs = 0.0
        npix = 0
        for pos in masks[1:]:
            if not pos.any():
                continue
            posf = maskf(pos)
            # predict with `known` = earlier stages (pos not yet revealed),
            # finalising the previous stage's embeddings inside stage_forward
            pred, E = stage_forward(model, E, prev, known, known_vals,
                                    args.max_radius, torch)
            tgt = x[:, posf]
            abs_err = abs_err + (pred[:, posf] - tgt).abs().sum()
            base_abs += (known_vals.mean(1, keepdim=True) - tgt).abs().sum().item()
            npix += tgt.numel()
            prev = known.copy()
            known = known | pos                     # teacher-force next stage
            known_vals = reveal(known_vals, posf)

        loss = abs_err / max(npix, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append((step, loss.item(), base_abs / max(npix, 1)))
        run_loss += loss.item()
        run_base += base_abs / max(npix, 1)

        if step % 100 == 0:
            bar.set_postfix(mae=f"{run_loss / 100:.5f}",
                            baseline=f"{run_base / 100:.5f}")
            run_loss = run_base = 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d": args.d}, out)
    print(f"saved {out}")
    save_curve(history, out.with_suffix(".loss"))


def save_curve(history, stem):
    """Write the loss curve as CSV always, and a PNG if matplotlib is around."""
    csv = stem.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("step,mae,baseline\n")
        for s, mae, b in history:
            f.write(f"{s},{mae:.6f},{b:.6f}\n")
    print(f"saved {csv}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    st = [h[0] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(st, [h[1] for h in history], label="model MAE")
    plt.plot(st, [h[2] for h in history], label="known-mean baseline", alpha=.6)
    plt.xlabel("step")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    png = stem.with_suffix(".png")
    plt.savefig(png, dpi=110)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
