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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = list_images(args.data)
    if not paths:
        raise SystemExit(f"no images found under {args.data}")
    print(f"{len(paths)} images")

    model = build_model(args.d)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} params, d={args.d}")

    run_loss = run_base = 0.0
    for step in range(1, args.steps + 1):
        x = sample_batch(paths, args.batch, args.crop)  # (B, N) truth
        levels = random.choice((3, 4, 5))
        stride = random.choice((8, 16, 32))
        masks = stage_masks(args.crop, args.crop, levels, stride, anchor_block=4)

        E = torch.zeros(args.batch, x.shape[1], args.d)
        known_vals = torch.full_like(x, 0.5)

        def reveal(kv, posf):
            noise = (torch.rand_like(x) * 2 - 1) * args.noise
            return torch.where(posf, (x + noise).clamp(0, 1), kv)

        # stage 0: anchors revealed exactly, no prediction
        prev = np.zeros((args.crop, args.crop), bool)
        known = masks[0].copy()
        known_vals = reveal(known_vals, torch.from_numpy(masks[0].reshape(-1)))

        # pixel-weighted MAE: sum absolute error over every hole across all
        # stages, divide by total holes, so each pixel counts once (the dense
        # final stages dominate, matching the L1 the quantizer really pays).
        abs_err = torch.zeros(())
        base_abs = 0.0
        npix = 0
        for pos in masks[1:]:
            if not pos.any():
                continue
            posf = torch.from_numpy(pos.reshape(-1))
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
        run_loss += loss.item()
        run_base += base_abs / max(npix, 1)

        if step % 100 == 0:
            print(f"step {step:5d}  MAE {run_loss / 100:.5f}  "
                  f"(known-mean baseline {run_base / 100:.5f})  ")
            run_loss = run_base = 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d": args.d}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
