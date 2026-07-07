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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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


@lru_cache(maxsize=1024)
def _decode_rgb(path):
    """Decode the whole image once and keep it in RAM (uint8 RGB). Random crops
    then slice from memory instead of re-decoding megapixels every sample — the
    decode was ~half the training-step CPU time.
    ponytail: 1024-image LRU (covers DIV2K's 800; ~8 GB at 2K uint8). Lower
    maxsize if the node is RAM-tight, raise/drop it to cache a bigger set."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), np.uint8)  # (H, W, 3)


def load_plane(path, crop):
    """Random `crop`x`crop` patch from a random plane (R/G/B/gray), in [0,1]."""
    a = _decode_rgb(path)  # cached uint8; never mutated (we only slice + copy)
    if min(a.shape[:2]) < crop:
        from PIL import Image
        im = Image.fromarray(a).resize((max(crop, a.shape[1]),
                                        max(crop, a.shape[0])))
        a = np.asarray(im, np.uint8)
    h, w, _ = a.shape
    y = random.randint(0, h - crop)
    x = random.randint(0, w - crop)
    patch = a[y:y + crop, x:x + crop].astype(np.float32) / 255.0
    ch = random.randint(0, 3)
    if ch < 3:
        return patch[..., ch]
    return patch @ np.array([0.299, 0.587, 0.114], np.float32)  # luma


def sample_batch(paths, batch, crop):
    planes = [load_plane(random.choice(paths), crop) for _ in range(batch)]
    return torch.from_numpy(np.stack(planes)).reshape(batch, -1)  # (B, N)


def prefetch_batches(paths, batch, crop, steps, workers):
    """Yield `steps` CPU batches, decoding on `workers` background threads so
    image I/O overlaps the GPU step (PIL/numpy release the GIL, so threads
    decode in parallel). Keeps ~2*workers batches in flight.

    ponytail: workers==0 is the plain synchronous path — use it for a bit-
    reproducible seeded run, since parallel workers race on the global RNG and
    make the *content* of each batch non-deterministic (consumption order is
    still FIFO). Overlap matters most while the decode cache is cold or the set
    is larger than the cache; a fully-cached set barely needs it."""
    if workers <= 0:
        for _ in range(steps):
            yield sample_batch(paths, batch, crop)
        return
    ahead = max(2 * workers, 2)
    ex = ThreadPoolExecutor(max_workers=workers)
    futs = deque()
    submitted = 0
    try:
        while submitted < min(ahead, steps):
            futs.append(ex.submit(sample_batch, paths, batch, crop))
            submitted += 1
        for _ in range(steps):
            b = futs.popleft().result()
            if submitted < steps:
                futs.append(ex.submit(sample_batch, paths, batch, crop))
                submitted += 1
            yield b
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def qz(pred, truth, eb):
    """Linear-quantised reconstruction (matsz.quantizer): pred + the residual
    snapped to the 2*eb grid, so |recon - truth| <= eb. Works on np or torch."""
    round_ = torch.round if torch.is_tensor(truth) else np.round
    return pred + 2 * eb * round_((truth - pred) / (2 * eb))


def interp_eval(x, masks, eb, order):
    """Real interp-predictor closed loop with linear quantisation, batch 1 on
    CPU. At each stage the interp predictor fills the holes from the *quantised*
    reconstruction of earlier stages (never the truth), exactly as the codec's
    InterpPredictor does; returns (sum |interp - truth|, n_holes) — the baseline
    prediction MAE the GNN must beat."""
    from scipy.interpolate import griddata

    h, w = masks[0].shape
    truth = x.detach().cpu().numpy().reshape(h, w)
    recon = np.zeros((h, w), np.float32)
    known = masks[0].copy()
    recon[known] = qz(0.0, truth[known], eb)  # anchors: quantised against pred 0
    abs_err = 0.0
    npix = 0
    for pos in masks[1:]:
        if not pos.any():
            continue
        pts = np.column_stack(np.nonzero(known)).astype(np.float64)
        q = np.column_stack(np.nonzero(pos)).astype(np.float64)
        z = griddata(pts, recon[known], q, method=order)
        nan = np.isnan(z)  # holes outside the known convex hull
        if nan.any():
            z[nan] = griddata(pts, recon[known], q, method="nearest")[nan]
        tg = truth[pos]
        abs_err += float(np.abs(z - tg).sum())
        npix += tg.size
        recon[pos] = qz(z, tg, eb)  # feed back the quantised interp recon
        known = known | pos
    return abs_err, npix


def residual_rgb(res):
    """Signed residual -> RGB via a diverging colormap symmetric about zero
    (blue = under-predicted, red = over-predicted, white = exact)."""
    import matplotlib.cm as cm

    m = float(np.abs(res).max()) or 1.0
    return cm.get_cmap("seismic")((res / m + 1) / 2)[..., :3]


def maskf(m, device):  # flat bool mask on the given device
    return torch.from_numpy(m.reshape(-1)).to(device)


def run_stages(model, x, masks, reveal, d, max_radius, device, eb=None,
               collect=False):
    """Run the stage schedule over truth `x` (B, N); return (sum |pred - truth|,
    n_holes, known_vals, pred_only). Training passes `eb=None` and teacher-forces
    via `reveal(kv, posf)` (noisy truth). Eval passes an error bound `eb`: the
    fed-back known values are then the model's own linear-quantised
    reconstructions (real closed-loop inference, matching the codec), so the
    reported MAE is the prediction error the codec actually pays to encode.

    `collect` builds the per-pixel `pred_only` image (raw preds at the holes) for
    logging; it costs a scatter per stage and is only read when logging eval
    images, so training leaves it off and gets `pred_only=None`."""
    E = torch.zeros(x.shape[0], x.shape[1], d, device=device)
    m0 = maskf(masks[0], device)
    if eb is None:
        known_vals = reveal(torch.full_like(x, 0.5), m0)
    else:  # anchors quantised against pred 0, like the codec's stage 0
        known_vals = torch.where(m0, qz(torch.zeros_like(x), x, eb),
                                 torch.full_like(x, 0.5))
    prev = np.zeros(masks[0].shape, bool)
    known = masks[0].copy()
    # recon *without* residuals (raw preds at holes); only for image logging
    pred_only = known_vals.clone() if collect else None

    abs_err = torch.zeros(())
    npix = 0
    for pos in masks[1:]:
        if not pos.any():
            continue
        posf = maskf(pos, device)
        pidx = posf.nonzero(as_tuple=True)[0]  # holes; pred comes back compact
        pred, E = stage_forward(model, E, prev, known, known_vals,
                                max_radius, torch, predict_idx=pidx)
        tgt = x[:, pidx]
        abs_err = abs_err + (pred - tgt).abs().sum()
        npix += tgt.numel()
        if collect:
            pred_only[:, pidx] = pred
        prev = known.copy()
        known = known | pos
        if eb is None:
            known_vals = reveal(known_vals, posf)
        else:
            known_vals = known_vals.clone()
            known_vals[:, pidx] = qz(pred, tgt, eb)
    # known_vals = full closed-loop recon; pred_only = same minus the quantised
    # residual added at each hole (residual = known_vals - pred_only).
    return abs_err, npix, known_vals, pred_only


def load_eval_plane(path, device):
    """Whole-image luma plane from the eval image as ((1, N), (h, w))."""
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w, _ = a.shape
    luma = a @ np.array([0.299, 0.587, 0.114], np.float32)
    return torch.from_numpy(luma.reshape(1, -1)).to(device), (h, w)


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
    ap.add_argument("--baseline", choices=("cubic", "linear"), default="cubic",
                    help="SZ-style interpolation used for the reference line")
    ap.add_argument("--eval-image", default=str(Path(__file__).resolve().parent
                    .parent / "data" / "kodak" / "kodim17.png"),
                    help="held-out image the model + baseline are evaluated on")
    ap.add_argument("--eval-every", type=int, default=50,
                    help="evaluate model MAE on the eval image every N steps")
    ap.add_argument("--img-every", type=int, default=500,
                    help="log eval reconstruction images every N steps")
    ap.add_argument("--eval-eb", type=float, default=0.01,
                    help="error bound (in [0,1]) for the real-inference eval loop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=int, default=None,
                    help="fix the stage levels (default: random 3/4/5 per step; "
                         "set it for a deterministic, comparable profile)")
    ap.add_argument("--stride", type=int, default=None,
                    help="fix the anchor stride (default: random 8/16/32)")
    ap.add_argument("--workers", type=int, default=4,
                    help="background image-decode threads (0 = synchronous, "
                         "bit-reproducible; >0 overlaps I/O with the GPU step)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                    default="online", help="wandb logging mode")
    ap.add_argument("--wandb-project", default="gnn-sz")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--profile", type=int, default=0,
                    help="profile this many steps with torch.profiler, print "
                         "the op table + write trace.json, then exit")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    import wandb
    wandb.init(project=args.wandb_project, name=args.run_name,
               mode=args.wandb_mode, config=vars(args))

    device = torch.device(args.device)
    paths = list_images(args.data)
    if not paths:
        raise SystemExit(f"no images found under {args.data}")
    print(f"{len(paths)} images, device={device}")

    model = build_model(args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    decay_start = int(args.steps * 0.8)  # linear lr decay over the last 20% steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        1.0, (args.steps - s) / max(args.steps - decay_start, 1)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} params, d={args.d}")

    # Fixed held-out eval: one Kodak luma patch, fixed masks. Both the model and
    # the interp baseline run a real closed-loop inference with linear
    # quantisation at --eval-eb (predictions fed the *quantised* recon of earlier
    # stages, never the truth), so the reported MAE is the prediction error the
    # codec pays. The interp baseline is model-independent -> compute it once and
    # draw it as a reference line.
    eval_x, (eh, ew) = load_eval_plane(args.eval_image, device)
    eval_masks = stage_masks((eh, ew), 4, 16, anchor_block=1)

    be, bn = interp_eval(eval_x, eval_masks, args.eval_eb, args.baseline)
    baseline_mae = be / max(bn, 1)
    eval_name = Path(args.eval_image).name
    print(f"{args.baseline} interp baseline on "
          f"{eval_name}: MAE={baseline_mae:.5f} (eb={args.eval_eb})")
    wandb.summary["baseline_mae"] = baseline_mae
    truth_img = eval_x.detach().cpu().numpy().reshape(eh, ew)
    wandb.log({"eval/truth": wandb.Image(truth_img, caption=eval_name)}, step=0)

    run_loss = 0.0
    last_eval = float("nan")

    prof = None
    if args.profile:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=acts,
            schedule=torch.profiler.schedule(wait=1, warmup=1,
                                             active=args.profile),
            record_shapes=True, with_stack=True)
        prof.start()
        args.steps = args.profile + 2  # wait + warmup + active

    batches = prefetch_batches(paths, args.batch, args.crop, args.steps,
                               args.workers)
    bar = tqdm(range(1, args.steps + 1), desc="train")
    for step in bar:
        x = next(batches).to(device)  # (B, N) truth, decoded on a worker thread
        levels = args.levels or random.choice((3, 4, 5))
        stride = args.stride or random.choice((8, 16, 32))
        masks = stage_masks((args.crop, args.crop), levels, stride, anchor_block=1)

        def reveal(kv, posf):
            noise = (torch.rand_like(x) * 2 - 1) * args.noise
            return torch.where(posf, (x + noise).clamp(0, 1), kv)

        # pixel-weighted MAE: sum absolute error over every hole across all
        # stages, divide by total holes, so each pixel counts once (the dense
        # final stages dominate, matching the L1 the quantizer really pays).
        abs_err, npix, _, _ = run_stages(model, x, masks, reveal, args.d,
                                         args.max_radius, device)
        loss = abs_err / max(npix, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.item()
        wandb.log({"train/mae": loss.item(), "lr": sched.get_last_lr()[0]},
                  step=step)

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                ae, en, recon, pred_only = run_stages(
                    model, eval_x, eval_masks, None, args.d, args.max_radius,
                    device, eb=args.eval_eb, collect=True)
            model.train()
            last_eval = ae.item() / max(en, 1)
            log = {"eval/mae": last_eval,
                   "eval/mae_vs_baseline": last_eval / baseline_mae}
            if step % args.img_every == 0:  # img-every: use a multiple of eval-every
                recon = recon.detach().cpu().numpy().reshape(eh, ew)
                pred_only = pred_only.detach().cpu().numpy().reshape(eh, ew)
                # prediction reconstruction (no residuals) + the quantised
                # residuals it added, on a diverging map centred at zero.
                log["eval/recon_pred"] = wandb.Image(pred_only.clip(0, 1))
                log["eval/residual"] = wandb.Image(residual_rgb(recon - pred_only))
            wandb.log(log, step=step)

        if step % 100 == 0:
            bar.set_postfix(mae=f"{run_loss / 100:.5f}",
                            eval=f"{last_eval:.5f}")
            run_loss = 0.0

        if prof is not None:
            prof.step()

    if prof is not None:
        prof.stop()
        sort_key = ("cuda_time_total" if device.type == "cuda"
                    else "cpu_time_total")
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
        prof.export_chrome_trace("trace.json")
        print("wrote trace.json (open in chrome://tracing or perfetto.dev)")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d": args.d}, out)
    print(f"saved {out}")
    wandb.save(str(out))
    wandb.finish()


if __name__ == "__main__":
    main()
