"""Train the lightweight GNN predictor (deepsz/gnn_predictor.py) on natural
images. CPU-friendly: the model is ~20k params, a few thousand steps suffice.

    conda run -n nf python scripts/train_gnn.py --data /path/to/images

Each step crops random 128x128 patches, picks a random colour plane
(R / G / B / grayscale luma) so the net trains on grayscale as well as single
colour channels, teacher-forces the hierarchical stage schedule (levels.py)
with true values (+ quantization-like noise at the sampled error bound) and
minimises discretized-Laplacian NLL over the hole positions of every refinement
stage. Checkpoint -> data/gnn_predictor.pt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepsz.gnn_predictor import (CKPT_VERSION, build_model, build_stage_geoms,
                                  stage_forward)
from deepsz.levels import stage_masks

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
    """Linear-quantised reconstruction (deepsz.quantizer): pred + the residual
    snapped to the 2*eb grid, so |recon - truth| <= eb. Works on np or torch."""
    round_ = torch.round if torch.is_tensor(truth) else np.round
    if torch.is_tensor(eb) and torch.is_tensor(truth) and eb.ndim == 1:
        eb = eb.reshape(-1, *([1] * (truth.ndim - 1)))
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


def _batch_scalar(value, batch, device):
    if torch.is_tensor(value):
        value = value.to(device=device, dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            return value.expand(batch)
        return value.reshape(batch)
    return torch.full((batch,), float(value), dtype=torch.float32, device=device)


def sample_noise(batch, args, device):
    """Per-sample normalized teacher-forcing noise range."""
    if args.noise_range is None:
        return _batch_scalar(args.noise, batch, device)
    lo, hi = args.noise_range
    log_lo, log_hi = np.log(float(lo)), np.log(float(hi))
    return torch.empty(batch, device=device).uniform_(log_lo, log_hi).exp()


def discretized_laplace_nll(mu, log_b, target, eb):
    """Mean code length in bits for target under a discretized Laplace cell."""
    eb = torch.as_tensor(eb, device=target.device, dtype=target.dtype).reshape(-1, 1)
    b = torch.exp2(log_b).clamp_min(torch.finfo(target.dtype).tiny)
    log_half = -math.log(2.0)
    log2e = 1.0 / math.log(2.0)

    abs_r = (target - mu).abs()
    rho = abs_r / b
    e = eb / b

    # |residual| >= eb: quantization cell is wholly on one side of zero.
    tail_mass = torch.log1p(-torch.exp((-2.0 * e).clamp_max(
        -torch.finfo(target.dtype).eps)))
    logp_tail = log_half - (rho - e) + tail_mass

    # |residual| < eb: quantization cell straddles zero. Clamp rho to this
    # branch's domain before evaluating so torch.where's eager branch execution
    # cannot overflow on far-tail residuals.
    rho_cross = torch.minimum(rho, e)
    cross_p = (1.0 - 0.5 * torch.exp(-(e - rho_cross))
               - 0.5 * torch.exp(-(e + rho_cross)))
    logp_cross = torch.log(cross_p.clamp_min(1e-30))

    logp = torch.where(abs_r >= eb, logp_tail, logp_cross)
    return -logp * log2e


def entropy_bits_from_codes(codes):
    """Order-0 entropy H(codes), in bits/symbol."""
    if not codes:
        return float("nan")
    flat = torch.cat([c.reshape(-1).detach().cpu() for c in codes])
    if flat.numel() == 0:
        return float("nan")
    _, counts = torch.unique(flat, return_counts=True)
    p = counts.to(torch.float64) / flat.numel()
    return float(-(p * torch.log2(p)).sum())


def run_stages(model, x, geoms, d, device, eb, teacher_force=False,
               collect=False, collect_bins=False):
    """Run the stage schedule over truth `x` (B, N) using precomputed per-stage
    geometry `geoms` (from ``build_stage_geoms``); return (sum NLL bits,
    n_holes, known_vals, pred_only, aux). ``aux`` includes summed prediction
    absolute error for diagnostic MAE logging. Training passes
    ``teacher_force=True`` and feeds truth plus uniform +/- eb noise. Eval
    leaves teacher forcing off: the fed-back known values are the model's own
    linear-quantised reconstructions (real closed-loop inference, matching the
    codec).

    Mirrors the codec's evolution: at stage i we finalize the points revealed in
    stage i-1 into the field, then predict stage i. The head context of stage i-1
    equals the finalize context of stage i (same field, same geometry), so it is
    fed back as `finalize_ctx` instead of being pooled twice.

    `collect` builds the per-pixel `pred_only` image (raw preds at the holes) for
    logging; it costs a scatter per stage and is only read when logging eval
    images, so training leaves it off and gets `pred_only=None`."""
    B, N = x.shape
    eb = _batch_scalar(eb, B, device)
    E = torch.zeros(B, N, geoms[0].ndim, d, device=device)
    a0 = geoms[0].query_idx                      # anchors
    known_vals = torch.full_like(x, 0.5)

    def reveal(idx):  # teacher-force truth (+ training noise) at `idx`
        nz = (torch.rand(B, idx.numel(), device=device) * 2 - 1) * eb[:, None]
        known_vals[:, idx] = (x[:, idx] + nz).clamp(0, 1)

    if teacher_force:
        reveal(a0)
    else:  # anchors quantised against pred 0, like the codec's stage 0
        known_vals[:, a0] = qz(torch.zeros(B, a0.numel(), device=device),
                               x[:, a0], eb)
    # recon *without* residuals (raw preds at holes); only for image logging
    pred_only = known_vals.clone() if collect else None

    nll_sum = torch.zeros((), device=device)
    abs_err = torch.zeros((), device=device)
    npix = 0
    bins = []
    head_ctx = None
    for i in range(1, len(geoms)):
        gp, gh = geoms[i - 1], geoms[i]
        (pred, log_b), E, head_ctx = stage_forward(
            model, E, gp, gh, known_vals[:, gp.query_idx], torch,
            finalize_ctx=head_ctx, eb=eb)
        idx = gh.query_idx
        tgt = x[:, idx]
        nll = discretized_laplace_nll(pred, log_b, tgt, eb)
        nll_sum = nll_sum + nll.sum()
        abs_err = abs_err + (pred.detach() - tgt).abs().sum()
        npix += tgt.numel()
        if collect_bins:
            bins.append(torch.round((tgt - pred) / (2 * eb[:, None])).to(torch.int64))
        if collect:
            pred_only[:, idx] = pred
        if teacher_force:
            reveal(idx)
        else:
            known_vals[:, idx] = qz(pred, tgt, eb)
    # known_vals = full closed-loop recon; pred_only = same minus the quantised
    # residual added at each hole (residual = known_vals - pred_only).
    return nll_sum, npix, known_vals, pred_only, {"bins": bins, "abs_err": abs_err}


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
                    help="fixed uniform +/- noise on known values, in normalized "
                         "[0,1] units")
    ap.add_argument("--noise-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                    default=None,
                    help="sample each training example's +/- noise log-uniformly "
                         "from [MIN, MAX], overriding --noise")
    ap.add_argument("--max-radius", type=int, default=64)
    ap.add_argument("--baseline", choices=("cubic", "linear"), default="cubic",
                    help="SZ-style interpolation used for the reference line")
    ap.add_argument("--eval-image", default=str(Path(__file__).resolve().parent
                    .parent / "data" / "kodak" / "kodim17.png"),
                    help="held-out image the model + baseline are evaluated on")
    ap.add_argument("--eval-every", type=int, default=50,
                    help="evaluate model bpp on the eval image every N steps")
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
    if args.noise_range is not None:
        lo, hi = args.noise_range
        if lo <= 0 or hi < lo:
            raise SystemExit("--noise-range must satisfy 0 < MIN <= MAX")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # per-run dir: <out-parent>/runs/<date>-<config-hash>/ holds the checkpoint
    # and a config.json snapshot, so concurrent/repeated runs never clobber.
    cfg_hash = hashlib.sha1(repr(sorted(vars(args).items())).encode()).hexdigest()[:6]
    run_dir = (Path(args.out).resolve().parent / "runs"
               / f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg_hash}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    out = run_dir / Path(args.out).name
    print(f"run dir: {run_dir}")

    wandb_config = {k: v for k, v in vars(args).items()
                    if k not in {"noise", "noise_range"}}

    import wandb
    wandb.init(project=args.wandb_project, name=args.run_name,
               mode=args.wandb_mode, config=wandb_config, dir=str(run_dir))

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
    eval_masks = stage_masks((eh, ew), 4, 16, anchor_block=1)  # interp baseline
    eval_geoms, _ = build_stage_geoms((eh, ew), 4, 16, 1, args.max_radius,
                                      torch, device)

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
        # stride and levels must be paired so the schedule densifies to stride 1
        # (levels == log2(stride)); independent draws mismatch and train the net
        # on broken schedules (see levels.stage_plan's guard).
        stride = args.stride or random.choice((8, 16, 32))
        levels = args.levels or stride.bit_length() - 1
        geoms, _ = build_stage_geoms((args.crop, args.crop), levels, stride, 1,
                                     args.max_radius, torch, device)

        # Pixel-weighted discretized-Laplacian NLL: the sampled eb both
        # conditions the head and sets the teacher-forcing noise amplitude.
        eb = sample_noise(x.shape[0], args, device)
        nll, npix, _, _, aux = run_stages(model, x, geoms, args.d, device,
                                          eb=eb, teacher_force=True)
        loss = nll / max(npix, 1)
        train_mae = aux["abs_err"].detach() / max(npix, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.item()
        wandb.log({
            "train/bpp": loss.item(),
            "train/mae": train_mae.item(),
            "lr": sched.get_last_lr()[0],
        }, step=step)

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                bits, en, recon, pred_only, aux = run_stages(
                    model, eval_x, eval_geoms, args.d, device,
                    eb=args.eval_eb, collect=True, collect_bins=True)
            model.train()
            # eval runs the whole image (>>train crop); free its reserved pool
            # so the next train step doesn't OOM on the fragmented remainder.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            last_eval = bits.item() / max(en, 1)
            eval_mae = aux["abs_err"].item() / max(en, 1)
            marginal = entropy_bits_from_codes(aux["bins"])
            log = {"eval/bpp_model": last_eval,
                   "eval/bpp_marginal": marginal,
                   "eval/bpp_gain_frac": ((marginal - last_eval) / marginal
                                          if marginal > 0 else float("nan"))}
            log["eval/mae"] = eval_mae
            if step % args.img_every == 0:  # img-every: use a multiple of eval-every
                recon = recon.detach().cpu().numpy().reshape(eh, ew)
                pred_only = pred_only.detach().cpu().numpy().reshape(eh, ew)
                # prediction reconstruction (no residuals) + the quantised
                # residuals it added, on a diverging map centred at zero.
                log["eval/recon_pred"] = wandb.Image(pred_only.clip(0, 1))
                log["eval/residual"] = wandb.Image(residual_rgb(recon - pred_only))
            wandb.log(log, step=step)

        if step % 100 == 0:
            bar.set_postfix(bpp=f"{run_loss / 100:.5f}",
                            eval_bpp=f"{last_eval:.5f}")
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

    torch.save({
        "state_dict": model.state_dict(),
        "d": args.d,
        "version": CKPT_VERSION,
    }, out)
    print(f"saved {out}")
    wandb.save(str(out))
    wandb.finish()


if __name__ == "__main__":
    main()
