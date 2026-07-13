"""Train the lightweight GNN predictor (deepsz/gnn_predictor.py) on natural
images mixed with anisotropic synthetic 4-D fields. CPU-friendly: the model is
~20k params, a few thousand steps suffice.

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
import contextlib
import hashlib
import json
import math
import os
import random
import threading
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
from deepsz.gnn_predictor import (CKPT_VERSION, _CompactFrame, anchor_finalize,
                                  build_chunk_geoms, build_model,
                                  build_stage_geoms, chunk_coarse,
                                  stage_forward)
from deepsz.levels import stage_masks

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}


def list_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT]


def load_tensor(path: str) -> np.ndarray:
    """Load a raw n-D tensor from .npy or a torch .pt/.pth for the codec eval."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".pt", ".pth")):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, torch.Tensor):
            raise ValueError(f"{path} holds {type(obj).__name__}, expected a tensor")
        return obj.detach().cpu().numpy()
    raise ValueError(f"unsupported extension: {path} (use .npy/.pt/.pth)")


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


def sample_synthetic_batch(batch, shape, correlation, rng=None,
                           permute_axes=True):
    """Return smooth anisotropic Gaussian random fields in ``[0, 1]``.

    The entries of ``correlation`` are Gaussian smoothing sigmas in grid cells;
    a larger value makes adjacent values more correlated. By default their axis
    assignment is independently shuffled for each field, preventing the model
    from memorizing that one fixed axis is always the most useful. Two filtered
    scales keep samples richer than a single blurred-noise realization.
    Reflecting boundaries avoid teaching the model an artificial periodic seam.
    """
    from scipy.ndimage import gaussian_filter

    shape = tuple(int(n) for n in shape)
    correlation = tuple(float(s) for s in correlation)
    rng = np.random.default_rng() if rng is None else rng
    fields = []
    for _ in range(batch):
        sigma = correlation
        if permute_axes:
            sigma = tuple(np.asarray(correlation)[rng.permutation(len(shape))])
        fine = gaussian_filter(rng.standard_normal(shape), sigma=sigma,
                               mode="reflect")
        coarse = gaussian_filter(rng.standard_normal(shape),
                                 sigma=tuple(2.0 * s for s in sigma),
                                 mode="reflect")
        field = fine + 0.5 * coarse
        # Robust scaling retains local variation when one realization contains
        # a rare extreme. Values match the normalized image training range.
        lo, hi = np.percentile(field, (1.0, 99.0))
        if hi <= lo:  # only realistically reachable for degenerate tiny shapes
            field = np.full(shape, 0.5, np.float32)
        else:
            field = np.clip((field - lo) / (hi - lo), 0.0, 1.0)
        fields.append(field.astype(np.float32, copy=False))
    return torch.from_numpy(np.stack(fields)).reshape(batch, -1)


def mixed_batch_sizes(crop, synthetic_shape, synthetic_fraction,
                      image_batch, synthetic_batch):
    """Choose source batch sizes and return their actual point fraction.

    A 2-D crop and a 4-D field are indivisible training examples, so arbitrary
    fractions can only be approximated at that granularity. ``synthetic_batch``
    fixes the number of fields; the image batch is derived to make the fraction
    of scalar points as close as possible to ``synthetic_fraction``. At the two
    endpoints the explicitly configured batch size for the active source is
    retained.
    """
    fraction = float(synthetic_fraction)
    if fraction == 0.0:
        return int(image_batch), 0, 0.0
    if fraction == 1.0:
        return 0, int(synthetic_batch), 1.0
    image_points = int(crop) ** 2
    field_points = math.prod(int(n) for n in synthetic_shape)
    synthetic_batch = int(synthetic_batch)
    target_image_points = (synthetic_batch * field_points
                           * (1.0 - fraction) / fraction)
    image_batch = max(1, round(target_image_points / image_points))
    ns = synthetic_batch * field_points
    ni = image_batch * image_points
    return image_batch, synthetic_batch, ns / (ni + ns)


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


def training_autocast(fp16, device):
    """CUDA FP16 autocast matching the codec's message-pass execution mode."""
    if fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


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


class _PeakRSS:
    """Sample process resident-set size in a background thread and hold the peak
    (in MB) reached over the ``with`` body. Linux-only (/proc/self/statm); the
    training host is Linux. Captures allocator peaks the torch CUDA counters
    miss (host-side numpy/zstd/rANS buffers)."""

    _PAGE = os.sysconf("SC_PAGE_SIZE")

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _rss_mb(self) -> float:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * self._PAGE / 1e6

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._rss_mb())

    def __enter__(self):
        self.peak = self._rss_mb()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss_mb())


def eval_tensor_codec(model, d, args, tensor, eb, device, ckpt_path):
    """Roundtrip `tensor` (a small n-D array, e.g. 4-D) through the *real* codec
    at the current weights and return distortion / rate / peak-RAM / time metrics
    for wandb. Unlike the closed-loop image eval, this exercises the full
    compress+decompress byte path (rANS, zstd, chunking), so the rate is the
    actual stream size and the RAM/time are what the deployed codec pays.

    The codec loads weights from disk, so the live model is frozen to `ckpt_path`
    first (overwritten each call; the model cache keys on mtime so this reloads).
    """
    from deepsz.gnn_codec import GNNCompressorCodec

    torch.save({"state_dict": model.state_dict(), "d": d,
                "version": CKPT_VERSION}, ckpt_path)
    codec = GNNCompressorCodec(
        ckpt_path, error_bound=eb, levels=(args.levels or 4),
        anchor_stride=(args.stride or 16), max_radius=args.max_radius,
        device=str(device), agg_level=args.agg_level,
        fp16=args.fp16, compile=args.compile)

    base_gpu = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_gpu = torch.cuda.memory_allocated()

    with _PeakRSS() as rss:
        t0 = time.time()
        stream = codec.compress(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_comp = time.time() - t0
        t0 = time.time()
        rec = codec.uncompress(stream).numpy().reshape(tensor.shape)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_dec = time.time() - t0

    a = tensor.astype(np.float64)
    r = rec.astype(np.float64)
    max_err = float(np.abs(a - r).max())
    mse = float(np.mean((a - r) ** 2))
    vrange = max(float(a.max()) - float(a.min()), 1e-12)
    psnr = 10 * np.log10(vrange ** 2 / mse) if mse > 0 else float("inf")
    bpv = 8 * len(stream) / tensor.size  # tensor is float32 -> 32 bpv raw

    metrics = {
        "eval_tensor/psnr_db": psnr,          # distortion
        "eval_tensor/max_abs_err": max_err,   # distortion (bound check: <= eb)
        "eval_tensor/bits_per_value": bpv,    # rate
        "eval_tensor/ratio": 4 * tensor.size / len(stream),  # rate (vs f32)
        "eval_tensor/compress_s": t_comp,     # inference time (encode)
        "eval_tensor/decompress_s": t_dec,    # inference time (decode)
        "eval_tensor/peak_host_mb": rss.peak,  # max ram (host RSS peak)
    }
    if device.type == "cuda":
        metrics["eval_tensor/peak_gpu_mb"] = (
            torch.cuda.max_memory_allocated() - base_gpu) / 1e6  # max ram (gpu)
        torch.cuda.empty_cache()
    return metrics


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


def _run_chunk(model, cg, geoms, E, known_vals, x, gidx, eb, device, reveal):
    """One chunk of the chunked-scene step: the codec's local stage chain with
    teacher forcing. Returns (nll bits, n holes, abs err, finalized E)."""
    nll = torch.zeros((), device=device)
    abs_err = torch.zeros((), device=device)
    npix = 0
    ctx = None
    for j in range(1, len(cg.chain)):
        prev, s = cg.chain[j - 1], cg.chain[j]
        gp, gh = geoms[prev], geoms[s]
        fvals = None if gp is None else known_vals[:, gidx[prev]]
        (pred, log_b), E, ctx = stage_forward(model, E, gp, gh, fvals, torch,
                                              finalize_ctx=ctx, eb=eb)
        tgt = x[:, gidx[s]]
        nll = nll + discretized_laplace_nll(pred, log_b, tgt, eb).sum()
        abs_err = abs_err + (pred.detach() - tgt).abs().sum()
        npix += tgt.numel()
        reveal(gidx[s])
    last = cg.chain[-1]
    g = geoms[last]
    if g is not None:  # finalize the last stage so the coarse means see it
        if ctx is None:
            ctx = model.embed(E, g)
        finalized = model.finalize(ctx, known_vals[:, gidx[last]]).to(E.dtype)
        E = E.index_copy(1, g.query_idx,
                         finalized)
    return nll, npix, abs_err, E


def run_chunked_scene(model, x, hw, axis, order, levels, stride, d, device, eb,
                      agg_level=None):
    """Teacher-forced two-chunk closed loop, mirroring the chunked codec: the
    square scene is split in half along ``axis``; global anchors are revealed
    and give every chunk its level-0 coarse embedding, then the chunks are
    coded in ``order`` — the first sees only anchor context across the border,
    the second sees the first's per-level coarse embeddings as coded halo.
    Same geometry/halo/coarse code as ChunkedGNNPredictor, with gradients."""
    B, N = x.shape
    h, w = hw
    eb = _batch_scalar(eb, B, device)
    edges = (h, w // 2) if axis == 1 else (h // 2, w)
    grid = (1, 2) if axis == 1 else (2, 1)
    known_vals = torch.full_like(x, 0.5)

    def reveal(idx):
        if not torch.is_tensor(idx):
            idx = torch.from_numpy(np.asarray(idx, np.int64)).to(device)
        nz = (torch.rand(B, idx.numel(), device=device) * 2 - 1) * eb[:, None]
        known_vals[:, idx] = (x[:, idx] + nz).clamp(0, 1)

    # global anchor pass + per-chunk level-0 coarse
    coarse = torch.zeros(B, 2, levels + 1, 2, d, device=device)
    origins, aidx = [], []
    for ci in range(2):
        starts = tuple(g * e for g, e in zip(np.unravel_index(ci, grid), edges))
        origins.append(starts)
        axes = [o + np.arange(e)[(np.arange(e) + o) % stride == 0]
                for o, e in zip(starts, edges)]
        aidx.append((axes[0][:, None] * w + axes[1][None, :]).reshape(-1))
    for ci in range(2):
        reveal(aidx[ci])
    for ci in range(2):
        fin = anchor_finalize(model, known_vals[:, aidx[ci]], 2)
        coarse[:, ci, 0] = model.coarse(fin.mean(1), math.log2(stride))

    cg = build_chunk_geoms(edges, levels, stride, 1, torch, device, agg_level)
    coded = np.zeros(2, bool)
    nll = torch.zeros((), device=device)
    abs_err = torch.zeros((), device=device)
    npix = 0
    for ci in order:
        frame = _CompactFrame(cg, origins[ci], hw, edges, grid, coded,
                              torch, device)
        E = torch.zeros(B, frame.n_compact, 2, d, device=device)
        if len(frame.h_gflat):                             # trailing halo block
            ids_t = torch.from_numpy(frame.h_ids).to(device)
            lv_t = torch.from_numpy(frame.h_lv.astype(np.int64)).to(device)
            gflat_t = torch.from_numpy(frame.h_gflat).to(device)
            cvec = coarse[:, ids_t, lv_t]                  # (B, Hs, K, d)
            halo = model.finalize(cvec, known_vals[:, gflat_t])
            E = torch.cat([E[:, :frame.halo_rows.start], halo], dim=1)
        gidx = [None if c is None else torch.from_numpy(np.ravel_multi_index(
            [(c[:, k] + origins[ci][k]) for k in range(2)], hw)).to(device)
            for c in cg.coords]
        n1, np1, a1, E = _run_chunk(model, cg, frame.geoms, E, known_vals, x,
                                    gidx, eb, device, reveal)
        nll, npix, abs_err = nll + n1, npix + np1, abs_err + a1
        coarse[:, ci] = chunk_coarse(model, E, cg, torch)
        coded[ci] = True
    return nll, npix, {"abs_err": abs_err}


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
    ap.add_argument("--synthetic-frac", type=float, default=0.5,
                    help="target fraction of scalar training points from 4-D "
                         "fields in every optimizer step; the image batch size "
                         "is derived to approach this fraction")
    ap.add_argument("--synthetic-shape", type=int, nargs=4,
                    default=(16, 16, 16, 16), metavar=("N0", "N1", "N2", "N3"),
                    help="shape of each generated 4-D training field")
    ap.add_argument("--synthetic-correlation", type=float, nargs=4,
                    default=(6.0, 3.0, 1.5, 0.75),
                    metavar=("S0", "S1", "S2", "S3"),
                    help="Gaussian correlation length (grid cells) per axis; "
                         "larger means smoother/more correlated")
    ap.add_argument("--synthetic-batch", type=int, default=1,
                    help="number of 4-D fields per optimizer step; with a mixed "
                         "step, --batch is derived from the target point fraction")
    ap.add_argument("--synthetic-stride", type=int, default=8,
                    help="anchor stride for synthetic fields (power of two)")
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
    ap.add_argument("--agg-level", type=int, default=2,
                    help="maximum L1 neighbourhood aggregation level: 1 uses "
                         "axis-aligned directions only; 2 also uses two-axis "
                         "diagonals (default: 2)")
    ap.add_argument("--chunk-frac", type=float, default=0.5,
                    help="fraction of steps trained on the two-chunk scene "
                         "(chunked-codec halo regime); the rest train the "
                         "whole-crop schedule")
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
    ap.add_argument("--eval-tensor", default=None,
                    help="small n-D tensor (.npy/.pt, e.g. a 4-D field) roundtripped "
                         "through the full codec each --eval-tensor-every steps; logs "
                         "distortion / rate / peak-RAM / inference-time to wandb")
    ap.add_argument("--eval-tensor-eb", type=float, default=None,
                    help="absolute error bound for --eval-tensor (default: --eval-eb)")
    ap.add_argument("--eval-tensor-every", type=int, default=500,
                    help="run the --eval-tensor codec roundtrip every N steps "
                         "(full compress+decompress; keep it a multiple of eval-every)")
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
    ap.add_argument("--fp16", action="store_true",
                    help="train the GNN message pass with CUDA FP16 autocast "
                         "and gradient scaling; readout/loss remain FP32")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the message-pass embed with dynamic "
                         "shapes (one-off compilation cost on first steps)")
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
    if not 0.0 <= args.synthetic_frac <= 1.0:
        raise SystemExit("--synthetic-frac must be in [0, 1]")
    if any(n < 2 for n in args.synthetic_shape):
        raise SystemExit("--synthetic-shape entries must all be >= 2")
    if any(s <= 0 for s in args.synthetic_correlation):
        raise SystemExit("--synthetic-correlation entries must all be > 0")
    if args.synthetic_batch < 1:
        raise SystemExit("--synthetic-batch must be >= 1")
    if (args.synthetic_stride < 2 or
            args.synthetic_stride & (args.synthetic_stride - 1)):
        raise SystemExit("--synthetic-stride must be a power of two >= 2")
    if args.agg_level < 1:
        raise SystemExit("--agg-level must be >= 1")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    synthetic_rng = np.random.default_rng(args.seed + 1)

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
    amp_enabled = args.fp16 and device.type == "cuda"
    if args.fp16 and not amp_enabled:
        print("NOTE: --fp16 only engages on CUDA; using FP32 on "
              f"device={device}")
    paths = list_images(args.data)
    if not paths:
        raise SystemExit(f"no images found under {args.data}")
    print(f"{len(paths)} images, device={device}")
    image_batch, synthetic_batch, synthetic_point_fraction = mixed_batch_sizes(
        args.crop, args.synthetic_shape, args.synthetic_frac,
        args.batch, args.synthetic_batch)
    if args.synthetic_frac:
        image_points = image_batch * args.crop ** 2
        synthetic_points = synthetic_batch * math.prod(args.synthetic_shape)
        print("synthetic mix: "
              f"target={args.synthetic_frac:g}, "
              f"actual point fraction={synthetic_point_fraction:.6g}, "
              f"image={image_batch}x{args.crop}^2={image_points} points, "
              f"synthetic={synthetic_batch}x{tuple(args.synthetic_shape)}="
              f"{synthetic_points} points, "
              f"correlation={tuple(args.synthetic_correlation)}, "
              f"stride={args.synthetic_stride}")

    model = build_model(args.d).to(device)
    if args.compile:
        mode = os.environ.get("DEEPSZ_COMPILE_MODE") or None
        model.embed = torch.compile(model.embed, dynamic=True, mode=mode)
        print(f"torch.compile enabled for model.embed (mode={mode or 'default'})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    decay_start = int(args.steps * 0.7)  # linear lr decay over the last 30% steps
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
                                      torch, device, args.agg_level)

    be, bn = interp_eval(eval_x, eval_masks, args.eval_eb, args.baseline)
    baseline_mae = be / max(bn, 1)
    eval_name = Path(args.eval_image).name
    print(f"{args.baseline} interp baseline on "
          f"{eval_name}: MAE={baseline_mae:.5f} (eb={args.eval_eb})")
    wandb.summary["baseline_mae"] = baseline_mae
    truth_img = eval_x.detach().cpu().numpy().reshape(eh, ew)
    wandb.log({"eval/truth": wandb.Image(truth_img, caption=eval_name)}, step=0)

    # Optional full-codec roundtrip eval on a small n-D tensor (e.g. 4-D). Loaded
    # once as float32; weights are frozen to this temp checkpoint each eval so the
    # codec (which reads weights from disk) sees the current model.
    eval_tensor = None
    eval_tensor_ckpt = run_dir / "eval_tensor.pt"
    eval_tensor_eb = (args.eval_eb if args.eval_tensor_eb is None
                      else args.eval_tensor_eb)
    if args.eval_tensor:
        eval_tensor = load_tensor(args.eval_tensor).astype(np.float32)
        print(f"eval-tensor: {args.eval_tensor} {eval_tensor.shape} "
              f"eb={eval_tensor_eb}")

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

    batches = prefetch_batches(paths, image_batch, args.crop, args.steps,
                               args.workers)
    bar = tqdm(range(1, args.steps + 1), desc="train")
    for step in bar:
        opt.zero_grad()
        # Weight each source mean by its *actual number of scalar points* in
        # this mixed batch. Thus the objective is the mean over all points used
        # by the step, including rounding needed for indivisible examples.
        image_weight = 1.0 - synthetic_point_fraction
        synthetic_weight = synthetic_point_fraction
        combined_loss = 0.0
        combined_mae = 0.0
        train_log = {}

        # The 2-D and 4-D examples cannot share a rectangular tensor batch.
        # Run them sequentially and backward immediately so their gradients
        # accumulate without retaining both computation graphs in GPU memory.
        if image_weight:
            x = next(batches).to(device)  # decoded on a worker thread
            stride = args.stride or random.choice((8, 16, 32))
            levels = args.levels or stride.bit_length() - 1
            eb = sample_noise(x.shape[0], args, device)
            with training_autocast(args.fp16, device):
                if random.random() < args.chunk_frac:
                    axis = random.randint(0, 1)
                    order = random.choice(([0, 1], [1, 0]))
                    nll, npix, aux = run_chunked_scene(
                        model, x, (args.crop, args.crop), axis, order, levels,
                        stride, args.d, device, eb, args.agg_level)
                else:
                    geoms, _ = build_stage_geoms(
                        (args.crop, args.crop), levels, stride, 1,
                        args.max_radius, torch, device, args.agg_level)
                    nll, npix, _, _, aux = run_stages(
                        model, x, geoms, args.d, device, eb=eb,
                        teacher_force=True)
            image_loss = nll / max(npix, 1)
            image_mae = aux["abs_err"].detach() / max(npix, 1)
            scaler.scale(image_weight * image_loss).backward()
            combined_loss += image_weight * image_loss.item()
            combined_mae += image_weight * image_mae.item()
            train_log.update({"train/image_bpp": image_loss.item(),
                              "train/image_mae": image_mae.item()})

        if synthetic_weight:
            field_shape = tuple(args.synthetic_shape)
            x = sample_synthetic_batch(
                synthetic_batch, field_shape,
                args.synthetic_correlation, synthetic_rng).to(device)
            stride = args.synthetic_stride
            levels = args.levels or stride.bit_length() - 1
            eb = sample_noise(x.shape[0], args, device)
            geoms, _ = build_stage_geoms(
                field_shape, levels, stride, 1, args.max_radius, torch, device,
                args.agg_level)
            with training_autocast(args.fp16, device):
                nll, npix, _, _, aux = run_stages(
                    model, x, geoms, args.d, device, eb=eb,
                    teacher_force=True)
            synthetic_loss = nll / max(npix, 1)
            synthetic_mae = aux["abs_err"].detach() / max(npix, 1)
            scaler.scale(synthetic_weight * synthetic_loss).backward()
            combined_loss += synthetic_weight * synthetic_loss.item()
            combined_mae += synthetic_weight * synthetic_mae.item()
            train_log.update({"train/synthetic_bpp": synthetic_loss.item(),
                              "train/synthetic_mae": synthetic_mae.item()})

        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += combined_loss
        train_log.update({"train/bpp": combined_loss,
                          "train/mae": combined_mae,
                          "train/synthetic_point_fraction":
                              synthetic_point_fraction,
                          "lr": sched.get_last_lr()[0]})
        wandb.log(train_log, step=step)

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad(), training_autocast(args.fp16, device):
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

        if eval_tensor is not None and step % args.eval_tensor_every == 0:
            model.eval()
            with torch.no_grad():
                tmetrics = eval_tensor_codec(model, args.d, args, eval_tensor,
                                             eval_tensor_eb, device,
                                             eval_tensor_ckpt)
            model.train()
            wandb.log(tmetrics, step=step)
            print(f"[{step}] eval-tensor: "
                  f"PSNR={tmetrics['eval_tensor/psnr_db']:.2f}dB "
                  f"bpv={tmetrics['eval_tensor/bits_per_value']:.3f} "
                  f"maxerr={tmetrics['eval_tensor/max_abs_err']:.3g} "
                  f"host={tmetrics['eval_tensor/peak_host_mb']:.0f}MB "
                  f"enc={tmetrics['eval_tensor/compress_s']:.2f}s "
                  f"dec={tmetrics['eval_tensor/decompress_s']:.2f}s")

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
