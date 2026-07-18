"""Per-level bpp + MSE breakdown, INTERP vs chunked GNN (vs skel), in fp32.

For every interpolation level we report:
  - bpp        : order-0 entropy of that level's quantized codes (+32 b/outlier)
                 = the ideal entropy-coder cost, free of rANS/zstd overhead.
  - pred-RMSE  : RMSE of the raw prediction (x - pred) BEFORE the residual is
                 coded. This is prediction QUALITY and is what drives bpp;
                 it is NOT bounded by eb.
  - rec-RMSE   : RMSE of the reconstruction (x - recon) AFTER coding. Bounded:
                 |x - recon| <= eb at every point, so rec-RMSE <= eb.

Levels are tagged by the per-stage error bound (each interpolation level has a
distinct eb under the default schedule; the anchor pass = coarsest level).
Point counts per level are analytic (point_levels).

Purpose: locate WHERE the GNN loses to interp and test whether the skeleton
codec (global 1-skeleton line context across chunk seams) recovers the
finest-level seam penalty -- run in fp32 so the fp16 message-pass noise floor
(~1e-3) no longer masks the finest predictions.

Env knobs (all optional):
  N=64  EB=1e-4  LEVELS=5  STRIDE=32  BLOCK=1  CHUNK=32  AGG=2
  CODECS="interp,gnn,skel"   TUNE=fast   FP16=0   DATA=<path.npy>
  CKPT=checkpoints/d64.pt
"""
import os, sys, time, gc, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepsz.quantizer import _recon_from_codes

CKPT   = os.environ.get("CKPT", "checkpoints/d64.pt")
EB     = float(os.environ.get("EB", "1e-4"))
N      = int(os.environ.get("N", "64"))
LEVELS = int(os.environ.get("LEVELS", "5"))
STRIDE = int(os.environ.get("STRIDE", "32"))
BLOCK  = int(os.environ.get("BLOCK", "1"))
CHUNK  = int(os.environ.get("CHUNK", "32"))
AGG    = int(os.environ.get("AGG", "2"))
TUNE   = os.environ.get("TUNE", "fast")
FP16   = os.environ.get("FP16", "0") == "1"
CODECS = [c.strip() for c in os.environ.get("CODECS", "interp,gnn,skel").split(",") if c.strip()]
DATA   = os.environ.get("DATA", "").strip()
NDIM   = 4


def make_field(n):
    """Non-periodic synthetic RTI-like field, normalised to [0,1]."""
    u = [(np.arange(n, dtype=np.float32) / (n - 1)) for _ in range(NDIM)]
    ux, uy, uz, uw = (u[0].reshape(n, 1, 1, 1), u[1].reshape(1, n, 1, 1),
                      u[2].reshape(1, 1, n, 1), u[3].reshape(1, 1, 1, n))
    h = (0.5 + 0.06 * np.sin(4.3 * np.pi * uy + 0.7) * np.sin(3.1 * np.pi * uz + 1.3)
         + 0.03 * np.sin(2.7 * np.pi * uw + 0.4)).astype(np.float32)
    base = np.tanh((ux - h) / 0.08).astype(np.float32)
    turb = (0.12 * np.sin(5.7 * ux + 2.1 * uy + 3.3 * uz + 1.9 * uw)
            + 0.07 * np.sin(9.1 * ux - 4.4 * uy + 2.2 * uz)
            + 0.04 * np.sin(13.7 * uy + 6.6 * uz - 5.1 * uw)).astype(np.float32)
    f = base + turb
    f = (f - f.min()) / (f.max() - f.min())
    return np.ascontiguousarray(f.astype(np.float32))


def load_field():
    if DATA:
        a = np.load(DATA).astype(np.float32)
        if a.ndim != NDIM:
            raise SystemExit(f"DATA must be {NDIM}-D, got shape {a.shape}")
        a = a[tuple(slice(0, N) for _ in range(NDIM))]
        a = (a - a.min()) / (a.max() - a.min())
        return np.ascontiguousarray(a.astype(np.float32))
    return make_field(N)


def analytic_pts(n):
    from deepsz.levels import point_levels
    c = np.indices((n,) * NDIM)
    lv = point_levels([c[i] for i in range(NDIM)], LEVELS, STRIDE, BLOCK)
    return np.bincount(lv.ravel(), minlength=LEVELS + 1)


class LevelStats:
    """Per-eb histogram of codes + running sums for pred/recon squared error."""
    def __init__(self):
        self.hist = collections.defaultdict(collections.Counter)
        self.nout = collections.defaultdict(int)
        self.sse_pred = collections.defaultdict(float)   # sum (x-pred)^2
        self.sse_rec = collections.defaultdict(float)    # sum (x-recon)^2
        self.npt = collections.defaultdict(int)

    def add(self, eb, x, pred, codes, radius):
        e = round(float(eb), 15)
        x = np.asarray(x, dtype=np.float64).ravel()
        pred = np.asarray(pred, dtype=np.float64).ravel()
        codes = np.asarray(codes).ravel()
        # ideal-entropy histogram
        vals, cnts = np.unique(codes, return_counts=True)
        c = self.hist[e]
        for v, nn in zip(vals.tolist(), cnts.tolist()):
            c[v] += nn
        is_out = codes == 0
        self.nout[e] += int(is_out.sum())
        # prediction error (drives bpp)
        self.sse_pred[e] += float(np.sum((x - pred) ** 2))
        # reconstruction error (bounded by eb): recon = decoder arithmetic,
        # outliers reproduce x exactly.
        recon = _recon_from_codes(pred.astype(np.float32),
                                  codes.astype(np.uint32), float(eb), int(radius))
        recon = recon.astype(np.float64)
        recon[is_out] = x[is_out]
        self.sse_rec[e] += float(np.sum((x - recon) ** 2))
        self.npt[e] += x.size

    def per_level(self):
        out = {}
        for lv, e in enumerate(sorted(self.hist.keys())):   # smallest eb = level 0
            c = self.hist[e]
            counts = np.array(list(c.values()), dtype=np.float64)
            tot = counts.sum()
            p = counts / tot
            ent = float(-(counts * np.log2(p)).sum())
            bpp = (ent + 32.0 * self.nout[e]) / tot
            pred_rmse = (self.sse_pred[e] / self.npt[e]) ** 0.5
            rec_rmse = (self.sse_rec[e] / self.npt[e]) ** 0.5
            out[lv] = dict(bpp=bpp, ent_bpp=ent / tot, tot=int(tot),
                           nout=self.nout[e], pred_rmse=pred_rmse,
                           rec_rmse=rec_rmse, eb=e)
        return out


def hook_all(stats):
    """Wrap `quantize` in every module that owns a reference, restore on exit."""
    import deepsz.codec as cc
    mods = [cc]
    for name in ("gnn_codec", "skel_codec"):
        try:
            mods.append(__import__("deepsz." + name, fromlist=[name]))
        except Exception:
            pass
    saved = []
    for m in mods:
        if not hasattr(m, "quantize"):
            continue
        orig = m.quantize
        saved.append((m, orig))

        def make(orig):
            def q(x, pred, eb, radius=1 << 15, *a, **k):
                codes, outliers = orig(x, pred, eb, radius, *a, **k)
                stats.add(eb, x, pred, codes, radius)
                return codes, outliers
            return q
        m.quantize = make(orig)
    return saved


def unhook(saved):
    for m, orig in saved:
        m.quantize = orig


def run_interp(field, stats):
    import deepsz.codec as cc
    from deepsz.predictor import InterpPredictor
    pred = InterpPredictor(order="cubic", levels=LEVELS,
                           anchor_stride=STRIDE, anchor_block=BLOCK)
    t0 = time.time()
    stream, st = cc.compress(field, EB, pred, levels=LEVELS,
                             anchor_stride=STRIDE, anchor_block=BLOCK, tune=TUNE)
    return len(stream), time.time() - t0


def run_gnn(field, stats, skel):
    from deepsz.gnn_codec import GNNCompressorCodec
    kw = dict(error_bound=EB, levels=LEVELS, anchor_stride=STRIDE,
              anchor_block=BLOCK, agg_level=AGG, chunk_size=CHUNK,
              tune=TUNE, fp16=FP16, compile=False)
    codec = None
    try:
        if skel:
            from deepsz.skel_codec import SkeletonGNNCodec
            codec = SkeletonGNNCodec(CKPT, line_order="cubic",
                                     interfaces=False, **kw)
        else:
            codec = GNNCompressorCodec(CKPT, **kw)
        t0 = time.time()
        stream = codec.compress(field)
        return len(stream), time.time() - t0
    finally:
        del codec
        try:
            import torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass


def report(tag, total, dt, stats, pts):
    npts = N ** NDIM
    pl = stats.per_level()
    print(f"\n### {tag}   {N}^4={npts:,}   {total:,} B   "
          f"bpv {8*total/npts:.4f}   ratio(f32) {npts*4/total:.2f}   {dt:.1f}s")
    print(f"  {'lvl':>3}{'stride':>7}{'pts':>13}{'pt%':>7}"
          f"{'bpp':>9}{'pred-RMSE':>12}{'rec-RMSE':>11}{'out%':>7}"
          f"{'lvl-bits':>13}{'bit%':>7}")
    lvl_bits = {lv: pl[lv]["bpp"] * pts[lv] for lv in pl}
    tb = sum(lvl_bits.values())
    for lv in range(LEVELS + 1):
        if lv not in pl:
            continue
        d = pl[lv]
        stride = STRIDE >> lv if lv else STRIDE
        bits = lvl_bits[lv]
        print(f"  {lv:>3}{stride:>7}{pts[lv]:>13,}{100*pts[lv]/npts:>6.2f}%"
              f"{d['bpp']:>9.3f}{d['pred_rmse']:>12.2e}{d['rec_rmse']:>11.2e}"
              f"{100*d['nout']/d['tot']:>6.2f}%{int(bits):>13,}{100*bits/tb:>6.2f}%")
    print(f"      ideal-entropy total {tb/8/1e3:.1f} KB ({tb/npts:.4f} bpv)  "
          f"vs real stream {total/1e3:.1f} KB")
    return pl


if __name__ == "__main__":
    prec = "fp16" if FP16 else "fp32"
    print(f"bench_levels: N={N} eb={EB} levels={LEVELS} stride={STRIDE} "
          f"chunk={CHUNK} agg={AGG} tune={TUNE} precision={prec}")
    print(f"  data={'synthetic RTI' if not DATA else DATA}  codecs={CODECS}")
    field = load_field()
    print(f"  field range [{field.min():.4f},{field.max():.4f}] std {field.std():.4f}")
    pts = analytic_pts(N)
    print("  pts/level: " + " ".join(f"L{i}:{c:,}" for i, c in enumerate(pts)))

    results = {}
    for codec in CODECS:
        st = LevelStats()
        saved = hook_all(st)
        try:
            if codec == "interp":
                tot, dt = run_interp(field, st)
            elif codec == "gnn":
                tot, dt = run_gnn(field, st, skel=False)
            elif codec == "skel":
                tot, dt = run_gnn(field, st, skel=True)
            else:
                print(f"  (unknown codec {codec}, skipping)"); continue
        finally:
            unhook(saved)
        results[codec] = report(codec.upper(), tot, dt, st, pts)

    # finest-level head-to-head (where ~90% of the bits live)
    if len(results) > 1:
        fin = LEVELS
        print(f"\n===== finest level L{fin} (stride 1) head-to-head =====")
        print(f"  {'codec':>8}{'bpp':>9}{'pred-RMSE':>12}{'rec-RMSE':>11}")
        for c in CODECS:
            if c in results and fin in results[c]:
                d = results[c][fin]
                print(f"  {c:>8}{d['bpp']:>9.3f}{d['pred_rmse']:>12.2e}{d['rec_rmse']:>11.2e}")
