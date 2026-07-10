"""Lightweight, dimension-agnostic GNN predictor for the DeepSZ closed loop.

Same interface as the other predictors (see predictor.py):
    predict(recon, known) -> pred
with recon float32 (C, *S) in original units, known bool (*S). It is a pure
function of (recon * known, known), so encoder and decoder reproduce it
bitwise. `S` may be any spatial shape of any rank; the network's weights are
shared across directions, so a model trained on 2-D images runs unchanged on
n-D grids.

Design (single hop; long-range context comes from the codec's hierarchical
stage schedule, which progressively densifies `known`):

  * for every *line* through a point (the (3^n - 1)/2 axis and diagonal
    directions) find the nearest known sample on each side;
  * store one embedding per lattice axis at every point. Per axis k
    independently, rotate each neighbour's axis-k embedding (RoPE) by a phase
    proportional to cos(line direction, axis k) — signed per side — so the
    axis-k channel reads its neighbours through their angle to axis k;
  * turn a two-sided pair into a trend/curvature message (BiDirEmbed) or a
    one-sided neighbour into an extrapolation message (DirEmbed);
  * pool the per-line messages into one context per axis with single-query
    attention, then pool axes with a second single-query attention before
    reading out a Laplacian mean/scale (PredHead), conditioned on the
    normalized error bound;
  * once a point's own value is revealed — known, but carrying the small error
    left by residual coding — embed it with InitEmbed and fuse it into the
    per-axis contexts with MixEmbed to form that point's finalized embeddings
    (the ones stored in the propagating field). In training the revealed value
    is the truth plus noise, so MixEmbed learns to trust it only up to that
    error.

Axes never mix during propagation; direction enters only as the rotary phase,
so with the frequency bank zeroed every axis channel is identical — the axial
structure is a pure inductive bias layered on the direction-blind model.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import numpy as np

from .levels import point_levels, stage_masks

# torch is imported lazily inside the class / model so that importing this
# module (e.g. for FLAG constants) stays cheap.

CKPT_VERSION = 5


def half_directions(ndim: int) -> list[tuple[int, ...]]:
    """One representative per line: all offset vectors in {-1,0,1}^ndim whose
    first non-zero component is +1 (so d and -d collapse to one line)."""
    dirs = []
    for d in itertools.product((-1, 0, 1), repeat=ndim):
        first = next((x for x in d if x != 0), 0)
        if first > 0:
            dirs.append(d)
    return dirs


def _nearest_steps(pat: np.ndarray, dvec, P: int) -> np.ndarray:
    """For each residue r in [0,P)^ndim, the smallest step t>=1 along ``dvec``
    that lands on a True cell of the periodic pattern ``pat`` (period P), or 0
    if none. The hit sequence along any lattice line is periodic in t with
    period dividing P, so the global-nearest hit is the first one within [1,P]:
    a t0(r)+kP tail never beats it. O(P^(ndim+1)) on the tiny period tile."""
    grids = np.indices(pat.shape)                       # (ndim, P, ...)
    t0 = np.zeros(pat.shape, np.int64)                  # 0 == no hit yet
    for t in range(1, P + 1):
        r = tuple((grids[k] + t * dvec[k]) % P for k in range(pat.ndim))
        take = (t0 == 0) & pat[r]
        t0[take] = t
    return t0


def _shift(arr: np.ndarray, offset, fill):
    """result[p] = arr[p + offset], out-of-bounds filled with `fill`."""
    out = np.full_like(arr, fill)
    src, dst = [], []
    for o, n in zip(offset, arr.shape):
        if o >= 0:
            src.append(slice(o, n)); dst.append(slice(0, n - o))
        else:
            src.append(slice(0, n + o)); dst.append(slice(-o, n))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _nearest_in_dir(known: np.ndarray, flat: np.ndarray, dvec, max_radius: int):
    """Nearest known sample stepping along +dvec. Returns (idx, dist, valid)."""
    found_idx = np.zeros(known.shape, np.int64)
    found_dist = np.ones(known.shape, np.float32)
    found = np.zeros(known.shape, bool)
    if not known.any():
        return found_idx, found_dist, found
    limit = min(max_radius, max(known.shape))
    for step in range(1, limit + 1):
        off = tuple(step * c for c in dvec)
        new = _shift(known, off, False) & ~found
        if new.any():
            found_idx[new] = _shift(flat, off, -1)[new]
            found_dist[new] = step
            found |= new
            if found.all():
                break
    return found_idx, found_dist, found


def _line_static(dvec, torch, device=None):
    """Unit line direction and its Euclidean log-distance correction."""
    vec = torch.as_tensor(dvec, dtype=torch.float32, device=device)
    nnz = (vec != 0).sum().to(torch.float32)
    return vec / torch.sqrt(nnz), 0.5 * torch.log2(nnz)


class _StageGeom:
    """Neighbour geometry for one stage: the fixed set of ``M`` query points and,
    per half-direction, the +/- side neighbour's flat index / step distance /
    validity as torch tensors of length M. Query points only — no full-grid
    tensors — so memory scales with the stage, not the image."""

    __slots__ = ("ip", "in_", "dp", "dn", "vp", "vn", "cos", "lognnz",
                 "query_idx", "idx_np", "M", "ndim")

    def __init__(self, pat, query_coords, shape, max_radius, torch, device):
        ndim = len(shape)
        self.ndim = ndim
        P = pat.shape[0]
        shp = np.asarray(shape)
        limit = min(max_radius, int(shp.max()))
        Q = query_coords                                # (M, ndim)
        self.M = int(len(Q))
        self.idx_np = (np.ravel_multi_index([Q[:, k] for k in range(ndim)], shape)
                       if self.M else np.zeros(0, np.int64))

        def t(a):
            x = torch.from_numpy(np.ascontiguousarray(a))
            return x.to(device) if device is not None else x

        self.query_idx = t(self.idx_np.astype(np.int64))
        res = tuple((Q[:, k] % P) for k in range(ndim)) if self.M else None
        line_data = {k: [] for k in ("ip", "in_", "dp", "dn", "vp", "vn")}
        cos, lognnz = [], []
        for d in half_directions(ndim):
            ln = {}
            for side, sd in (("p", np.asarray(d)), ("n", -np.asarray(d))):
                if not self.M:
                    ln["i" + side] = t(np.zeros(0, np.int64))
                    ln["d" + side] = t(np.zeros(0, np.float32))
                    ln["v" + side] = t(np.zeros(0, bool))
                    continue
                step = _nearest_steps(pat, sd, P)[res]          # (M,) infinite-lattice
                nb = Q + step[:, None] * sd                     # neighbour coords
                inb = np.all((nb >= 0) & (nb < shp), axis=1)
                valid = (step >= 1) & (step <= limit) & inb     # legacy: in-bounds & <=limit
                nbc = np.clip(nb, 0, shp - 1)
                flat = np.ravel_multi_index([nbc[:, k] for k in range(ndim)], shape)
                # legacy defaults where no neighbour: idx 0, dist 1.0 (finite, so
                # log2 stays defined; the pool masks these lines out via `valid`).
                ln["i" + side] = t(np.where(valid, flat, 0).astype(np.int64))
                ln["d" + side] = t(np.where(valid, step, 1).astype(np.float32))
                ln["v" + side] = t(valid)
            line_data["ip"].append(ln["ip"])
            line_data["in_"].append(ln["in"])
            line_data["dp"].append(ln["dp"])
            line_data["dn"].append(ln["dn"])
            line_data["vp"].append(ln["vp"])
            line_data["vn"].append(ln["vn"])
            c, ld = _line_static(d, torch, device)
            cos.append(c)
            lognnz.append(ld)
        for name, values in line_data.items():
            setattr(self, name, torch.stack(values, dim=0))
        self.cos = torch.stack(cos, dim=0)
        self.lognnz = torch.stack(lognnz, dim=0).unsqueeze(1)


class _LegacyGeom:
    """Mask-based geometry for the old stage_forward API. Slower than the
    schedule-aware `_StageGeom`, but keeps older trainer/eval callers working."""

    __slots__ = ("ip", "in_", "dp", "dn", "vp", "vn", "cos", "lognnz",
                 "query_idx", "idx_np", "M", "ndim")

    def __init__(self, known, max_radius, torch, device=None, query_idx=None):
        n = known.size
        self.ndim = known.ndim
        flat = np.arange(n, dtype=np.int64).reshape(known.shape)
        if query_idx is None:
            idx = np.arange(n, dtype=np.int64)
        else:
            idx = np.asarray(query_idx, np.int64).reshape(-1)
        self.idx_np = idx
        self.M = int(len(idx))

        def t(a):
            x = torch.from_numpy(np.ascontiguousarray(a))
            return x.to(device) if device is not None else x

        self.query_idx = t(idx.astype(np.int64))
        line_data = {k: [] for k in ("ip", "in_", "dp", "dn", "vp", "vn")}
        cos, lognnz = [], []
        for h in half_directions(known.ndim):
            neg = tuple(-c for c in h)
            ip, dp, vp = _nearest_in_dir(known, flat, h, max_radius)
            in_, dn, vn = _nearest_in_dir(known, flat, neg, max_radius)
            line_data["ip"].append(t(ip.reshape(-1)[idx].astype(np.int64)))
            line_data["in_"].append(t(in_.reshape(-1)[idx].astype(np.int64)))
            line_data["dp"].append(t(dp.reshape(-1)[idx].astype(np.float32)))
            line_data["dn"].append(t(dn.reshape(-1)[idx].astype(np.float32)))
            line_data["vp"].append(t(vp.reshape(-1)[idx]))
            line_data["vn"].append(t(vn.reshape(-1)[idx]))
            c, ld = _line_static(h, torch, device)
            cos.append(c)
            lognnz.append(ld)
        for name, values in line_data.items():
            setattr(self, name, torch.stack(values, dim=0))
        self.cos = torch.stack(cos, dim=0)
        self.lognnz = torch.stack(lognnz, dim=0).unsqueeze(1)


def _period_prefixes(shape, levels, stride, block):
    """Periodic `known`-before-stage pattern for every stage, on one period tile
    (P=stride). Because each schedule mask is a per-axis residue condition mod a
    divisor of the anchor stride, the real `known` mask satisfies
    ``known[idx] == pat[idx % P]``; evaluating the schedule on a P-sized grid
    yields that period tile with no boundary truncation."""
    P = stride
    tile = (P,) * len(shape)
    pats, cum = [], np.zeros(tile, bool)
    for mask in stage_masks(tile, levels, stride, block):
        pats.append(cum.copy())                         # known BEFORE this stage
        cum |= mask
    return pats


_GEOM_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def build_stage_geoms(shape, levels, stride, block, max_radius, torch, device=None):
    """Per-stage `_StageGeom` list (empty stages dropped) plus a
    ``|known|-before-stage -> list index`` map, for the whole schedule of one
    region shape. Closed-form lattice geometry, computed at the query points
    only; cached per (shape, levels, stride, block, max_radius, device) and
    shared by encoder tuning sweeps, decoder, and the trainer.

    ponytail: unbounded cache, bounded in practice (a handful of shapes/configs);
    add an LRU cap only if a caller feeds unboundedly many distinct configs."""
    key = (tuple(int(n) for n in shape), levels, stride, block, max_radius, str(device))
    hit = _GEOM_CACHE.get(key)
    if hit is not None:
        return hit
    shape = tuple(int(n) for n in shape)
    masks = stage_masks(shape, levels, stride, block)
    pats = _period_prefixes(shape, levels, stride, block)
    geoms, count_to_i, cum = [], {}, 0
    for s, mask in enumerate(masks):
        n = int(mask.sum())
        if n:  # empty stages get no predict call; skipping keeps counts unique
            Q = np.stack(np.nonzero(mask), axis=1)
            count_to_i[cum] = len(geoms)
            geoms.append(_StageGeom(pats[s], Q, shape, max_radius, torch, device))
        cum += n
    out = (geoms, count_to_i)
    _GEOM_CACHE[key] = out
    return out


def _mlp(torch, sizes):
    import torch.nn as nn
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    layers.pop()  # drop trailing activation
    return nn.Sequential(*layers)


def build_model(d: int = 32):
    """Construct the axial, dimension-agnostic GNN."""
    import torch
    import torch.nn as nn

    assert d % 2 == 0, "d must be even for rotary axis embeddings"

    # Hidden width of the message/fusion/readout MLPs. Decoupled from d so the
    # per-axis field (memory ~ ndim * d) stays cheap while these functions keep
    # capacity — the wide activations are transient, over one stage's points.
    h = 2 * d

    class InitEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [1, d, d])

        def forward(self, v):  # v: (..., 1) normalized value
            return self.net(v)

    class Rope(nn.Module):
        """Rotate a neighbour's per-axis embedding by a phase proportional to
        the signed cosine between the line and each lattice axis. RoPE, but the
        "position" is the direction cosine in [-1, 1] rather than an unbounded
        token index — so the frequencies are spread linearly over [~0, pi]
        (a full turn across the cosine range) instead of the usual geometric
        decay, which for a bounded position leaves almost every channel
        unrotated. Low pairs stay near pass-through (content), high pairs are
        strongly direction-sensitive."""

        def __init__(self):
            super().__init__()
            freq = torch.linspace(math.pi / (d // 2), math.pi, d // 2)
            self.register_buffer("freq", freq)

        def forward(self, e, cos, sign):
            # e: (B, L, M, K, d); cos: (L, K)
            # theta broadcasts over B (dim 0) and M (dim 2).
            theta = (sign * cos)[:, None, :, None] * self.freq  # (L, 1, K, d/2)
            cs, sn = torch.cos(theta), torch.sin(theta)
            B, L, M, K, _ = e.shape
            pairs = e.reshape(B, L, M, K, d // 2, 2)
            e1, e2 = pairs[..., 0], pairs[..., 1]        # (B, L, M, K, d/2)
            out = torch.stack((e1 * cs - e2 * sn,
                               e1 * sn + e2 * cs), dim=-1)
            return out.reshape(B, L, M, K, d)

    class DirEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [d + 2, h, d])

        def forward(self, e, sign, logd):  # one neighbour + (sign, log2 dist)
            return self.net(torch.cat([e, sign, logd], dim=-1))

    class BiDirEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [2 * d + 2, h, d])

        def forward(self, e_neg, e_pos, logd_neg, logd_pos):
            return self.net(torch.cat([e_neg, e_pos, logd_neg, logd_pos], dim=-1))

    class AttnPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.wk = nn.Linear(d, d)
            self.wv = nn.Linear(d, d)
            self.q = nn.Parameter(torch.randn(d) * d ** -0.5)
            self.null_k = nn.Parameter(torch.randn(d) * d ** -0.5)
            self.null_v = nn.Parameter(torch.zeros(d))

        def forward(self, msgs, valid):
            # msgs: (L, B, N, d); valid: (L, N) bool
            k = self.wk(msgs)
            v = self.wv(msgs)
            scale = self.q.shape[0] ** -0.5
            scores = (k * self.q).sum(-1) * scale  # (L, B, N)
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            L, B, N, dd = msgs.shape
            sn = (self.null_k * self.q).sum() * scale
            scores = torch.cat([scores, sn.expand(1, B, N)], dim=0)
            v = torch.cat([v, self.null_v.expand(1, B, N, dd)], dim=0)
            w = torch.softmax(scores, dim=0)  # (L+1, B, N)
            return (w.unsqueeze(-1) * v).sum(0)  # (B, N, d)

    class PredHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [d + 1, h, 2])

        def forward(self, e, eb):
            B, M, _ = e.shape
            eb = torch.as_tensor(eb, dtype=e.dtype, device=e.device).reshape(-1)
            if eb.numel() == 1:
                eb = eb.expand(B)
            elif eb.numel() != B:
                raise ValueError(f"eb has {eb.numel()} entries for batch {B}")
            log_eb = torch.log2(eb.clamp_min(torch.finfo(e.dtype).tiny))
            cond = log_eb.view(B, 1, 1).expand(B, M, 1)
            out = self.net(torch.cat([e, cond], dim=-1))
            mu = torch.sigmoid(out[..., 0])
            log_b = out[..., 1].clamp(-8.0, 0.0)
            return mu, log_b

    class MixEmbed(nn.Module):
        """Fuse a point's pooled neighbour context with the embedding of its
        own now-known value into the finalized embedding stored in the field.
        The value carries the small residual-coding error (noise in training),
        so this lets the field remember what was actually reconstructed there
        rather than the raw prediction."""

        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [2 * d, h, d])

        def forward(self, ctx, value_emb):  # (B, N, d), (B, N, d) -> (B, N, d)
            return self.net(torch.cat([ctx, value_emb], dim=-1))

    class CoarseProj(nn.Module):
        """Project a chunk's per-level mean finalized embedding into the
        context space MixEmbed expects for halo neighbours, conditioned on the
        level's stride so one MLP serves every level. Used only by the chunked
        codec path: out-of-chunk neighbours are represented as
        ``mix(coarse[chunk, level], InitEmbed(value))`` instead of their dense
        finalized embedding (which is never stored)."""

        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [d + 1, h, d])

        def forward(self, mean_emb, log_s):  # (..., K, d), scalar
            cond = mean_emb.new_full((*mean_emb.shape[:-1], 1), float(log_s))
            return self.net(torch.cat([mean_emb, cond], dim=-1))

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d
            self.init = InitEmbed()
            self.rope = Rope()
            self.dir = DirEmbed()
            self.bidir = BiDirEmbed()
            self.line_pool = AttnPool()
            self.axis_pool = AttnPool()
            self.head = PredHead()
            self.mix = MixEmbed()
            self.coarse = CoarseProj()

        def _line_messages(self, E, geom):
            """Per-line messages for this stage's M query points, built from
            neighbour *embeddings* E (not raw values) so trends/periodicity
            propagate hop by hop. Returns msgs (L, B, M, K, d) and valid (L, M)
            — the axis dim K is carried through Dir/BiDir as a batch dim, each
            axis reading its neighbours through the rotary phase. geom already
            holds the +/- neighbour of each query point, so this touches O(M)
            rows of the field, never the whole grid."""
            B = E.shape[0]
            # Batch all directions into the MLPs at once. This turns the old
            # per-line Dir/BiDir calls into three larger matmuls per embed pass,
            # which is much friendlier to GPU inference.
            ep = E[:, geom.ip]                      # (B, L, M, K, d)
            en = E[:, geom.in_]
            ep = self.rope(ep, geom.cos, 1.0)       # (B, L, M, K, d)
            en = self.rope(en, geom.cos, -1.0)
            _, L, M, K, _ = ep.shape
            lp = (torch.log2(geom.dp) + geom.lognnz
                  ).view(1, L, M, 1, 1).expand(B, L, M, K, 1)
            lnn = (torch.log2(geom.dn) + geom.lognnz
                   ).view(1, L, M, 1, 1).expand(B, L, M, K, 1)
            sign = ep.new_ones(B, L, M, K, 1)

            bidir = self.bidir(en, ep, lnn, lp)
            dpos = self.dir(ep, sign, lp)
            dneg = self.dir(en, -sign, lnn)
            both = (geom.vp & geom.vn).view(1, L, M, 1, 1)
            vp_only = (geom.vp & ~geom.vn).view(1, L, M, 1, 1)
            msg = torch.where(both, bidir, torch.where(vp_only, dpos, dneg))
            return (msg.permute(1, 0, 2, 3, 4).contiguous(),
                    geom.vp | geom.vn)

        def embed(self, E, geom):
            """Per-axis contexts at geom's query points: single-query attention
            over the per-line neighbour messages (no self value), pooled
            independently per axis. For an anchor with no known neighbours every
            line is masked and each axis falls back to the learned null token."""
            msgs, valid = self._line_messages(E, geom)  # (L, B, M, K, d), (L, M)
            L, B, M, K, _ = msgs.shape
            flat = msgs.reshape(L, B, M * K, self.d)
            vflat = valid.repeat_interleave(K, dim=1)   # (L, M*K)
            ctx = self.line_pool(flat, vflat)           # (B, M*K, d)
            return ctx.reshape(B, M, K, self.d)         # (B, M, ndim, d)

        def finalize(self, ctx, self_val):
            """Finalized embedding for points whose value has just been
            revealed: embed the (noisy) known value with InitEmbed and fuse it
            into every axis context via MixEmbed. `self_val` is the
            reconstructed value — truth + noise in training, the quantised
            recon at inference — so the mix learns to trust it up to eb."""
            if ctx.dim() != 4:
                raise ValueError(
                    f"finalize requires axial context (B, M, ndim, d), got "
                    f"shape {tuple(ctx.shape)}")
            value_emb = self.init(self_val.unsqueeze(-1)).unsqueeze(2)
            return self.mix(ctx, value_emb.expand_as(ctx))

        def head_of(self, ctx, eb):
            K, M = ctx.shape[2], ctx.shape[1]
            msgs = ctx.permute(2, 0, 1, 3)
            valid = torch.ones(K, M, dtype=torch.bool, device=ctx.device)
            return self.head(self.axis_pool(msgs, valid), eb)

    return GNN()


def _load_inference_model(checkpoint_path, torch, device):
    """Load immutable inference weights once per checkpoint revision/device."""
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, str(device))
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    version = int(ckpt.get("version", 1))
    if version != CKPT_VERSION:
        raise ValueError(
            "Rotary axial GNN checkpoint format v5 is required. Retrain with "
            "scripts/train_gnn.py."
        )
    d = int(ckpt["d"])
    model = build_model(d).eval()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    checkpoint_hash = hashlib.sha256(path.read_bytes()).digest()[:16]
    out = (d, model, checkpoint_hash)
    _MODEL_CACHE[key] = out
    return out


def _stage_forward_geoms(model, E, geom_prev, geom_head, finalize_vals, torch,
                         finalize_ctx=None, eb=0.01):
    """Finalize with and return pre-axis-pool contexts for stage reuse."""
    if geom_prev is not None and geom_prev.M:
        ctx = finalize_ctx if finalize_ctx is not None else model.embed(E, geom_prev)
        finalized = model.finalize(ctx, finalize_vals)
        E = E.index_copy(1, geom_prev.query_idx, finalized)     # write newly-known
    head_ctx = model.embed(E, geom_head)
    return model.head_of(head_ctx, eb), E, head_ctx


def stage_forward(model, E, *args, **kwargs):
    """One codec stage of the propagating GNN.

    Supports both APIs:
    - optimized geometry API:
      ``stage_forward(model, E, geom_prev, geom_head, finalize_vals, torch,
      finalize_ctx=None, eb=...) -> ((mu, log_b), E, head_ctx)``
    - legacy mask API:
      ``stage_forward(model, E, prev_mask, known_mask, norm, max_radius, torch,
      predict_idx=None) -> (values, E)``
    """
    if len(args) < 4:
        raise TypeError("stage_forward needs either geometry or mask arguments")

    # New path: geometry objects have `M` and `query_idx`; the codec/GNNPredictor
    # uses this faster schedule-aware form.
    if args[0] is None or hasattr(args[0], "M"):
        geom_prev, geom_head, finalize_vals, torch = args[:4]
        finalize_ctx = kwargs.pop("finalize_ctx", None)
        eb = kwargs.pop("eb", 0.01)
        if kwargs:
            raise TypeError(f"unexpected keyword argument {next(iter(kwargs))!r}")
        return _stage_forward_geoms(model, E, geom_prev, geom_head, finalize_vals,
                                    torch, finalize_ctx=finalize_ctx, eb=eb)

    # Legacy path used by older training/eval code.
    if len(args) < 5:
        raise TypeError("legacy stage_forward needs max_radius and torch")
    prev_mask, known_mask, norm, max_radius, torch = args[:5]
    predict_idx = kwargs.pop("predict_idx", None)
    eb = kwargs.pop("eb", 0.01)
    if kwargs:
        raise TypeError(f"unexpected keyword argument {next(iter(kwargs))!r}")
    device = E.device
    newly = known_mask & ~prev_mask
    if newly.any():
        idx_np = np.nonzero(newly.reshape(-1))[0]
        geom_prev = _LegacyGeom(prev_mask, max_radius, torch, device, idx_np)
        ctx = model.embed(E, geom_prev)
        finalized = model.finalize(ctx, norm[:, geom_prev.query_idx])
        E = E.index_copy(1, geom_prev.query_idx, finalized)
    geom_head = _LegacyGeom(known_mask, max_radius, torch, device, predict_idx)
    values = model.head_of(model.embed(E, geom_head), eb)
    return values, E


class GNNPredictor:
    """GNN predictor loaded from a trained checkpoint. The stage schedule
    (`levels`, `anchor_stride`, `anchor_block`) must match the codec's, so the
    precomputed neighbour geometry lines up with the masks the codec feeds in;
    `max_radius` caps the neighbour distance (anchors always sit closer)."""

    from .bitstream import FLAG_GNN as _FLAG
    stream_flag = _FLAG
    tile_free = True  # runs on the whole tensor as one region, no tiling
    tunable = True    # encoder sweeps eb_ratio (no centre mode; see codec.encode)
    provides_scale = True

    def __init__(self, checkpoint_path, vmin: float, vmax: float,
                 tile_size: int = 64, max_radius: int = 64, device: str = "cpu",
                 levels: int = 4, anchor_stride: int = 16, anchor_block: int = 1):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.tile_size = int(tile_size)
        self.max_radius = int(max_radius)
        self.levels = int(levels)
        self.anchor_stride = int(anchor_stride)
        self.anchor_block = int(anchor_block)

        self.d, self.model, self.checkpoint_hash = _load_inference_model(
            checkpoint_path, torch, self.device)
        self._sched: dict = {}   # shape -> (stage geoms, count->index map)
        self._reset()

    def _reset(self):
        self._E = None           # persistent embedding field (C, N, ndim, d)
        self._ctx = None         # last stage's head context (next finalize reuses it)
        self._stage = None       # list index of the last predicted stage

    def _schedule(self, shape):
        key = tuple(int(n) for n in shape)
        g = self._sched.get(key)
        if g is None:
            g = build_stage_geoms(key, self.levels, self.anchor_stride,
                                  self.anchor_block, self.max_radius,
                                  self._torch, self.device)
            self._sched[key] = g
        return g

    def predict(self, recon: np.ndarray, known: np.ndarray,
               pos: np.ndarray | None = None, eb: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Predict the current stage's holes (`pos`, the codec's stage mask).
        Everything scales with the stage: geometry is precomputed at the query
        points only, values are normalized only at the just-revealed points, and
        the finalize context is inherited from the previous stage's head (same
        field, same geometry) instead of being pooled twice. Returns
        ``(pred, scale)`` in original data units, where ``scale`` is the
        Laplacian ``b`` parameter predicted by the second head."""
        torch = self._torch
        if pos is None:
            raise ValueError("GNNPredictor.predict requires the stage mask `pos`")
        if eb is None:
            eb = getattr(self, "eb", None)
        if eb is None:
            raise ValueError("GNNPredictor.predict requires `eb`")
        c = recon.shape[0]
        span = self.vmax - self.vmin
        norm_eb = float(eb) / span
        geoms, count_to_i = self._schedule(recon.shape[1:])
        i = count_to_i.get(int(known.sum()))
        if not i:  # None (unknown count) or 0 (anchors are coded directly)
            raise ValueError("known mask does not match the GNN stage schedule")

        cont = self._E is not None and self._stage == i - 1
        if not cont:
            ndim = recon.ndim - 1
            self._E = torch.zeros(
                c, known.size, ndim, self.d, device=self.device)
            self._ctx = None
        geom_prev, geom_head = geoms[i - 1], geoms[i]

        # normalized values at the just-revealed points (finalize's input), read
        # compactly from recon — no full (C, N) clip/scatter each stage.
        vals = recon.reshape(c, -1)[:, geom_prev.idx_np]
        fvals = torch.from_numpy(
            ((np.clip(vals, self.vmin, self.vmax) - self.vmin) / span
             ).astype(np.float32)).to(self.device)
        finalize_ctx = self._ctx if cont else None
        with torch.no_grad():
            (values, log_b), self._E, self._ctx = stage_forward(
                self.model, self._E, geom_prev, geom_head, fvals, torch,
                finalize_ctx=finalize_ctx, eb=norm_eb)
        self._stage = i
        pred = values.cpu().numpy().reshape(c, -1) * span + self.vmin
        scale = np.exp2(log_b.cpu().numpy().reshape(c, -1)) * span
        return (np.clip(pred, self.vmin, self.vmax).astype(np.float32),
                scale.astype(np.float32))


# ---------------------------------------------------------------------------
# Chunked inference: the tensor is coded chunk by chunk (global anchors first),
# dense embeddings exist only for the current chunk + halo, and finished chunks
# leave behind one CoarseProj'd mean embedding per level (see ChunkedGNNPredictor).
# ---------------------------------------------------------------------------


class _ChunkGeoms:
    """Stage geometry and index metadata for one chunk shape, in the
    halo-padded local frame (halo = ``anchor_stride`` on every side — every
    valid periodic neighbour is within ``anchor_stride`` steps, see
    `_nearest_steps`). Origin-independent: chunk origins are multiples of the
    stride and the halo equals it, so local coordinates are congruent to global
    ones mod the pattern period and every aligned chunk of the same shape
    shares this object (cached in ``build_chunk_geoms``)."""

    def __init__(self, chunk_shape, levels, stride, block, torch, device):
        self.chunk_shape = tuple(int(n) for n in chunk_shape)
        self.levels, self.stride, self.block = levels, stride, block
        self.halo = stride
        ndim = len(self.chunk_shape)
        self.ndim = ndim
        self.padded_shape = tuple(n + 2 * stride for n in self.chunk_shape)
        self.n_padded = int(np.prod(self.padded_shape))

        masks = stage_masks(self.chunk_shape, levels, stride, block)
        pats = _period_prefixes(self.chunk_shape, levels, stride, block)
        self.geoms, self.coords = [], []   # per stage; None for empty stages
        for s, mask in enumerate(masks):
            Q = np.stack(np.nonzero(mask), axis=1)   # chunk-frame coords
            if not len(Q):
                self.geoms.append(None)
                self.coords.append(None)
                continue
            self.geoms.append(_StageGeom(pats[s], Q + stride, self.padded_shape,
                                         stride, torch, device))
            self.coords.append(Q)
        # prediction chain: stage 0 is always the base (anchors, possibly empty
        # in a ragged tail chunk -> None geom, nothing to finalize), followed by
        # every non-empty refinement stage in order.
        self.chain = [0] + [s for s in range(1, len(self.geoms))
                            if self.geoms[s] is not None]

        pad_flat = np.arange(self.n_padded, dtype=np.int64
                             ).reshape(self.padded_shape)
        inner = tuple(slice(stride, stride + n) for n in self.chunk_shape)
        hmask = np.ones(self.padded_shape, bool)
        hmask[inner] = False
        self.halo_flat = pad_flat[hmask]                             # (H,)
        # chunk-frame coords of halo cells (negative / >= chunk_shape allowed)
        self.halo_coords = np.stack(np.nonzero(hmask), axis=1) - stride
        self.interior_flat = pad_flat[inner].reshape(-1)
        # per-level buckets of interior padded indices, for the coarse means
        lv = point_levels(list(np.indices(self.chunk_shape).reshape(ndim, -1)),
                          levels, stride, block)
        self.level_flat = [self.interior_flat[lv == l]
                           for l in range(levels + 1)]


_CHUNK_GEOM_CACHE: dict = {}


def build_chunk_geoms(chunk_shape, levels, stride, block, torch, device=None):
    """Cached `_ChunkGeoms` per (chunk shape, schedule, device). Interior
    chunks all share one entry; ragged edge chunks add at most a few shape
    variants (ponytail: unbounded like _GEOM_CACHE, bounded in practice)."""
    key = (tuple(int(n) for n in chunk_shape), levels, stride, block, str(device))
    hit = _CHUNK_GEOM_CACHE.get(key)
    if hit is None:
        hit = _ChunkGeoms(chunk_shape, levels, stride, block, torch, device)
        _CHUNK_GEOM_CACHE[key] = hit
    return hit


class _OverlaidGeom:
    """A cached padded-frame `_StageGeom` with its periodic validity ANDed with
    one chunk's runtime usability mask: in-chunk neighbours are always usable
    (the local schedule reveals them in pattern order), halo neighbours only if
    their point is actually decoded (already-coded chunk, or a global anchor)
    and inside the tensor. Shares every other tensor with the base geometry."""

    __slots__ = _StageGeom.__slots__

    def __init__(self, base, usable):
        for name in ("ip", "in_", "dp", "dn", "cos", "lognnz",
                     "query_idx", "idx_np", "M", "ndim"):
            setattr(self, name, getattr(base, name))
        self.vp = base.vp & usable[base.ip]
        self.vn = base.vn & usable[base.in_]


def chunk_halo_info(cg, origin, shape, edges, grid, coded):
    """Usability mask and halo-fill indices for one chunk of a chunk grid.

    Returns ``(usable, sel_flat, chunk_ids, lv, gflat)``: a bool mask over the
    padded frame (interior always usable, halo cells only when decoded), and —
    for the usable halo cells — their padded flat index, owning chunk id,
    dyadic level and global flat index. Shared by the inference predictor and
    the trainer so both sides build halo context identically."""
    ndim = len(shape)
    usable = np.zeros(cg.n_padded, bool)
    usable[cg.interior_flat] = True
    gc = cg.halo_coords + np.asarray(origin, np.int64)
    shp = np.asarray(shape)
    inb = np.all((gc >= 0) & (gc < shp), axis=1)
    gci = gc[inb]
    chunk_ids = np.ravel_multi_index(
        [gci[:, k] // edges[k] for k in range(ndim)], grid)
    lv = point_levels([gci[:, k] for k in range(ndim)],
                      cg.levels, cg.stride, cg.block)
    ok = np.asarray(coded)[chunk_ids] | (lv == 0)
    sel_flat = cg.halo_flat[inb][ok]
    usable[sel_flat] = True
    gflat = np.ravel_multi_index(
        [gci[ok][:, k] for k in range(ndim)], shape)
    return usable, sel_flat, chunk_ids[ok], lv[ok], gflat


def anchor_finalize(model, vals, ndim):
    """Finalized embedding of anchor points as the codec computes it: anchors
    have nothing known before them, so their pooled context is the line pool's
    null token and the finalized embedding is a pure function of the value.
    ``vals``: (B, M) normalized values -> (B, M, ndim, d)."""
    B, M = vals.shape
    null = model.line_pool.null_v.view(1, 1, 1, -1).expand(B, M, ndim, -1)
    return model.finalize(null, vals)


def halo_embed(model, coarse_vecs, vals):
    """Representation of an out-of-chunk known neighbour: fuse its chunk's
    per-level coarse embedding with the embedding of its reconstructed value.
    ``coarse_vecs``: (B, H, K, d); ``vals``: (B, H) normalized."""
    return model.finalize(coarse_vecs, vals)


def chunk_coarse(model, E_pad, cg, torch):
    """Per-level coarse embeddings of a finished chunk: mean of the finalized
    interior embeddings per level -> CoarseProj (conditioned on the level
    stride). Levels with no points in this (ragged) chunk stay zero — they are
    never read, since a halo point of that level would itself be such an
    interior point. Returns (B, levels + 1, ndim, d)."""
    B = E_pad.shape[0]
    out = E_pad.new_zeros(B, cg.levels + 1, cg.ndim, model.d)
    for l, flat in enumerate(cg.level_flat):
        if not len(flat):
            continue
        idx = torch.from_numpy(flat).to(E_pad.device)
        mean = E_pad.index_select(1, idx).mean(dim=1)        # (B, K, d)
        s = cg.stride if l == 0 else max(cg.stride >> l, 1)
        out[:, l] = model.coarse(mean, math.log2(s))
    return out


class ChunkedGNNPredictor:
    """Chunk-by-chunk GNN predictor with bounded memory.

    Coding order (mirrored bitwise by the decoder): a global anchor pass, then
    chunks in raster order, each running its local stage schedule with a dense
    embedding field over chunk + ``anchor_stride`` halo only. What survives a
    chunk is its per-level coarse embedding table entry (CoarseProj of the mean
    finalized embedding), used to represent its points when they appear in a
    later chunk's halo. Everything model-sized is O(chunk); the only O(N)
    state is the caller's recon array.

    Per-tensor protocol driven by the codec (encode and decode identically):
        begin(shape, chunk_edges, channels)
        anchor_coarse(recon)                     # after the anchor pass
        for ci in range(n_chunks):               # raster order
            start_chunk(ci, recon)
            for each non-empty local stage s >= 1, in order:
                pred, scale = predict_stage(s, recon, eb)
                ... caller quantizes and writes recon ...
            finish_chunk(ci, recon)
    """

    provides_scale = True

    def __init__(self, checkpoint_path, vmin: float, vmax: float,
                 device: str = "cpu", levels: int = 4, anchor_stride: int = 16,
                 anchor_block: int = 1):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        span = self.vmax - self.vmin
        self.span = span if span > 0 else 1.0
        self.levels = int(levels)
        self.anchor_stride = int(anchor_stride)
        self.anchor_block = int(anchor_block)
        self.d, self.model, self.checkpoint_hash = _load_inference_model(
            checkpoint_path, torch, self.device)

    # -- per-tensor lifecycle -------------------------------------------------
    def begin(self, shape, chunk_edges, channels: int = 1):
        torch = self._torch
        self.shape = tuple(int(n) for n in shape)
        self.edges = tuple(int(e) for e in chunk_edges)
        if len(self.edges) != len(self.shape):
            raise ValueError("chunk_edges must have one entry per axis")
        for e in self.edges:
            if e < self.anchor_stride or e % self.anchor_stride:
                raise ValueError("chunk edges must be positive multiples of "
                                 "anchor_stride")
        self.grid = tuple(-(-n // e) for n, e in zip(self.shape, self.edges))
        self.n_chunks = int(np.prod(self.grid))
        self.C = int(channels)
        ndim = len(self.shape)
        self.coarse = torch.zeros(self.C, self.n_chunks, self.levels + 1,
                                  ndim, self.d, device=self.device)
        self.coded = np.zeros(self.n_chunks, bool)
        self._cg = None

    def chunk_slices(self, ci: int):
        cidx = np.unravel_index(ci, self.grid)
        return tuple(slice(i * e, min((i + 1) * e, n))
                     for i, e, n in zip(cidx, self.edges, self.shape))

    def _norm(self, vals: np.ndarray):
        v = (np.clip(vals, self.vmin, self.vmax) - self.vmin) / self.span
        return self._torch.from_numpy(v.astype(np.float32)).to(self.device)

    def anchor_coarse(self, recon: np.ndarray):
        """Level-0 coarse embeddings for every chunk, right after the global
        anchor pass — so every chunk has anchor context on all sides before any
        chunk is coded. Anchors finalize from the null context, so this is a
        pure function of their reconstructed values."""
        torch = self._torch
        ndim = len(self.shape)
        log_s = math.log2(self.anchor_stride)
        with torch.no_grad():
            for ci in range(self.n_chunks):
                axes = []
                for sl in self.chunk_slices(ci):
                    c = np.arange(sl.start, sl.stop)
                    axes.append(c[(c % self.anchor_stride) < self.anchor_block])
                if any(len(a) == 0 for a in axes):
                    continue                     # ragged chunk with no anchors
                vals = recon[(slice(None), *np.ix_(*axes))].reshape(self.C, -1)
                fin = anchor_finalize(self.model, self._norm(vals), ndim)
                self.coarse[:, ci, 0] = self.model.coarse(fin.mean(1), log_s)

    def start_chunk(self, ci: int, recon: np.ndarray):
        torch = self._torch
        sls = self.chunk_slices(ci)
        origin = np.array([sl.start for sl in sls], np.int64)
        cshape = tuple(sl.stop - sl.start for sl in sls)
        cg = build_chunk_geoms(cshape, self.levels, self.anchor_stride,
                               self.anchor_block, torch, self.device)

        # usability over the padded frame: interior always, halo per rule
        usable_np, sel_flat, chunk_ids, lv, gflat = chunk_halo_info(
            cg, origin, self.shape, self.edges, self.grid, self.coded)
        usable = torch.from_numpy(usable_np).to(self.device)

        # dense local field; fill usable halo cells from coarse + recon value
        E = torch.zeros(self.C, cg.n_padded, cg.ndim, self.d,
                        device=self.device)
        if len(sel_flat):
            vals = self._norm(recon.reshape(self.C, -1)[:, gflat])
            ids = torch.from_numpy(chunk_ids).to(self.device)
            lvs = torch.from_numpy(lv.astype(np.int64)).to(self.device)
            cvec = self.coarse[:, ids, lvs]                # (C, Hs, K, d)
            with torch.no_grad():
                E[:, torch.from_numpy(sel_flat).to(self.device)] = \
                    halo_embed(self.model, cvec, vals)

        self._cg = cg
        self._ci = ci
        self._E = E
        self._geoms = [None if g is None else _OverlaidGeom(g, usable)
                       for g in cg.geoms]
        # global flat index per stage (finalize values are read from recon)
        self._gidx = [None if c is None else np.ravel_multi_index(
            [(c[:, k] + origin[k]) for k in range(len(self.shape))], self.shape)
            for c in cg.coords]
        self._ctx = None
        self._pos = 0                                      # index into cg.chain

    def predict_stage(self, s: int, recon: np.ndarray, eb: float):
        """Predict local stage ``s`` (must be the next non-empty stage in the
        chunk's schedule). Returns (pred, scale) in original units, ordered
        like ``np.nonzero`` of the local stage mask."""
        torch = self._torch
        cg = self._cg
        j = self._pos + 1
        if j >= len(cg.chain) or cg.chain[j] != s:
            raise ValueError(f"stage {s} out of order for this chunk")
        prev = cg.chain[j - 1]
        gp, gh = self._geoms[prev], self._geoms[s]
        fvals = None if gp is None else \
            self._norm(recon.reshape(self.C, -1)[:, self._gidx[prev]])
        with torch.no_grad():
            (values, log_b), self._E, self._ctx = stage_forward(
                self.model, self._E, gp, gh, fvals, torch,
                finalize_ctx=self._ctx, eb=float(eb) / self.span)
        self._pos = j
        pred = values.cpu().numpy().reshape(self.C, -1) * self.span + self.vmin
        scale = np.exp2(log_b.cpu().numpy().reshape(self.C, -1)) * self.span
        return (np.clip(pred, self.vmin, self.vmax).astype(np.float32),
                scale.astype(np.float32))

    def finish_chunk(self, ci: int, recon: np.ndarray):
        """Finalize the last stage's points into the field, store the chunk's
        per-level coarse embeddings, and drop the dense field."""
        torch = self._torch
        cg = self._cg
        if ci != self._ci:
            raise ValueError("finish_chunk out of order")
        if self._pos != len(cg.chain) - 1:
            raise ValueError("finish_chunk before all non-empty stages were "
                             "predicted")
        last = cg.chain[self._pos]
        g = self._geoms[last]
        E = self._E
        with torch.no_grad():
            if g is not None:
                fvals = self._norm(recon.reshape(self.C, -1)[:,
                                                             self._gidx[last]])
                ctx = self._ctx if self._ctx is not None else \
                    self.model.embed(E, g)
                E = E.index_copy(1, g.query_idx,
                                 self.model.finalize(ctx, fvals))
            self.coarse[:, ci] = chunk_coarse(self.model, E, cg, torch)
        self.coded[ci] = True
        self._E = self._ctx = self._cg = None
