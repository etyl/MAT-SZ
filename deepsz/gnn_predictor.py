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
  * embed each neighbour value (InitEmbed), turn a two-sided pair into a
    trend/curvature message (BiDirEmbed) or a one-sided neighbour into an
    extrapolation message (DirEmbed);
  * pool the per-line messages with a single-query attention layer (AttnPool)
    into a *context* embedding, then read out the value (PredHead);
  * once a point's own value is revealed — known, but carrying the small error
    left by residual coding — embed it with InitEmbed and fuse it into the
    pooled context with MixEmbed to form that point's finalized embedding (the
    one stored in the propagating field). In training the revealed value is the
    truth plus noise, so MixEmbed learns to trust it only up to that error.

The six modules are exposed as separate nn.Modules as requested.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import numpy as np

from .levels import stage_masks

# torch is imported lazily inside the class / model so that importing this
# module (e.g. for FLAG constants) stays cheap.


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


class _StageGeom:
    """Neighbour geometry for one stage: the fixed set of ``M`` query points and,
    per half-direction, the +/- side neighbour's flat index / step distance /
    validity as torch tensors of length M. Query points only — no full-grid
    tensors — so memory scales with the stage, not the image."""

    __slots__ = ("lines", "query_idx", "idx_np", "M")

    def __init__(self, pat, query_coords, shape, max_radius, torch, device):
        ndim = len(shape)
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
        self.lines = []
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
            self.lines.append({"ip": ln["ip"], "in": ln["in"], "dp": ln["dp"],
                               "dn": ln["dn"], "vp": ln["vp"], "vn": ln["vn"]})


class _LegacyGeom:
    """Mask-based geometry for the old stage_forward API. Slower than the
    schedule-aware `_StageGeom`, but keeps older trainer/eval callers working."""

    __slots__ = ("lines", "query_idx", "idx_np", "M")

    def __init__(self, known, max_radius, torch, device=None, query_idx=None):
        n = known.size
        flat = np.arange(n, dtype=np.int64).reshape(known.shape)
        if query_idx is None:
            idx = np.arange(n, dtype=np.int64)
        else:
            idx = np.asarray(query_idx, np.int64).reshape(-1)
        self.idx_np = idx
        self.M = int(len(idx))

        def t(a):
            x = torch.from_numpy(np.ascontiguousarray(a.reshape(-1)))
            return x.to(device) if device is not None else x

        self.query_idx = t(idx.astype(np.int64))
        self.lines = []
        for h in half_directions(known.ndim):
            neg = tuple(-c for c in h)
            ip, dp, vp = _nearest_in_dir(known, flat, h, max_radius)
            in_, dn, vn = _nearest_in_dir(known, flat, neg, max_radius)
            self.lines.append({
                "ip": t(ip.reshape(-1)[idx].astype(np.int64)),
                "in": t(in_.reshape(-1)[idx].astype(np.int64)),
                "dp": t(dp.reshape(-1)[idx].astype(np.float32)),
                "dn": t(dn.reshape(-1)[idx].astype(np.float32)),
                "vp": t(vp.reshape(-1)[idx]),
                "vn": t(vn.reshape(-1)[idx]),
            })


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
    """Construct the GNN (its five sub-modules held as attributes)."""
    import torch
    import torch.nn as nn

    class InitEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [1, d, d])

        def forward(self, v):  # v: (..., 1) normalized value
            return self.net(v)

    class DirEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [d + 2, d, d])

        def forward(self, e, sign, logd):  # one neighbour + (sign, log2 dist)
            return self.net(torch.cat([e, sign, logd], dim=-1))

    class BiDirEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [2 * d + 2, d, d])

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
            self.net = _mlp(torch, [d, d, 1])

        def forward(self, e):
            return torch.sigmoid(self.net(e)).squeeze(-1)  # (..,) in [0,1]

    class MixEmbed(nn.Module):
        """Fuse a point's pooled neighbour context with the embedding of its
        own now-known value into the finalized embedding stored in the field.
        The value carries the small residual-coding error (noise in training),
        so this lets the field remember what was actually reconstructed there
        rather than the raw prediction."""

        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [2 * d, d, d])

        def forward(self, ctx, value_emb):  # (B, N, d), (B, N, d) -> (B, N, d)
            return self.net(torch.cat([ctx, value_emb], dim=-1))

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d
            self.init = InitEmbed()
            self.dir = DirEmbed()
            self.bidir = BiDirEmbed()
            self.attn = AttnPool()
            self.head = PredHead()
            self.mix = MixEmbed()

        def _line_messages(self, E, geom):
            """Per-line messages for this stage's M query points, built from
            neighbour *embeddings* E (not raw values) so trends/periodicity
            propagate hop by hop. Returns msgs (L, B, M, d) and valid (L, M).
            geom already holds the +/- neighbour of each query point (the codec
            only ever consumes one stage's points), so this touches O(M) rows of
            the field, never the whole grid."""
            B = E.shape[0]
            M = geom.M
            one = torch.ones(B, M, 1, device=E.device)
            msgs, valids = [], []
            for ln in geom.lines:
                ip, in_, dp, dn = ln["ip"], ln["in"], ln["dp"], ln["dn"]
                vp, vn = ln["vp"], ln["vn"]
                ep = E[:, ip]                      # (B, M, d) +side neighbour
                en = E[:, in_]                     # -side neighbour
                lp = torch.log2(dp).view(1, M, 1).expand(B, M, 1)
                lnn = torch.log2(dn).view(1, M, 1).expand(B, M, 1)
                bidir = self.bidir(en, ep, lnn, lp)
                dpos = self.dir(ep, one, lp)
                dneg = self.dir(en, -one, lnn)
                both = (vp & vn).view(1, M, 1)
                vp_only = (vp & ~vn).view(1, M, 1)
                msg = torch.where(both, bidir, torch.where(vp_only, dpos, dneg))
                msgs.append(msg)
                valids.append(vp | vn)             # (M,)
            return torch.stack(msgs, 0), torch.stack(valids, 0)

        def embed(self, E, geom):
            """Pooled *context* embedding at geom's query points: single-query
            attention over the per-line neighbour messages (no self value). For
            an anchor with no known neighbours every line message is masked out
            and the pool falls back to the learned null token."""
            msgs, valid = self._line_messages(E, geom)
            return self.attn(msgs, valid)                        # (B, M, d)

        def finalize(self, ctx, self_val):
            """Finalized embedding for points whose value has just been
            revealed: embed the (noisy) known value with InitEmbed and fuse it
            with the pooled context via MixEmbed. `self_val` is the
            reconstructed value — truth + noise in training, the quantised
            recon at inference — so the mix learns to trust it up to eb."""
            value_emb = self.init(self_val.unsqueeze(-1))        # (B, N, d)
            return self.mix(ctx, value_emb)                      # (B, N, d)

        def head_of(self, pooled):
            return self.head(pooled)                             # (B, N)

    return GNN()


def _stage_forward_geoms(model, E, geom_prev, geom_head, finalize_vals, torch,
                         finalize_ctx=None):
    if geom_prev is not None and geom_prev.M:
        ctx = finalize_ctx if finalize_ctx is not None else model.embed(E, geom_prev)
        finalized = model.finalize(ctx, finalize_vals)
        E = E.index_copy(1, geom_prev.query_idx, finalized)     # write newly-known
    head_ctx = model.embed(E, geom_head)
    return model.head_of(head_ctx), E, head_ctx


def stage_forward(model, E, *args, **kwargs):
    """One codec stage of the propagating GNN.

    Supports both APIs:
    - optimized geometry API:
      ``stage_forward(model, E, geom_prev, geom_head, finalize_vals, torch,
      finalize_ctx=None) -> (values, E, head_ctx)``
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
        if kwargs:
            raise TypeError(f"unexpected keyword argument {next(iter(kwargs))!r}")
        return _stage_forward_geoms(model, E, geom_prev, geom_head, finalize_vals,
                                    torch, finalize_ctx=finalize_ctx)

    # Legacy path used by older training/eval code.
    if len(args) < 5:
        raise TypeError("legacy stage_forward needs max_radius and torch")
    prev_mask, known_mask, norm, max_radius, torch = args[:5]
    predict_idx = kwargs.pop("predict_idx", None)
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
    values = model.head_of(model.embed(E, geom_head))
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

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.d = ckpt["d"]
        state_dict = ckpt["state_dict"]
        self.model = build_model(self.d).eval().to(self.device)
        self.model.load_state_dict(state_dict)
        self.checkpoint_hash = hashlib.sha256(
            Path(checkpoint_path).read_bytes()).digest()[:16]
        self._sched: dict = {}   # shape -> (stage geoms, count->index map)
        self._reset()

    def _reset(self):
        self._E = None           # persistent embedding field (C, N, d)
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
               pos: np.ndarray | None = None) -> np.ndarray:
        """Predict the current stage's holes (`pos`, the codec's stage mask).
        Everything scales with the stage: geometry is precomputed at the query
        points only, values are normalized only at the just-revealed points, and
        the finalize context is inherited from the previous stage's head (same
        field, same geometry) instead of being pooled twice."""
        torch = self._torch
        if pos is None:
            raise ValueError("GNNPredictor.predict requires the stage mask `pos`")
        c = recon.shape[0]
        span = self.vmax - self.vmin
        geoms, count_to_i = self._schedule(recon.shape[1:])
        i = count_to_i.get(int(known.sum()))
        if not i:  # None (unknown count) or 0 (anchors are coded directly)
            raise ValueError("known mask does not match the GNN stage schedule")

        cont = self._E is not None and self._stage == i - 1
        if not cont:
            self._E = torch.zeros(c, known.size, self.d, device=self.device)
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
            values, self._E, self._ctx = stage_forward(
                self.model, self._E, geom_prev, geom_head, fvals, torch,
                finalize_ctx=finalize_ctx)
        self._stage = i
        pred = values.cpu().numpy().reshape(c, -1) * span + self.vmin
        return np.clip(pred, self.vmin, self.vmax).astype(np.float32)
