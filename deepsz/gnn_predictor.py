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

import contextlib
import hashlib
import itertools
import math
import os
from pathlib import Path

import numpy as np

from .levels import point_levels, stage_masks

# torch is imported lazily inside the class / model so that importing this
# module (e.g. for FLAG constants) stays cheap.

CKPT_VERSION = 5

# Query-tile for the message pass: caps the transient (B, L, m, K, d) buffers so
# GPU peak scales with _M_TILE, not the stage's full M (line_pool is per-query,
# so tiling is math-identical within a stream). Off by default: a fixed tile
# shreds the whole-tensor path (a big 2D stage has M~150k -> ~150 tiny blocks,
# ~2x slower/image) and isn't needed now that batch=1 fits (batching is
# saturated) and fp16 halves the buffers. Set DEEPSZ_M_TILE=1024 only when
# running big chunk_batch on tight VRAM. Stored in meta so decode replays it.
_M_TILE = int(os.environ.get("DEEPSZ_M_TILE", 1 << 30))  # effectively no tiling


def half_directions(ndim: int, agg_level: int | None = None) -> list[tuple[int, ...]]:
    """One representative per line: all offset vectors in {-1,0,1}^ndim whose
    first non-zero component is +1 (so d and -d collapse to one line).

    ``agg_level`` caps the *neighbourhood aggregation level* — the L1 length of
    the direction, i.e. its number of non-zero components (how many axes a hop
    moves along at once). Level 1 keeps only the ``ndim`` axis-aligned face
    directions (direct neighbours); level 2 adds the 2-axis diagonals (2 hops in
    L1); ... level ``ndim`` (or ``None``) keeps all ``(3^ndim - 1)/2`` lines, the
    full neighbourhood. Since the network is direction-blind (direction enters
    only as the rotary phase and the pool masks unused lines), dropping the
    higher-L1 lines is a pure inference-time cost/accuracy trade-off that shrinks
    the per-stage message tensor's L dimension — the dominant factor in high-D."""
    dirs = []
    for d in itertools.product((-1, 0, 1), repeat=ndim):
        first = next((x for x in d if x != 0), 0)
        if first > 0 and (agg_level is None or sum(x != 0 for x in d) <= agg_level):
            dirs.append(d)
    return dirs


def _nearest_steps(pat: np.ndarray, dvec, P: int, res=None) -> np.ndarray:
    """Smallest step t>=1 along ``dvec`` landing on a True cell of the periodic
    pattern ``pat`` (period P), or 0 if none. The hit sequence along any lattice
    line is periodic in t with period dividing P, so the nearest hit is the first
    within [1,P]. With ``res`` (a tuple of ndim residue arrays, len M), evaluated
    only at those M query residues -> O(P*M); without it, at every residue of the
    P^ndim tile -> O(P^(ndim+1)). Query points are all we ever index, so pass res."""
    if res is None:
        res = np.indices(pat.shape)                     # every residue: (ndim, P, ...)
    t0 = np.zeros(res[0].shape, np.int64)               # 0 == no hit yet
    for t in range(1, P + 1):
        r = tuple((res[k] + t * dvec[k]) % P for k in range(pat.ndim))
        take = (t0 == 0) & pat[r]
        t0[take] = t
        if np.all(t0):
            break
    return t0


_NEAREST_TILE_CACHE: dict = {}


def _nearest_steps_at(pat: np.ndarray, dvec, P: int, res, *,
                      query_only: bool = False) -> np.ndarray:
    """`_nearest_steps` evaluated at the query residues ``res``, via a cached
    full period tile when that is cheaper: the tile costs O(P^(ndim+1)) once
    per (pat, dvec) and O(M) per lookup, vs O(P*M) per call for the direct
    path. ponytail: tile capped at 2^20 cells (4-D at stride 32); beyond that
    fall back to the direct path rather than build a giant tile."""
    M = len(res[0])
    # A chunk schedule partitions one period tile across many stages. Building
    # a full P**ndim lookup independently for every (stage, direction) looks
    # cheaper for a single large stage, but is catastrophically expensive over
    # the whole schedule (76 stages for levels=5 in 4-D). Across all stages the
    # query counts sum to only P**ndim, so evaluating at query residues is the
    # linear-work strategy for chunk geometry.
    if query_only:
        return _nearest_steps(pat, dvec, P, res)
    if pat.size > 1 << 20 or pat.size > P * M:
        return _nearest_steps(pat, dvec, P, res)
    key = (pat.tobytes(), pat.shape, tuple(int(c) for c in dvec), P)
    tile = _NEAREST_TILE_CACHE.get(key)
    if tile is None:
        tile = _nearest_steps(pat, dvec, P)
        _NEAREST_TILE_CACHE[key] = tile
    return tile[tuple(res)]


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

    def __init__(self, pat, query_coords, shape, max_radius, torch, device,
                 agg_level=None, query_only=False):
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
        for d in half_directions(ndim, agg_level):
            ln = {}
            for side, sd in (("p", np.asarray(d)), ("n", -np.asarray(d))):
                if not self.M:
                    ln["i" + side] = t(np.zeros(0, np.int64))
                    ln["d" + side] = t(np.zeros(0, np.float32))
                    ln["v" + side] = t(np.zeros(0, bool))
                    continue
                step = _nearest_steps_at(
                    pat, sd, P, res, query_only=query_only)     # (M,) at query residues
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

    def __init__(self, known, max_radius, torch, device=None, query_idx=None,
                 agg_level=None):
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
        for h in half_directions(known.ndim, agg_level):
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


def build_stage_geoms(shape, levels, stride, block, max_radius, torch, device=None,
                      agg_level=None):
    """Per-stage `_StageGeom` list (empty stages dropped) plus a
    ``|known|-before-stage -> list index`` map, for the whole schedule of one
    region shape. Closed-form lattice geometry, computed at the query points
    only; cached per (shape, levels, stride, block, max_radius, agg_level,
    device) and shared by encoder tuning sweeps, decoder, and the trainer.

    ``agg_level`` caps the neighbourhood aggregation level (see
    `half_directions`); ``None`` keeps the full neighbourhood.

    ponytail: unbounded cache, bounded in practice (a handful of shapes/configs);
    add an LRU cap only if a caller feeds unboundedly many distinct configs."""
    key = (tuple(int(n) for n in shape), levels, stride, block, max_radius,
           agg_level, str(device))
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
            geoms.append(_StageGeom(pats[s], Q, shape, max_radius, torch, device,
                                    agg_level))
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
    F = nn.functional

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
            cos = cos.to(e.dtype)        # geom is fp32; match model dtype (fused)
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
            # Split the first Linear over its input blocks instead of concatenating
            # e (B,L,M,K,d) with the two scalar columns — avoids materializing the
            # big (…,d+2) buffer (the `cat` was ~14% of GPU time). Same math; net.*
            # weights are unchanged so checkpoints load as-is.
            w = self.net[0].weight                       # (h, d+2)
            x = F.linear(e, w[:, :d], self.net[0].bias) \
                + F.linear(torch.cat([sign, logd], -1), w[:, d:])
            for layer in self.net[1:]:
                x = layer(x)
            return x

    class BiDirEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [2 * d + 2, h, d])

        def forward(self, e_neg, e_pos, logd_neg, logd_pos):
            w = self.net[0].weight                       # (h, 2d+2)
            x = F.linear(e_neg, w[:, :d]) \
                + F.linear(e_pos, w[:, d:2 * d], self.net[0].bias) \
                + F.linear(torch.cat([logd_neg, logd_pos], -1), w[:, 2 * d:])
            for layer in self.net[1:]:
                x = layer(x)
            return x

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
            # Softmax in fp32 even when the model is fp16: -inf-masked scores
            # overflow/NaN in half. Cast the weights back to v's dtype so the
            # weighted sum stays in the model dtype (was autocast's fp32 rule).
            w = torch.softmax(scores.float(), dim=0).to(v.dtype)  # (L+1, B, N)
            return (w.unsqueeze(-1) * v).sum(0)  # (B, N, d)

    class PredHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [d + 1, h, 2])
            # Start unconfident: delta ~= 8 (b ~= 256*eb), inside the clamp so
            # gradients flow both ways. A zero init means b ~= eb, and with
            # untrained predictions (|r| ~ 0.1) at eb=1e-6 the NLL tail then
            # fires ~1/b gradient spikes from step 1.
            with torch.no_grad():
                self.net[-1].bias[1] = 8.0

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
            # Laplace scale is eb-relative: `delta` spans the deployed rANS scale
            # grid [eb/16, 4096*eb] (log2 offsets -4..12, see rans.SCALE_LO_DIV/
            # SCALE_HI_MULT), so the head can express sub-eb confidence at ANY eb.
            # The old span-relative clamp(-8,0) pinned every point to the broadest
            # grid levels at low eb; the earlier eb-relative ceiling of +6 pinned
            # ~half the points at very low eb (<=1e-5), where prediction error
            # stays orders of magnitude above eb (bench_levels sat+%).
            delta = out[..., 1].clamp(-4.0, 12.0)
            log_b = log_eb.view(B, 1) + delta
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

        def _line_messages(self, E, geom, msl=slice(None)):
            """Per-line messages for a slice ``msl`` of this stage's M query
            points, built from neighbour *embeddings* E (not raw values) so
            trends/periodicity propagate hop by hop. Returns msgs (L, B, m, K, d)
            and valid (L, m) — the axis dim K is carried through Dir/BiDir as a
            batch dim, each axis reading its neighbours through the rotary phase.
            geom already holds the +/- neighbour of each query point, so this
            touches O(m) rows of the field, never the whole grid."""
            B = E.shape[0]
            ip, in_ = geom.ip[:, msl], geom.in_[:, msl]     # (L, m)
            dp, dn = geom.dp[:, msl], geom.dn[:, msl]
            vp, vn = geom.vp[:, msl], geom.vn[:, msl]
            # Batch all directions into the MLPs at once. This turns the old
            # per-line Dir/BiDir calls into three larger matmuls per embed pass,
            # which is much friendlier to GPU inference.
            ep = E[:, ip]                           # (B, L, m, K, d)
            en = E[:, in_]
            ep = self.rope(ep, geom.cos, 1.0)       # (B, L, m, K, d)
            en = self.rope(en, geom.cos, -1.0)
            _, L, M, K, _ = ep.shape
            # .to(ep.dtype): geom distances are fp32; match model dtype so the
            # dir/bidir MLPs get fp16 inputs under true-half (cast fuses in embed).
            lp = (torch.log2(dp) + geom.lognnz).to(ep.dtype
                  ).view(1, L, M, 1, 1).expand(B, L, M, K, 1)
            lnn = (torch.log2(dn) + geom.lognnz).to(ep.dtype
                   ).view(1, L, M, 1, 1).expand(B, L, M, K, 1)
            sign = ep.new_ones(B, L, M, K, 1)
            both = (vp & vn).view(1, L, M, 1, 1)
            vp_only = (vp & ~vn).view(1, L, M, 1, 1)
            # Every single-neighbour point uses the *same* dir weights, differing
            # only in which neighbour/sign/dist it feeds. Select that input before
            # the MLP (ep if only +, else en) so dir runs once, not twice — the
            # second pass was always discarded by the where. Both-neighbour points
            # get overwritten by bidir. Still one big buffer at a time, so the
            # finest-stage memory peak is unchanged.
            e_sel = torch.where(vp_only, ep, en)
            msg = self.dir(e_sel,
                           torch.where(vp_only, sign, -sign),
                           torch.where(vp_only, lp, lnn))
            msg = torch.where(both, self.bidir(en, ep, lnn, lp), msg)
            del ep, en, e_sel
            return (msg.permute(1, 0, 2, 3, 4).contiguous(), (vp | vn))

        def _embed_block(self, E, geom, msl):
            msgs, valid = self._line_messages(E, geom, msl)  # (L,B,m,K,d),(L,m)
            L, B, m, K, _ = msgs.shape
            flat = msgs.reshape(L, B, m * K, self.d)
            vflat = valid.repeat_interleave(K, dim=1)        # (L, m*K)
            ctx = self.line_pool(flat, vflat)                # (B, m*K, d)
            return ctx.reshape(B, m, K, self.d)              # (B, m, ndim, d)

        def embed(self, E, geom):
            """Per-axis contexts at geom's query points: single-query attention
            over the per-line neighbour messages (no self value), pooled
            independently per axis. For an anchor with no known neighbours every
            line is masked and each axis falls back to the learned null token.

            Tiled over the query dim: the transient (B, L, m, K, d) message
            buffers and their 2*d-wide MLP activations are the dominant GPU peak
            at the finest stages, so we cap m at _M_TILE and stream blocks into a
            small (B, M, K, d) output. line_pool is per-query independent, so
            this is bit-identical to embedding all M at once."""
            M = geom.M
            if M <= _M_TILE:
                return self._embed_block(E, geom, slice(None))
            ctx = E.new_empty(E.shape[0], M, geom.ndim, self.d)
            for m0 in range(0, M, _M_TILE):
                msl = slice(m0, min(m0 + _M_TILE, M))
                ctx[:, msl] = self._embed_block(E, geom, msl)
            return ctx

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
            # The predicted value (mu, log_b) is what gets quantized against, so
            # keep the readout in fp32 even under fp16 autocast — confines fp16 to
            # the message pass and protects compression ratio at small eb.
            with torch.autocast(device_type=ctx.device.type, enabled=False):
                ctx = ctx.float()
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
    # One live revision per (path, device): the trainer overwrites the eval
    # checkpoint every eval, so keying on mtime alone leaks a model (and its
    # compiled embed) per revision until OOM.
    for k in [k for k in _MODEL_CACHE if k[0] == key[0] and k[3] == key[3]]:
        del _MODEL_CACHE[k]

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
        finalized = model.finalize(ctx, finalize_vals).to(E.dtype)  # fp16 -> E dtype
        # ponytail: in-place. no grad here, and each model.embed re-stages E, so
        # mutating it between calls is safe; out-of-place clones E every stage
        # (~200ms of DtoD at chunk 32). Only unsafe if E were a CUDA-graph static
        # buffer, but reduce-overhead re-copies inputs, so eager E stays mutable.
        E.index_copy_(1, geom_prev.query_idx, finalized)        # write newly-known
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
    tunable = True    # encoder sweeps eb_ratio (no centre mode; see codec.encode)
    fast_eb_ratio = 0.8  # single-encode (tune=fast) default; tighter coarse
                         # levels help the learned fine-level prediction
    provides_scale = True
    fp16 = False      # fp16 autocast on the message pass (encode/decode must match)
    compile = False   # torch.compile the embed pass (encode/decode must match)

    def _maybe_compile(self):
        # Wrap the embed pass once. It fuses the elementwise message-pass ops
        # (rope/where/dir/bidir), the ~40% of GPU time not in the GEMMs.
        # dynamic=True: one graph for every stage/chunk M, no recompile storm.
        # enc and dec both compile (flag replayed) so their float paths match.
        if self.compile and not getattr(self, "_compiled", False):
            # DEEPSZ_COMPILE_MODE=reduce-overhead -> CUDA graphs, kills per-kernel
            # launch latency on the ~30 tiny message-pass kernels (launch-bound).
            # ponytail: CUDA graphs want static shapes; with varying stage M they
            # recapture per new shape, so it only wins once shapes settle/repeat.
            mode = os.environ.get("DEEPSZ_COMPILE_MODE") or None
            self.model.embed = self._torch.compile(
                self.model.embed, dynamic=True, mode=mode)
            self._compiled = True

    def _amp(self):
        self._maybe_compile()
        if self.fp16 and self.device.type == "cuda":
            return self._torch.autocast(device_type="cuda",
                                        dtype=self._torch.float16)
        return contextlib.nullcontext()

    def __init__(self, checkpoint_path, vmin: float, vmax: float,
                 max_radius: int = 64, device: str = "cpu",
                 levels: int = 4, anchor_stride: int = 16, anchor_block: int = 1,
                 agg_level: int | None = None):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.max_radius = int(max_radius)
        self.levels = int(levels)
        self.anchor_stride = int(anchor_stride)
        self.anchor_block = int(anchor_block)
        # Neighbourhood aggregation level: cap on the L1 length of the neighbour
        # lines (see `half_directions`). None = full neighbourhood. Encoder and
        # decoder must agree, so the codec stores it in the stream meta.
        self.agg_level = None if agg_level is None else int(agg_level)

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
                                  self._torch, self.device, self.agg_level)
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
        with torch.no_grad(), self._amp():
            (values, log_b), self._E, self._ctx = stage_forward(
                self.model, self._E, geom_prev, geom_head, fvals, torch,
                finalize_ctx=finalize_ctx, eb=norm_eb)
        self._stage = i
        vals_np, logb_np = torch.stack((values, log_b)).cpu().numpy()  # one D2H
        pred = vals_np.reshape(c, -1) * span + self.vmin
        scale = np.exp2(logb_np.reshape(c, -1)) * span
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

    def __init__(self, chunk_shape, levels, stride, block, torch, device,
                 agg_level=None, progress=None):
        self.chunk_shape = tuple(int(n) for n in chunk_shape)
        self.levels, self.stride, self.block = levels, stride, block
        self.agg_level = agg_level
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
                if progress is not None:
                    progress(1)
                continue
            self.geoms.append(_StageGeom(pats[s], Q + stride, self.padded_shape,
                                         stride, torch, device, agg_level,
                                         query_only=True))
            self.coords.append(Q)
            if progress is not None:
                progress(1)
        # prediction chain: stage 0 is always the base (anchors, possibly empty
        # in a ragged tail chunk -> None geom, nothing to finalize), followed by
        # every non-empty refinement stage in order.
        self.chain = [0] + [s for s in range(1, len(self.geoms))
                            if self.geoms[s] is not None]

        # Interior padded-flat indices, built directly (O(interior), never the
        # O(shell) full padded grid). The compact field lays interior first, so
        # interior cell i has compact index i + 1 (row 0 is a dummy the invalid
        # / not-yet-decoded neighbour lines point at).
        idx0 = np.indices(self.chunk_shape).reshape(ndim, -1)     # chunk-frame
        self.interior_flat = np.ravel_multi_index(idx0 + stride, self.padded_shape)
        lv = point_levels(list(idx0), levels, stride, block)
        self.level_pos = [np.nonzero(lv == l)[0].astype(np.int64) + 1
                          for l in range(levels + 1)]

        # Padded-flat halo cells that appear as a *valid* neighbour of some
        # stage: the thin band the field must actually hold. Derived from the
        # stage geometries (O(interior)), so the dead rest of the shell is never
        # materialised. Its chunk-frame coords let the per-chunk halo pass test
        # usability without an O(shell) mask.
        seen = []
        for g in self.geoms:
            if g is None:
                continue
            seen.append(g.ip[g.vp])
            seen.append(g.in_[g.vn])
        ref = (torch.unique(torch.cat(seen)).cpu().numpy() if seen
               else np.zeros(0, np.int64))
        self.ref_halo_flat = ref[~np.isin(ref, self.interior_flat)].astype(np.int64)
        self.ref_halo_coords = (np.stack(
            np.unravel_index(self.ref_halo_flat, self.padded_shape), 1) - stride)


_CHUNK_GEOM_CACHE: dict = {}


def build_chunk_geoms(chunk_shape, levels, stride, block, torch, device=None,
                      agg_level=None, progress=None):
    """Cached `_ChunkGeoms` per (chunk shape, schedule, agg_level, device).
    Interior chunks all share one entry; ragged edge chunks add at most a few
    shape variants (ponytail: unbounded like _GEOM_CACHE, bounded in practice).

    ``agg_level`` caps the neighbourhood aggregation level (see
    `half_directions`); ``None`` keeps the full neighbourhood."""
    key = (tuple(int(n) for n in chunk_shape), levels, stride, block, agg_level,
           str(device))
    hit = _CHUNK_GEOM_CACHE.get(key)
    if hit is None:
        hit = _ChunkGeoms(chunk_shape, levels, stride, block, torch, device,
                          agg_level, progress)
        _CHUNK_GEOM_CACHE[key] = hit
    elif progress is not None:
        progress(len(hit.geoms))
    return hit


def _build_remap(present, n_padded, torch, device):
    """Map padded-flat index -> compact field row. ``present`` lists the padded
    cells the field holds (interior first, then usable halo); cell ``present[i]``
    lives at row ``i + 1`` (row 0 is the dummy that invalid / not-yet-decoded
    neighbour lines point at). A dense table over the padded frame makes each
    remap one gather instead of a sort + searchsorted + compare chain.
    ponytail: table capped at 2^24 cells (~128MB, covers 4-D chunks at stride
    32); bigger frames take the searchsorted path."""
    if n_padded <= 1 << 24:
        table = np.zeros(n_padded, np.int64)
        table[present] = np.arange(1, len(present) + 1)
        tt = torch.from_numpy(table).to(device)
        return lambda flat: tt[flat]

    order = np.argsort(present, kind="stable")
    spres = torch.from_numpy(np.ascontiguousarray(present[order])).to(device)
    comp = torch.from_numpy((order + 1).astype(np.int64)).to(device)
    n = spres.numel()

    def remap(flat):                 # (torch int64) -> (torch int64) compact
        if n == 0:
            return torch.zeros_like(flat)
        pos = torch.searchsorted(spres, flat).clamp_(max=n - 1)
        return torch.where(spres[pos] == flat, comp[pos],
                           torch.zeros_like(flat))
    return remap


class _CompactGeom:
    """A `_StageGeom` with its neighbour indices remapped into a chunk's compact
    field and its periodic validity ANDed with runtime usability. A neighbour is
    usable iff it landed on a real compact row (remap != 0): interior always
    does, halo only when that cell is decoded and referenced. Shares every other
    tensor with the base geometry."""

    __slots__ = _StageGeom.__slots__

    def __init__(self, base, remap):
        for name in ("dp", "dn", "cos", "lognnz", "idx_np", "M", "ndim"):
            setattr(self, name, getattr(base, name))
        self.ip = remap(base.ip)
        self.in_ = remap(base.in_)
        self.query_idx = remap(base.query_idx)
        self.vp = base.vp & (self.ip != 0)
        self.vn = base.vn & (self.in_ != 0)


class _CompactFrame:
    """Per-chunk compact field layout: geoms with remapped indices, and the
    (contiguous) halo row block plus the metadata to fill it from coarse+value.
    ``n_compact`` = 1 dummy + interior + usable-referenced halo."""

    __slots__ = ("geoms", "n_interior", "n_compact", "halo_rows",
                 "h_ids", "h_lv", "h_gflat")

    def __init__(self, cg, origin, shape, edges, grid, coded, torch, device):
        halo_present, h_ids, h_lv, h_gflat = chunk_halo_info(
            cg, origin, shape, edges, grid, coded)
        self.n_interior = int(len(cg.interior_flat))
        present = np.concatenate([cg.interior_flat, halo_present])
        self.n_compact = 1 + len(present)
        remap = _build_remap(present, cg.n_padded, torch, device)
        self.geoms = [None if g is None else _CompactGeom(g, remap)
                      for g in cg.geoms]
        # halo cells are laid out right after interior, so their rows are a
        # contiguous slice — no remap needed to fill them.
        self.halo_rows = slice(self.n_interior + 1, self.n_compact)
        self.h_ids, self.h_lv, self.h_gflat = h_ids, h_lv, h_gflat


def chunk_halo_info(cg, origin, shape, edges, grid, coded):
    """Usable, referenced halo cells for one chunk of a chunk grid.

    Walks only the referenced band ``cg.ref_halo_flat`` (never the O(shell)
    padded frame). Returns ``(halo_present, chunk_ids, lv, gflat)`` for the band
    cells that are inside the tensor and already decoded (coded chunk, or a
    global anchor): their padded flat index, owning chunk id, dyadic level and
    global flat index. Shared by the inference predictor and the trainer so both
    build identical context."""
    ndim = len(shape)
    gc = cg.ref_halo_coords + np.asarray(origin, np.int64)
    shp = np.asarray(shape)
    inb = np.all((gc >= 0) & (gc < shp), axis=1)
    gci = gc[inb]
    chunk_ids = np.ravel_multi_index(
        [gci[:, k] // edges[k] for k in range(ndim)], grid)
    lv = point_levels([gci[:, k] for k in range(ndim)],
                      cg.levels, cg.stride, cg.block)
    ok = np.asarray(coded)[chunk_ids] | (lv == 0)
    halo_present = cg.ref_halo_flat[inb][ok]
    gflat = np.ravel_multi_index(
        [gci[ok][:, k] for k in range(ndim)], shape)
    return halo_present, chunk_ids[ok], lv[ok], gflat


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
    for l, pos in enumerate(cg.level_pos):
        if not len(pos):
            continue
        idx = torch.from_numpy(pos).to(E_pad.device)         # compact rows
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
    chunk_batch = 1              # sub-batch size chosen by the codec, into meta
    fp16 = False                # fp16 autocast on the message pass (codec sets it)
    compile = False             # torch.compile the embed pass (codec sets it)

    def _maybe_compile(self):
        # Wrap the embed pass once (fuses the elementwise message-pass ops that
        # aren't in the GEMMs). dynamic=True keeps one graph across all M sizes;
        # enc and dec both compile (flag replayed) so their float paths match.
        if self.compile and not getattr(self, "_compiled", False):
            # DEEPSZ_COMPILE_MODE=reduce-overhead -> CUDA graphs, kills per-kernel
            # launch latency on the ~30 tiny message-pass kernels (launch-bound).
            # ponytail: CUDA graphs want static shapes; with varying stage M they
            # recapture per new shape, so it only wins once shapes settle/repeat.
            mode = os.environ.get("DEEPSZ_COMPILE_MODE") or None
            self.model.embed = self._torch.compile(
                self.model.embed, dynamic=True, mode=mode)
            self._compiled = True

    def __init__(self, checkpoint_path, vmin: float, vmax: float,
                 device: str = "cpu", levels: int = 4, anchor_stride: int = 16,
                 anchor_block: int = 1, agg_level: int | None = None):
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
        # Neighbourhood aggregation level (see GNNPredictor / half_directions).
        self.agg_level = None if agg_level is None else int(agg_level)
        self.d, self.model, self.checkpoint_hash = _load_inference_model(
            checkpoint_path, torch, self.device)

    # -- per-tensor lifecycle -------------------------------------------------
    def begin(self, shape, chunk_edges, channels: int = 1,
              geometry_progress=None):
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
        self._check_field_budget(ndim, channels, geometry_progress)
        self.coarse = torch.zeros(self.C, self.n_chunks, self.levels + 1,
                                  ndim, self.d, device=self.device)
        self.coded = np.zeros(self.n_chunks, bool)
        self._cg = None

    def _check_field_budget(self, ndim, channels, geometry_progress=None):
        """Fail fast (with numbers + fix) before torch OOMs. Two terms scale with
        chunk size: the compact field (1 dummy + interior + referenced halo band)
        and, larger, one stage's message tensor (B, L, M, K, d) where L is the
        half-direction count ((3^ndim-1)/2, so 40 in 4-D) and M the stage's query
        count. The embed pass tiles M by ``DEEPSZ_M_TILE``, so its activation
        estimate must use the tile size rather than the full stage."""
        torch = self._torch
        cshape = tuple(min(e, n) for e, n in zip(self.edges, self.shape))
        cg = build_chunk_geoms(cshape, self.levels, self.anchor_stride,
                               self.anchor_block, torch, self.device,
                               self.agg_level, geometry_progress)
        n_interior = int(len(cg.interior_flat))
        n_band = int(len(cg.ref_halo_flat))          # upper bound (all referenced)
        field_bytes = channels * (1 + n_interior + n_band) * ndim * self.d * 4
        M = max((g.M for g in cg.geoms if g is not None), default=0)
        m = min(M, _M_TILE)
        L = len(half_directions(ndim, self.agg_level))
        act_bytes = 4 * channels * L * m * ndim * self.d * 4  # ~4 live copies
        need = field_bytes + act_bytes
        if self.device.type == "cuda":
            budget = int(0.8 * torch.cuda.mem_get_info(self.device)[0])
        else:                                # cpu: cap at 64 GiB, still catches it
            budget = 64 << 30
        if need > budget:
            raise MemoryError(
                f"chunked GNN needs up to {need/1e9:.0f} GB per chunk "
                f"(field {field_bytes/1e9:.1f} + stage activation "
                f"{act_bytes/1e9:.1f} GB for tile={m:,} of M={M:,} queries "
                f"x L={L} lines, budget {budget/1e9:.1f} GB). Lower "
                f"DEEPSZ_M_TILE, or reduce both --chunk-size and "
                f"--anchor-stride (for example 16 and 16).")

    def chunk_slices(self, ci: int):
        cidx = np.unravel_index(ci, self.grid)
        return tuple(slice(i * e, min((i + 1) * e, n))
                     for i, e, n in zip(cidx, self.edges, self.shape))

    def _norm(self, vals: np.ndarray):
        v = (np.clip(vals, self.vmin, self.vmax) - self.vmin) / self.span
        return self._torch.from_numpy(v.astype(np.float32)).to(self.device)

    def anchor_coarse(self, recon: np.ndarray, progress=None):
        """Level-0 coarse embeddings for every chunk, right after the global
        anchor pass — so every chunk has anchor context on all sides before any
        chunk is coded. Anchors finalize from the null context, so this is a
        pure function of their reconstructed values. Chunks with the same anchor
        count batch together in the model's B dim (C is always 1 here)."""
        torch = self._torch
        ndim = len(self.shape)
        log_s = math.log2(self.anchor_stride)
        groups: dict = {}                         # anchor-count -> list of (ci, vals)
        empty_chunks = 0
        for ci in range(self.n_chunks):
            axes = []
            for sl in self.chunk_slices(ci):
                c = np.arange(sl.start, sl.stop)
                axes.append(c[(c % self.anchor_stride) < self.anchor_block])
            if any(len(a) == 0 for a in axes):
                empty_chunks += 1
                continue                          # ragged chunk with no anchors
            vals = recon[(slice(None), *np.ix_(*axes))].reshape(-1)   # C==1
            groups.setdefault(len(vals), ([], []))
            groups[len(vals)][0].append(ci)
            groups[len(vals)][1].append(vals)
        if progress is not None and empty_chunks:
            progress(empty_chunks)
        with torch.no_grad(), self._amp():
            for ids, vlist in groups.values():
                v = np.stack(vlist)                             # (G, M_a)
                fin = anchor_finalize(self.model, self._norm(v), ndim)
                # `coarse` persists between waves, so keep it fp32 even though
                # its producer ran under autocast.
                self.coarse[0, ids, 0] = self.model.coarse(
                    fin.mean(1), log_s).to(self.coarse.dtype)
                if progress is not None:
                    progress(len(ids))

    # ---- batched wave path (codec): many same-geometry chunks in the B dim ----

    def _amp(self):
        self._maybe_compile()
        if self.fp16 and self.device.type == "cuda":
            return self._torch.autocast(device_type="cuda",
                                        dtype=self._torch.float16)
        return contextlib.nullcontext()

    def max_batch(self, cshape):
        """How many same-shape chunks fit in the memory budget at once. Bounded by
        the finest stage's activation (B, L, M, K, d) plus the compact field, the
        two terms that scale with B (see `_check_field_budget`)."""
        torch = self._torch
        cg = build_chunk_geoms(cshape, self.levels, self.anchor_stride,
                               self.anchor_block, torch, self.device,
                               self.agg_level)
        ndim = len(cshape)
        n_field = 1 + int(len(cg.interior_flat)) + int(len(cg.ref_halo_flat))
        M = max((g.M for g in cg.geoms if g is not None), default=1)
        m = min(M, _M_TILE)                                   # embed tiles over M
        # _line_messages peaks at ~ep + en + msg + one Dir/BiDir (hidden width 2d)
        # all (B, L, m, K, d)-sized and live at once. ~8x the base buffer covers
        # those plus the concat/where temporaries; the field E persists alongside.
        # ponytail: this is a static estimate — the encode/decode progress line
        # prints the *measured* GPU peak, so lower --chunk-batch if that nears VRAM.
        per = ((n_field * ndim * self.d)                      # persistent field
               + 8 * len(half_directions(ndim, self.agg_level))
               * m * ndim * self.d) * 4
        if self.device.type == "cuda":
            budget = int(0.8 * torch.cuda.mem_get_info(self.device)[0])
        else:
            budget = 8 << 30                                   # cpu: modest cap
        return max(1, budget // max(per, 1))

    def start_wave(self, chunk_ids, recon: np.ndarray):
        """Begin a batch of mutually-independent, identical-geometry chunks. One
        representative frame drives the shared stage geometry; the halo/interior
        field values are gathered per chunk into the model's B dim."""
        torch = self._torch
        ndim = len(self.shape)
        B = len(chunk_ids)
        origins = np.array([[sl.start for sl in self.chunk_slices(ci)]
                            for ci in chunk_ids], np.int64)     # (B, ndim)
        cshape = tuple(sl.stop - sl.start
                       for sl in self.chunk_slices(chunk_ids[0]))
        cg = build_chunk_geoms(cshape, self.levels, self.anchor_stride,
                               self.anchor_block, torch, self.device,
                               self.agg_level)
        frame = _CompactFrame(cg, origins[0], self.shape, self.edges, self.grid,
                              self.coded, torch, self.device)
        E = torch.zeros(B, frame.n_compact, ndim, self.d, device=self.device)
        self._wave_fill_halo(E, frame, origins, recon)
        # per-chunk global flat indices per stage, for finalize reads from recon.
        # ravel(c + o) = c @ strides + o @ strides for in-bounds coords, so one
        # dot per stage replaces the (B, M, ndim) ravel_multi_index.
        strides = np.cumprod((1,) + self.shape[:0:-1])[::-1].astype(np.int64)
        obase = origins @ strides                              # (B,)
        self._wave_gidx = [None if c is None else
                           (c @ strides)[None, :] + obase[:, None]   # (B, M)
                           for c in cg.coords]
        self._cg = cg
        self._wave_ids = list(chunk_ids)
        self._E = E
        self._geoms = frame.geoms
        self._ctx = None
        self._pos = 0

    def _wave_fill_halo(self, E, frame, origins, recon):
        torch = self._torch
        ndim = len(self.shape)
        if frame.halo_rows.stop <= frame.halo_rows.start:
            return
        # band chunk-frame coords are shared across the wave (rep uses origins[0]);
        # per chunk only the global positions / owning chunks shift.
        rep_gc = np.stack(np.unravel_index(frame.h_gflat, self.shape), 1)
        band = rep_gc - origins[0]                             # (H, ndim)
        h_lv = torch.from_numpy(frame.h_lv.astype(np.int64)).to(self.device)
        flat = recon.reshape(-1)                               # C == 1
        vals_all, cvec_all = [], []
        for o in origins:
            gc = band + o
            gflat = np.ravel_multi_index([gc[:, k] for k in range(ndim)],
                                         self.shape)
            ids = np.ravel_multi_index(
                [gc[:, k] // self.edges[k] for k in range(ndim)], self.grid)
            vals_all.append(self._norm(flat[gflat][None, :]))         # (1, H)
            ids_t = torch.from_numpy(ids).to(self.device)
            cvec_all.append(self.coarse[0, ids_t, h_lv])              # (H, K, d)
        with torch.no_grad(), self._amp():
            E[:, frame.halo_rows] = halo_embed(
                self.model, torch.stack(cvec_all, 0), torch.cat(vals_all, 0)
            ).to(E.dtype)

    def predict_wave_stage(self, s: int, recon: np.ndarray, eb: float):
        """Batched `predict_stage`: returns (pred, scale) of shape (B, M) for the
        wave's B chunks, ordered like ``np.nonzero`` of the local stage mask."""
        torch = self._torch
        cg = self._cg
        j = self._pos + 1
        if j >= len(cg.chain) or cg.chain[j] != s:
            raise ValueError(f"stage {s} out of order for this wave")
        prev = cg.chain[j - 1]
        gp, gh = self._geoms[prev], self._geoms[s]
        fvals = None if gp is None else \
            self._norm(recon.reshape(-1)[self._wave_gidx[prev]])   # (B, M)
        with torch.no_grad(), self._amp():
            (values, log_b), self._E, self._ctx = stage_forward(
                self.model, self._E, gp, gh, fvals, torch,
                finalize_ctx=self._ctx, eb=float(eb) / self.span)
        self._pos = j
        # One D2H for both heads: two separate .cpu() are two blocking syncs per
        # stage, and this stage-loop is the wall bottleneck (GPU ~30% utilized).
        vals_np, logb_np = torch.stack((values, log_b)).cpu().numpy()
        pred = vals_np * self.span + self.vmin                     # (B, M)
        scale = np.exp2(logb_np) * self.span
        return (np.clip(pred, self.vmin, self.vmax).astype(np.float32),
                scale.astype(np.float32))

    def finish_wave(self, recon: np.ndarray):
        torch = self._torch
        cg = self._cg
        if self._pos != len(cg.chain) - 1:
            raise ValueError("finish_wave before all non-empty stages predicted")
        last = cg.chain[self._pos]
        g = self._geoms[last]
        E = self._E
        with torch.no_grad(), self._amp():
            if g is not None:
                fvals = self._norm(recon.reshape(-1)[self._wave_gidx[last]])
                ctx = self._ctx if self._ctx is not None else \
                    self.model.embed(E, g)
                fin = self.model.finalize(ctx, fvals).to(E.dtype)  # fp16->E dtype
                E.index_copy_(1, g.query_idx, fin)              # ponytail: in-place, see _stage_forward_geoms
            cc = chunk_coarse(self.model, E, cg, torch)   # (B, levels+1, K, d)
        ids = np.array(self._wave_ids)
        self.coarse[0, ids] = cc.to(self.coarse.dtype)
        self.coded[ids] = True
        self._E = self._ctx = self._cg = None

    def start_chunk(self, ci: int, recon: np.ndarray):
        torch = self._torch
        sls = self.chunk_slices(ci)
        origin = np.array([sl.start for sl in sls], np.int64)
        cshape = tuple(sl.stop - sl.start for sl in sls)
        cg = build_chunk_geoms(cshape, self.levels, self.anchor_stride,
                               self.anchor_block, torch, self.device,
                               self.agg_level)

        # compact field: 1 dummy + interior + usable-referenced halo band only —
        # the dead rest of the padded shell is never allocated.
        frame = _CompactFrame(cg, origin, self.shape, self.edges, self.grid,
                              self.coded, torch, self.device)
        E = torch.zeros(self.C, frame.n_compact, cg.ndim, self.d,
                        device=self.device)
        if len(frame.h_gflat):     # fill the halo rows from coarse + recon value
            vals = self._norm(recon.reshape(self.C, -1)[:, frame.h_gflat])
            ids = torch.from_numpy(frame.h_ids).to(self.device)
            lvs = torch.from_numpy(frame.h_lv.astype(np.int64)).to(self.device)
            cvec = self.coarse[:, ids, lvs]                # (C, Hs, K, d)
            with torch.no_grad():
                E[:, frame.halo_rows] = halo_embed(self.model, cvec, vals)

        self._cg = cg
        self._ci = ci
        self._E = E
        self._geoms = frame.geoms
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
        with torch.no_grad(), self._amp():
            (values, log_b), self._E, self._ctx = stage_forward(
                self.model, self._E, gp, gh, fvals, torch,
                finalize_ctx=self._ctx, eb=float(eb) / self.span)
        self._pos = j
        vals_np, logb_np = torch.stack((values, log_b)).cpu().numpy()  # one D2H
        pred = vals_np.reshape(self.C, -1) * self.span + self.vmin
        scale = np.exp2(logb_np.reshape(self.C, -1)) * self.span
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
                E.index_copy_(1, g.query_idx,                   # ponytail: in-place, see _stage_forward_geoms
                              self.model.finalize(ctx, fvals))
            self.coarse[:, ci] = chunk_coarse(self.model, E, cg, torch)
        self.coded[ci] = True
        self._E = self._ctx = self._cg = None
