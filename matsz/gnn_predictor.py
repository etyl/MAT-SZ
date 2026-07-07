"""Lightweight, dimension-agnostic GNN predictor for the MAT-SZ closed loop.

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
  * pool the per-line messages with a single-query attention layer (AttnPool),
    then read out the value (PredHead).

The five modules are exposed as separate nn.Modules as requested.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import numpy as np

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
    """Nearest known sample stepping along +dvec. Returns (idx, dist, valid):
    idx = flat index of that neighbour (0 where none), dist = step count,
    valid = whether a neighbour was found within max_radius."""
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


class _Geometry:
    """Precomputed, param-free neighbour geometry for one `known` mask.
    Holds per-line gather indices / distances / validity as torch tensors."""

    def __init__(self, known: np.ndarray, max_radius: int, torch, device=None):
        n = known.size
        flat = np.arange(n, dtype=np.int64).reshape(known.shape)
        self.shape = known.shape
        self.lines = []

        def t(a):  # numpy -> torch on the target device (indices/masks/dists)
            x = torch.from_numpy(a.reshape(-1))
            return x.to(device) if device is not None else x

        for h in half_directions(known.ndim):
            neg = tuple(-c for c in h)  # the -h side
            ip, dp, vp = _nearest_in_dir(known, flat, h, max_radius)
            in_, dn, vn = _nearest_in_dir(known, flat, neg, max_radius)
            self.lines.append({
                "ip": t(ip),
                "in": t(in_),
                "dp": t(dp),
                "dn": t(dn),
                "vp": t(vp),
                "vn": t(vn),
            })


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

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d
            self.init = InitEmbed()
            self.dir = DirEmbed()
            self.bidir = BiDirEmbed()
            self.attn = AttnPool()
            self.head = PredHead()

        def _line_messages(self, E, geom: _Geometry):
            """Per-line messages built from neighbour *embeddings* E (not raw
            values), so trends/periodicity propagate hop by hop. Returns
            msgs (L, B, N, d) and valid (L, N)."""
            B, N, _ = E.shape
            one = torch.ones(B, N, 1, device=E.device)
            msgs, valids = [], []
            for ln in geom.lines:
                ep = E[:, ln["ip"]]                # (B, N, d) +side neighbour
                en = E[:, ln["in"]]                # -side neighbour
                lp = torch.log2(ln["dp"]).view(1, N, 1).expand(B, N, 1)
                lnn = torch.log2(ln["dn"]).view(1, N, 1).expand(B, N, 1)
                bidir = self.bidir(en, ep, lnn, lp)
                dpos = self.dir(ep, one, lp)
                dneg = self.dir(en, -one, lnn)
                both = (ln["vp"] & ln["vn"]).view(1, N, 1)
                vp_only = (ln["vp"] & ~ln["vn"]).view(1, N, 1)
                msg = torch.where(both, bidir, torch.where(vp_only, dpos, dneg))
                msgs.append(msg)
                valids.append(ln["vp"] | ln["vn"])  # (N,)
            return torch.stack(msgs, 0), torch.stack(valids, 0)

        def embed(self, E, geom, self_val=None, self_valid=None):
            """Pooled embedding for every point: attention over the line
            messages plus, for points whose own value is known,
            InitEmbed(self_val). For an anchor (no neighbours) that lone init
            message is de facto the embedding."""
            msgs, valid = self._line_messages(E, geom)
            if self_val is not None:
                _, N = self_val.shape
                init_msg = self.init(self_val.unsqueeze(-1))     # (B, N, d)
                msgs = torch.cat([msgs, init_msg[None]], 0)
                valid = torch.cat([valid, self_valid.view(1, N)], 0)
            return self.attn(msgs, valid)                        # (B, N, d)

        def head_of(self, pooled):
            return self.head(pooled)                             # (B, N)

    return GNN()


def stage_forward(model, E, prev_mask, known_mask, norm, max_radius, torch):
    """One codec stage of the propagating GNN, shared by encoder, decoder and
    trainer so all three evolve the embedding field identically.

    E: (B, N, d) field from earlier stages. prev_mask / known_mask: bool nd
    grids of what was known before / is known now (known_mask >= prev_mask;
    the current stage's targets are NOT yet in known_mask, matching the codec).
    norm: (B, N) normalized values (only entries under known_mask are read).
    Returns (values (B, N), E_new (B, N, d))."""
    # ponytail: each call rebuilds both geometries and embeds all N points
    # (finalize uses only `newly`); fine on real HW, subset the gather if the
    # python-loop scan becomes the bottleneck on large grids.
    device = E.device
    N = known_mask.size
    newly = known_mask & ~prev_mask                 # revealed since last stage
    if newly.any():
        geom_prev = _Geometry(prev_mask, max_radius, torch, device)
        sv = torch.from_numpy(newly.reshape(-1)).to(device)
        pooled = model.embed(E, geom_prev, norm, sv)
        E = torch.where(sv.view(1, N, 1), pooled, E)  # finalize their embeddings
    geom = _Geometry(known_mask, max_radius, torch, device)
    values = model.head_of(model.embed(E, geom))    # predict every point
    return values, E


class GNNPredictor:
    """GNN predictor loaded from a trained checkpoint. `tile_size` defaults to
    64; `max_radius` caps the neighbour search (anchors always sit closer)."""

    from .bitstream import FLAG_GNN as _FLAG
    stream_flag = _FLAG

    def __init__(self, checkpoint_path, vmin: float, vmax: float,
                 tile_size: int = 64, max_radius: int = 64, device: str = "cpu"):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.tile_size = int(tile_size)
        self.max_radius = int(max_radius)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.d = ckpt["d"]
        self.model = build_model(self.d).eval().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.checkpoint_hash = hashlib.sha256(
            Path(checkpoint_path).read_bytes()).digest()[:16]
        self._E = None          # persistent embedding field (B, N, d)
        self._prev = None       # bool nd mask known before the current call

    def predict(self, recon: np.ndarray, known: np.ndarray) -> np.ndarray:
        torch = self._torch
        c = recon.shape[0]
        span = self.vmax - self.vmin
        norm = (np.clip(recon, self.vmin, self.vmax) - self.vmin) / span
        norm = np.where(known[None], norm, 0.5).astype(np.float32)

        # New tile / sequence: the field resets whenever `known` is not a strict
        # superset of the previous call (e.g. the first, anchors-only stage).
        cont = (self._prev is not None and self._prev.shape == known.shape
                and known.sum() > self._prev.sum()
                and bool((known & self._prev == self._prev).all()))
        if not cont:
            self._E = torch.zeros(c, known.size, self.d, device=self.device)
            self._prev = np.zeros(known.shape, bool)

        x = torch.from_numpy(norm.reshape(c, -1)).to(self.device)
        with torch.no_grad():
            values, self._E = stage_forward(self.model, self._E, self._prev,
                                            known, x, self.max_radius, torch)
        self._prev = known.copy()
        pred = values.cpu().numpy().reshape(recon.shape) * span + self.vmin
        return np.clip(pred, self.vmin, self.vmax).astype(np.float32)
