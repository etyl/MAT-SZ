"""Skeleton codec: anchor-grid *lines* coded globally with classic SZ
interpolation, chunk *interiors* coded by the GNN.

Motivation (see plans/polished-coalescing-sky.md): the chunked GNN codec codes
the high-information anchor lines per-chunk, so their reconstruction is
inconsistent across a chunk seam and coding is serialized by the color-wave
order. Here the 1-skeleton (every point lying on an anchor-grid line, i.e. with
exactly one off-grid coordinate) is densified to stride 1 **globally** with the
same 1-D dyadic cubic/linear interpolation SZ3 uses, before any chunk is coded.
Every chunk then sees globally-consistent line context on all sides and only its
interior (>=2 off-grid coords) is left to the GNN.

Point classes for anchor spacing ``stride`` (``anchor_block == 1``):
    anchor          0 coords off-grid            -> global direct quant
    line            exactly 1 coord off-grid     -> global SZ interp (this module)
    interior        >= 2 coords off-grid         -> per-chunk GNN
"""

from __future__ import annotations

import math
import struct

import numpy as np

from . import gnn_predictor as _gp
from .gnn_codec import (
    GNNCompressorCodec,
    _anchor_axes,
    _as_numpy,
    _chunk_stage_ebs,
    _code_anchor_stage,
    _decode_anchor_stage,
    _dtype_meta,
    _log,
    _progress_bar,
    _VERSION_SKEL,
    _read_stream,
    _restore_dtype,
    _write_stream,
)
from .gnn_predictor import (
    ChunkedGNNPredictor,
    build_chunk_geoms,
    _build_remap,
    _CompactGeom,
    _StageGeom,
    _period_prefixes,
    anchor_finalize,
)
from .bitstream import pack_stage, unpack_stage
from .rans import build_laplace_tables, scale_to_level
from .quantizer import dequantize, quantize
from .predictor import _interp_axis_at
from .levels import stage_masks


def _line_stride_levels(stride: int) -> int:
    """Number of dyadic levels to densify a line from anchors to stride 1."""
    L = int(round(math.log2(stride)))
    if (1 << L) != stride:
        raise ValueError(f"anchor_stride must be a power of two, got {stride}")
    return L


def _line_eb(s: int, eb: float, eb_ratio: float) -> float:
    """Level-wise error bound for a line stage of stride ``s`` (finest stride is
    1, so ``depth = log2(s)``): coarser lines get a tighter bound so their error
    propagates less into finer interpolation, and the finest (s=1) keeps the full
    ``eb`` -> global |x-recon| <= eb still holds. Matches ``levels.stage_ebs``."""
    return eb * eb_ratio ** math.log2(s) if s > 1 else eb


def _line_iter(shape, stride, block):
    """Yield (axis, sel, sub_shape, level_strides) for each anchor-line group.

    ``sel`` indexes the separable sub-array holding the axis-``j`` lines: anchor
    coordinates on every axis but ``j``, full resolution on ``j``. Never
    materialises a full-shape mask (mirrors ``_anchor_axes`` / the anchor pass).
    Axis line-sets are disjoint (a line point has exactly one off-grid axis), so
    the axis order is free as long as encoder and decoder agree.
    """
    ndim = len(shape)
    anch = _anchor_axes(shape, stride, block)
    L = _line_stride_levels(stride)
    strides = [stride >> k for k in range(1, L + 1)]
    for j in range(ndim):
        idx = [anch[i] if i != j else np.arange(shape[i]) for i in range(ndim)]
        sel = (slice(None), *np.ix_(*idx))
        sub_shape = tuple(len(a) for a in idx)
        yield j, sel, sub_shape, strides


def _line_midpoint_coords(sub_shape, axis, s):
    """Coords (tuple of index arrays) of the odd-``s`` midpoints along ``axis``
    of a line sub-array: every other-axis position, the odd multiples of ``s``
    along ``axis``. Empty tuple sentinel when there are none."""
    cj = np.arange(sub_shape[axis])
    mid = (cj % s == 0) & (cj % (2 * s) != 0)
    if not mid.any():
        return None
    mask = np.zeros(sub_shape, bool)
    slc = [slice(None)] * len(sub_shape)
    slc[axis] = mid
    mask[tuple(slc)] = True
    return np.nonzero(mask)


def _code_line_pass(values, recon, shape, stride, block, eb, eb_ratio, radius,
                    order, round_output):
    """Encoder side of the global line pass. Densifies every anchor-grid line to
    stride 1 by 1-D dyadic interpolation, writing reconstructed line values into
    ``recon`` and returning the list of packed stage payloads (axis-major,
    level-minor). ``recon`` must already hold the reconstructed anchors."""
    c = values.shape[0]
    parts: list[bytes] = []
    for axis, sel, sub_shape, strides in _line_iter(shape, stride, block):
        sub_v = values[sel]                       # (c, *sub_shape) copy
        sub_r = recon[sel]                         # anchors along axis present
        W = sub_r.astype(np.float64)
        for s in strides:
            coords = _line_midpoint_coords(sub_shape, axis, s)
            if coords is None:
                continue
            pred = _interp_axis_at(W, coords, axis, s, order, sub_shape)
            xv = sub_v[(slice(None), *coords)].reshape(c, -1)
            ebk = _line_eb(s, eb, eb_ratio)
            codes, outliers = quantize(xv, pred, ebk, radius,
                                       round_output=round_output)
            rec = dequantize(pred, codes, outliers, ebk, radius).reshape(c, -1)
            sub_r[(slice(None), *coords)] = rec
            W[(slice(None), *coords)] = rec.astype(np.float64)
            tables = build_laplace_tables(ebk, radius)
            # SZ interp has no per-point scale head, so fit ONE Laplace scale to
            # this stage's residual spread (b = mean|x-pred|, the MLE) and send it
            # as a byte. Hardcoding b=eb over-costs ~2.5x, since line residuals run
            # tens of eb; the decoder can't see residuals, so the level is sent.
            bhat = float(np.abs(xv - pred).mean())
            lvl = int(scale_to_level(np.full((1, 1), bhat, np.float32),
                                     ebk).reshape(-1)[0])
            levels64 = np.full(codes.size, lvl, np.uint8)
            parts.append(struct.pack("<B", lvl)
                         + pack_stage(codes, outliers, rans_levels=levels64,
                                      rans_tables=tables))
        recon[sel] = sub_r
    return parts


def _decode_line_pass(payload, off, recon, shape, stride, block, eb, eb_ratio,
                      radius, order):
    """Decoder twin of ``_code_line_pass``: rebuilds line recon from the payload
    in the identical axis/level order. Returns the new payload offset."""
    c = recon.shape[0]
    for axis, sel, sub_shape, strides in _line_iter(shape, stride, block):
        sub_r = recon[sel]
        W = sub_r.astype(np.float64)
        for s in strides:
            coords = _line_midpoint_coords(sub_shape, axis, s)
            if coords is None:
                continue
            pred = _interp_axis_at(W, coords, axis, s, order, sub_shape)
            ebk = _line_eb(s, eb, eb_ratio)
            m = pred.shape[1]
            tables = build_laplace_tables(ebk, radius)
            (lvl,) = struct.unpack_from("<B", payload, off)
            off += 1
            levels64 = np.full(c * m, lvl, np.uint8)
            codes, outliers, off = unpack_stage(payload, off, rans_levels=levels64,
                                                rans_tables=tables)
            rec = dequantize(pred, codes, outliers, ebk, radius).reshape(c, -1)
            sub_r[(slice(None), *coords)] = rec
            W[(slice(None), *coords)] = rec.astype(np.float64)
        recon[sel] = sub_r
    return off


# --- chunk interior coding (reuses the chunked GNN wave machinery) -----------

def _offgrid_count(cshape, stride):
    """Per-cell count of coordinates off the anchor grid, in the chunk-local
    frame. Chunk origins are multiples of the edge (a multiple of stride), so
    ``local % stride == global % stride`` and the count is origin-independent."""
    ndim = len(cshape)
    cnt = np.zeros(cshape, np.int64)
    for k in range(ndim):
        off = (np.arange(cshape[k]) % stride != 0)
        shp = [1] * ndim
        shp[k] = cshape[k]
        cnt += off.reshape(shp).astype(np.int64)
    return cnt


def _offgrid_global(gc, stride):
    """Off-grid coordinate count for global coords ``gc`` (N, ndim)."""
    return (gc % stride != 0).sum(axis=1)


def _on_internal_boundary(gc, edges):
    """Whether each global coord (N, ndim) lies on >=1 *internal* chunk-boundary
    hyperplane: some axis a with ``coord_a % edge_a == 0`` and ``coord_a > 0``
    (coord 0 is the tensor border, not a shared chunk seam)."""
    on = np.zeros(len(gc), bool)
    for a in range(gc.shape[1]):
        on |= (gc[:, a] % edges[a] == 0) & (gc[:, a] > 0)
    return on


def _classify(cshape, origin, edges, stride):
    """Chunk-frame boolean masks splitting the >=2-off-grid cells of one chunk
    into *interface* and *strict-interior*:
        interface = >=2 off-grid AND on this chunk's low face of some axis whose
                    neighbour across it exists (chunk index > 0). A chunk's region
                    is one edge wide, so the only internal boundary hyperplane it
                    touches is its own low face (local coord 0 on that axis).
        strict    = the remaining >=2 off-grid cells (no chunk-boundary coord).
    Interface cells are coded once (by this, their container chunk) in phase 1;
    strict interiors in phase 2 with the reconstructed interfaces as extra halo."""
    ndim = len(cshape)
    interior = _offgrid_count(cshape, stride) >= 2
    on_low_face = np.zeros(cshape, bool)
    for a in range(ndim):
        if origin[a] % edges[a] == 0 and origin[a] > 0:   # low face is internal
            sl = [slice(None)] * ndim
            sl[a] = 0
            on_low_face[tuple(sl)] = True
    iface = interior & on_low_face
    strict = interior & ~on_low_face
    return iface, strict


class _SubGeoms:
    """Stage geometry for a *subset* of a chunk's refinement points (the interface
    or the strict-interior cells), in the halo-padded local frame. Mirrors
    ``_ChunkGeoms`` but restricts every stage's query set to ``qmask`` and lays
    only those cells as field rows — off-subset neighbours (e.g. off-face strict
    interiors for an interface query) fall outside the present set and are masked
    by ``_CompactGeom``, which is exactly the "same-face / lines-only" context the
    interface pass needs, for free."""

    __slots__ = ("ndim", "padded_shape", "n_padded", "levels", "stride", "block",
                 "geoms", "coords", "chain", "interior_flat", "ref_halo_flat",
                 "ref_halo_coords")

    def __init__(self, cshape, levels, stride, block, qmask, torch, device,
                 agg_level):
        ndim = len(cshape)
        self.ndim = ndim
        self.levels, self.stride, self.block = levels, stride, block
        self.padded_shape = tuple(n + 2 * stride for n in cshape)
        self.n_padded = int(np.prod(self.padded_shape))
        masks = stage_masks(cshape, levels, stride, block)
        pats = _period_prefixes(cshape, levels, stride, block)
        self.geoms, self.coords = [], []
        for s, mask in enumerate(masks):
            Q = np.stack(np.nonzero(mask & qmask), axis=1)
            if not len(Q):
                self.geoms.append(None)
                self.coords.append(None)
                continue
            self.geoms.append(_StageGeom(pats[s], Q + stride, self.padded_shape,
                                         stride, torch, device, agg_level))
            self.coords.append(Q)
        self.chain = [0] + [s for s in range(1, len(self.geoms))
                            if self.geoms[s] is not None]
        qc = np.stack(np.nonzero(qmask), axis=1)
        self.interior_flat = (
            np.ravel_multi_index([(qc[:, k] + stride) for k in range(ndim)],
                                 self.padded_shape)
            if len(qc) else np.zeros(0, np.int64))
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


def _interior_split(cshape, cmasks, stride):
    """For each sub-stage mask, the interior (>=2 off-grid) subset:
    ``(pos_int, sel, n_int)`` where ``pos_int`` is the chunk-shape boolean mask
    of interior points, ``sel`` selects them within ``np.nonzero(pos)`` order
    (aligning with ``predict_stage`` output), and ``n_int`` their count.
    Line points (<=1 off-grid) are sourced from the global line pass instead."""
    interior_cell = _offgrid_count(cshape, stride) >= 2
    out = []
    for pos in cmasks:
        pos_int = pos & interior_cell
        out.append((pos_int, interior_cell[pos], int(pos_int.sum())))
    return out


class _SkelFrame:
    """Compact-field layout for a halo-free chunk: 1 dummy + interior + the
    skeleton-only halo band (contiguous rows), plus the global flat indices to
    fill that band from the recon. Mirrors ``_CompactFrame`` but its halo holds
    only skeleton cells, so it needs no per-chunk coarse table or ``coded`` map."""

    __slots__ = ("geoms", "n_interior", "n_compact", "halo_rows", "h_gflat")

    def __init__(self, geoms, n_interior, n_compact, halo_rows, h_gflat):
        self.geoms = geoms
        self.n_interior = n_interior
        self.n_compact = n_compact
        self.halo_rows = halo_rows
        self.h_gflat = h_gflat


class SkeletonGNNPredictor(ChunkedGNNPredictor):
    """Halo-free / parallel per-chunk predictor for the skeleton codec.

    A chunk's out-of-chunk context is restricted to **skeleton** cells (anchors +
    lines, i.e. <=1 coord off the anchor grid), which are all coded globally
    before any chunk. So a chunk never reads a neighbour chunk's interior: chunks
    are fully independent (any order, no color waves), the halo is filled from the
    global recon via ``anchor_finalize`` (value + null context), and no per-chunk
    coarse table is needed. Reuses ``predict_stage`` unchanged; only the compact
    frame (which cells are pre-known) and the finish step differ."""

    def _skel_halo(self, cg, origin):
        """Skeleton (<=1 off-grid) subset of a chunk's referenced halo band, with
        their global flat indices. Off-grid count is origin-independent (origins
        are multiples of the stride), so this selects the same local band for
        every interior chunk; only in-bounds clipping varies at the tensor edge."""
        ndim = len(self.shape)
        gc = cg.ref_halo_coords + np.asarray(origin, np.int64)     # (R, ndim) global
        shp = np.asarray(self.shape)
        inb = np.all((gc >= 0) & (gc < shp), axis=1)
        gci = gc[inb]
        off = np.zeros(len(gci), np.int64)
        for k in range(ndim):
            off += (gci[:, k] % self.anchor_stride != 0).astype(np.int64)
        ok = off <= 1                                              # skeleton only
        halo_present = cg.ref_halo_flat[inb][ok]
        gflat = (np.ravel_multi_index([gci[ok][:, k] for k in range(ndim)],
                                      self.shape)
                 if ok.any() else np.zeros(0, np.int64))
        return halo_present, gflat

    def _skel_frame(self, cg, origin):
        torch = self._torch
        halo_present, h_gflat = self._skel_halo(cg, origin)
        n_interior = int(len(cg.interior_flat))
        present = np.concatenate([cg.interior_flat, halo_present])
        n_compact = 1 + len(present)
        remap = _build_remap(present, cg.n_padded, torch, self.device)
        geoms = [None if g is None else _CompactGeom(g, remap) for g in cg.geoms]
        return _SkelFrame(geoms, n_interior, n_compact,
                          slice(n_interior + 1, n_compact), h_gflat)

    def start_chunk(self, ci, recon):
        torch = self._torch
        sls = self.chunk_slices(ci)
        origin = np.array([sl.start for sl in sls], np.int64)
        cshape = tuple(sl.stop - sl.start for sl in sls)
        cg = build_chunk_geoms(cshape, self.levels, self.anchor_stride,
                               self.anchor_block, torch, self.device,
                               self.agg_level)
        frame = self._skel_frame(cg, origin)
        E = torch.zeros(self.C, frame.n_compact, cg.ndim, self.d,
                        device=self.device)
        if len(frame.h_gflat):     # skeleton halo: value + null context, no coarse
            vals = self._norm(recon.reshape(self.C, -1)[:, frame.h_gflat])
            with torch.no_grad():
                E[:, frame.halo_rows] = anchor_finalize(
                    self.model, vals, cg.ndim).to(E.dtype)
        self._cg = cg
        self._ci = ci
        self._E = E
        self._geoms = frame.geoms
        self._gidx = [None if c is None else np.ravel_multi_index(
            [(c[:, k] + origin[k]) for k in range(len(self.shape))], self.shape)
            for c in cg.coords]
        self._ctx = None
        self._pos = 0

    def finish_chunk(self, ci, recon):
        # Halo-free: no neighbour reads this chunk's coarse embedding, so skip the
        # chunk_coarse pass finish_chunk normally does. Only mark it done.
        if ci != self._ci:
            raise ValueError("finish_chunk out of order")
        self.coded[ci] = True
        self._E = self._ctx = self._cg = None

    # --- batched (same-geometry) wave path ----------------------------------
    # Because chunks are halo-free, *all* chunks that share a compact frame batch
    # together with no color-wave constraint (unlike the base predictor, which
    # needs the color ordering for coded-neighbour halos). Grouping is by boundary
    # signature only (same tensor-edge clipping => identical skeleton halo band =>
    # identical geometry). ``predict_wave_stage`` is inherited from the base
    # unchanged (it only reads ``self._E``/``self._geoms``/``self._wave_gidx``,
    # all set up here); only the halo fill and finish differ.

    def start_wave(self, chunk_ids, recon):
        torch = self._torch
        ndim = len(self.shape)
        B = len(chunk_ids)
        origins = np.array([[sl.start for sl in self.chunk_slices(ci)]
                            for ci in chunk_ids], np.int64)          # (B, ndim)
        cshape = tuple(sl.stop - sl.start
                       for sl in self.chunk_slices(chunk_ids[0]))
        cg = build_chunk_geoms(cshape, self.levels, self.anchor_stride,
                               self.anchor_block, torch, self.device,
                               self.agg_level)
        frame = self._skel_frame(cg, origins[0])
        E = torch.zeros(B, frame.n_compact, ndim, self.d, device=self.device)
        if len(frame.h_gflat):     # skeleton halo: value + null context per chunk
            band = np.stack(np.unravel_index(frame.h_gflat, self.shape), 1) \
                - origins[0]                                          # (H, ndim)
            flat = recon.reshape(-1)                                 # C == 1
            vals_all = []
            for o in origins:
                gc = band + o
                gflat = np.ravel_multi_index([gc[:, k] for k in range(ndim)],
                                             self.shape)
                vals_all.append(self._norm(flat[gflat][None, :]))    # (1, H)
            with torch.no_grad(), self._amp():
                E[:, frame.halo_rows] = anchor_finalize(
                    self.model, torch.cat(vals_all, 0), ndim).to(E.dtype)
        # per-chunk global flat index per stage (see base start_wave)
        strides = np.cumprod((1,) + self.shape[:0:-1])[::-1].astype(np.int64)
        obase = origins @ strides                                    # (B,)
        self._wave_gidx = [None if c is None else
                           (c @ strides)[None, :] + obase[:, None]    # (B, M)
                           for c in cg.coords]
        self._cg = cg
        self._wave_ids = list(chunk_ids)
        self._E = E
        self._geoms = frame.geoms
        self._ctx = None
        self._pos = 0

    def finish_wave(self, recon):
        # Halo-free: no coarse table to store (nothing reads it). Just mark done.
        self.coded[np.array(self._wave_ids)] = True
        self._E = self._ctx = self._cg = None

    # --- two-phase interface path (Milestone B) -----------------------------
    # Codes a chunk's *interface* cells (phase 1, skeleton-only context) before
    # its *strict-interior* cells (phase 2, skeleton + reconstructed interfaces),
    # so a chunk's strict interiors near a seam see the neighbour's reconstructed
    # face. Each phase runs a query-restricted sub-chunk chain; ``predict_stage``
    # is reused verbatim (it only reads ``self._cg.chain``/``_geoms``/``_gidx``).

    def _sub_frame(self, sub, origin, known_pred):
        """Compact frame for a sub-chunk: query cells first (field rows), then the
        referenced halo band filtered to cells ``known_pred`` says are already
        reconstructed. Fills the halo from the global recon via ``anchor_finalize``
        (value + null context), like the halo-free path."""
        torch = self._torch
        ndim = len(self.shape)
        gc = sub.ref_halo_coords + np.asarray(origin, np.int64)
        shp = np.asarray(self.shape)
        inb = np.all((gc >= 0) & (gc < shp), axis=1)
        gci = gc[inb]
        keep = known_pred(gci) if len(gci) else np.zeros(0, bool)
        halo_present = sub.ref_halo_flat[inb][keep]
        h_gflat = (np.ravel_multi_index([gci[keep][:, k] for k in range(ndim)],
                                        self.shape)
                   if keep.any() else np.zeros(0, np.int64))
        n_interior = int(len(sub.interior_flat))
        present = np.concatenate([sub.interior_flat, halo_present])
        n_compact = 1 + len(present)
        remap = _build_remap(present, sub.n_padded, torch, self.device)
        geoms = [None if g is None else _CompactGeom(g, remap) for g in sub.geoms]
        return geoms, n_interior, n_compact, slice(n_interior + 1, n_compact), h_gflat

    def start_sub(self, ci, recon, sub, known_pred):
        torch = self._torch
        sls = self.chunk_slices(ci)
        origin = np.array([sl.start for sl in sls], np.int64)
        geoms, n_int, n_comp, halo_rows, h_gflat = self._sub_frame(
            sub, origin, known_pred)
        E = torch.zeros(self.C, n_comp, sub.ndim, self.d, device=self.device)
        if len(h_gflat):
            vals = self._norm(recon.reshape(self.C, -1)[:, h_gflat])
            with torch.no_grad():
                E[:, halo_rows] = anchor_finalize(
                    self.model, vals, sub.ndim).to(E.dtype)
        self._cg = sub
        self._ci = ci
        self._E = E
        self._geoms = geoms
        self._gidx = [None if c is None else np.ravel_multi_index(
            [(c[:, k] + origin[k]) for k in range(len(self.shape))], self.shape)
            for c in sub.coords]
        self._ctx = None
        self._pos = 0

    def finish_sub(self):
        self._E = self._ctx = self._cg = None


def _skel_waves(grid):
    """Group chunk ids for batched coding. Chunks are halo-free, so any chunks
    that share a compact frame batch together with no ordering constraint — the
    color waves the base codec needs (for coded-neighbour halos) are unnecessary.
    Group by boundary signature only: same tensor-edge clipping => identical
    skeleton halo band => identical geometry => batchable in the model B dim.
    (Same signature also implies same ragged cshape, as in ``_chunk_waves``.)"""
    groups: dict = {}
    for ci in range(int(np.prod(grid))):
        cidx = np.unravel_index(ci, grid)
        bsig = tuple((int(i) == 0, int(i) == g - 1) for i, g in zip(cidx, grid))
        groups.setdefault(bsig, []).append(ci)
    return list(groups.values())


def _compress_skeleton(values, ebs, radius, round_output, predictor, edges,
                       eb, eb_ratio, order, batch_cap=None):
    """Encode: global anchors + global line pass, then **halo-free** per-chunk GNN
    interiors, coded in batched same-geometry waves (chunks are independent — any
    grouping/order gives a valid stream; the decoder mirrors this one bitwise).
    Only the interior (>=2 off-grid) subset of each sub-stage is entropy-coded;
    lines come from the global pass. Peak memory O(batch * chunk)."""
    c = values.shape[0]
    shape = values.shape[1:]
    stride, block = predictor.anchor_stride, predictor.anchor_block
    if block != 1:
        raise ValueError("skeleton codec requires anchor_block == 1")
    recon = np.zeros_like(values)
    anch = _anchor_axes(shape, stride, block)
    parts = [_code_anchor_stage(values, recon, anch, ebs[0], radius, round_output)]
    parts += _code_line_pass(values, recon, shape, stride, block, eb, eb_ratio,
                             radius, order, round_output)
    _log(f"skel encode: shape={shape} edges={edges} anchors+lines done")
    predictor.begin(shape, edges, channels=c)
    _log("skel encode: chunk geometry ready, sizing model batch...")
    B_cap = predictor.max_batch(tuple(min(e, n) for e, n in zip(edges, shape)))
    if batch_cap is not None:
        B_cap = max(1, min(B_cap, int(batch_cap)))
    predictor.chunk_batch = B_cap
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    mask_cache: dict = {}
    waves = _skel_waves(predictor.grid)
    n_sub = sum(-(-len(group) // B_cap) for group in waves)
    _log(f"skel encode: {predictor.n_chunks} chunks, batch={B_cap}, "
         f"{n_sub} model-waves")
    bar = _progress_bar("skel encode", predictor.n_chunks, unit="chunk")
    for group in waves:
        for i in range(0, len(group), B_cap):
            ids = group[i:i + B_cap]
            cshape = tuple(sl.stop - sl.start
                           for sl in predictor.chunk_slices(ids[0]))
            if cshape not in mask_cache:
                cm = stage_masks(cshape, predictor.levels, stride, block)
                mask_cache[cshape] = (cm, [int(p.sum()) for p in cm],
                                      _interior_split(cshape, cm, stride))
            cmasks, counts, split = mask_cache[cshape]
            predictor.start_wave(ids, recon)
            for s in range(1, len(cmasks)):
                if counts[s] == 0:                 # empty sub-stage: not in chain
                    continue
                pred, scale = predictor.predict_wave_stage(s, recon, ebs[s])
                pos_int, sel, n_int = split[s]
                if n_int == 0:                      # all-line sub-stage: no coding
                    continue
                for bi, ci in enumerate(ids):
                    sls = predictor.chunk_slices(ci)
                    p = pred[bi][sel][None, :]
                    cvals = values[(slice(None), *sls)][:, pos_int]
                    codes, outliers = quantize(cvals, p, ebs[s], radius,
                                               round_output=round_output)
                    recon[(slice(None), *sls)][:, pos_int] = dequantize(
                        p, codes, outliers, ebs[s], radius).reshape(c, n_int)
                    parts.append(pack_stage(
                        codes, outliers,
                        rans_levels=scale_to_level(
                            scale[bi][sel][None, :], ebs[s]).reshape(-1),
                        rans_tables=stage_tables[s]))
            predictor.finish_wave(recon)
            bar.update(len(ids))
    bar.close()
    return b"".join(parts)


def _decompress_skeleton(payload, shape, ebs, radius, predictor, edges, batch,
                         eb, eb_ratio, order):
    """Decoder twin of ``_compress_skeleton`` (identical wave/stage/chunk order)."""
    c = 1
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros((c, *shape), np.float32)
    anch = _anchor_axes(shape, stride, block)
    off = _decode_anchor_stage(payload, 0, recon, anch, ebs[0], radius)
    off = _decode_line_pass(payload, off, recon, shape, stride, block, eb,
                            eb_ratio, radius, order)
    predictor.begin(shape, edges, channels=c)
    B_cap = max(1, int(batch))
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    mask_cache: dict = {}
    waves = _skel_waves(predictor.grid)
    n_sub = sum(-(-len(group) // B_cap) for group in waves)
    _log(f"skel decode: {predictor.n_chunks} chunks, batch={B_cap}, "
         f"{n_sub} model-waves")
    bar = _progress_bar("skel decode", predictor.n_chunks, unit="chunk")
    for group in waves:
        for i in range(0, len(group), B_cap):
            ids = group[i:i + B_cap]
            cshape = tuple(sl.stop - sl.start
                           for sl in predictor.chunk_slices(ids[0]))
            if cshape not in mask_cache:
                cm = stage_masks(cshape, predictor.levels, stride, block)
                mask_cache[cshape] = (cm, [int(p.sum()) for p in cm],
                                      _interior_split(cshape, cm, stride))
            cmasks, counts, split = mask_cache[cshape]
            predictor.start_wave(ids, recon)
            for s in range(1, len(cmasks)):
                if counts[s] == 0:
                    continue
                pred, scale = predictor.predict_wave_stage(s, recon, ebs[s])
                pos_int, sel, n_int = split[s]
                if n_int == 0:
                    continue
                for bi, ci in enumerate(ids):
                    sls = predictor.chunk_slices(ci)
                    levels64 = scale_to_level(
                        scale[bi][sel][None, :], ebs[s]).reshape(-1)
                    codes, outliers, off = unpack_stage(
                        payload, off, rans_levels=levels64,
                        rans_tables=stage_tables[s])
                    recon[(slice(None), *sls)][:, pos_int] = dequantize(
                        pred[bi][sel][None, :], codes, outliers, ebs[s],
                        radius).reshape(c, n_int)
            predictor.finish_wave(recon)
            bar.update(len(ids))
    bar.close()
    if off != len(payload):
        raise ValueError("trailing bytes in skeleton payload")
    return recon[0]


def _skel_iface_passes(values, recon, ebs, radius, round_output, predictor,
                       shape, stride, block, *, decode=False, payload=None,
                       off0=0):
    """Shared two-phase driver — phase 1 codes every chunk's interface cells
    (skeleton-only context), phase 2 its strict interiors (skeleton + the now-
    reconstructed interfaces). Encodes (returns the list of packed parts) or
    decodes into ``recon`` (returns the new payload offset); one function so the
    two directions are bit-for-bit mirrors. Both phases are chunk-order and
    (given the pre-coded skeleton/interfaces) mutually independent."""
    edges = predictor.edges
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    c = 1 if decode else values.shape[0]
    parts: list[bytes] = []
    off = off0

    def known_iface(gci):
        return _offgrid_global(gci, stride) <= 1                     # skeleton
    def known_interior(gci):
        return (_offgrid_global(gci, stride) <= 1) | \
            _on_internal_boundary(gci, edges)                        # + interfaces

    sub_cache: dict = {}
    for phase, known_pred in (("iface", known_iface),
                              ("interior", known_interior)):
        direction = "decode" if decode else "encode"
        bar = _progress_bar(f"skel {phase} {direction}", predictor.n_chunks,
                            unit="chunk")
        for ci in range(predictor.n_chunks):
            sls = predictor.chunk_slices(ci)
            origin = np.array([sl.start for sl in sls], np.int64)
            cshape = tuple(sl.stop - sl.start for sl in sls)
            key = (cshape, tuple(bool(o > 0) for o in origin), phase)
            sub = sub_cache.get(key)
            if sub is None:
                iface, strict = _classify(cshape, origin, edges, stride)
                qmask = iface if phase == "iface" else strict
                sub = _SubGeoms(cshape, predictor.levels, stride, block, qmask,
                                predictor._torch, predictor.device,
                                predictor.agg_level)
                sub_cache[key] = sub
            if len(sub.chain) <= 1:                 # no query cells this phase
                bar.update(1)
                continue
            predictor.start_sub(ci, recon, sub, known_pred)
            for jj in range(1, len(sub.chain)):
                s = sub.chain[jj]
                pred, scale = predictor.predict_stage(s, recon, ebs[s])
                Q = sub.coords[s]
                qidx = tuple(Q[:, k] for k in range(len(cshape)))
                M = len(Q)
                lvl = scale_to_level(scale[0][None, :], ebs[s]).reshape(-1)
                if decode:
                    codes, outliers, off = unpack_stage(
                        payload, off, rans_levels=lvl, rans_tables=stage_tables[s])
                    recon[(slice(None), *sls)][(slice(None), *qidx)] = dequantize(
                        pred[0][None, :], codes, outliers, ebs[s],
                        radius).reshape(c, M)
                else:
                    p = pred[0][None, :]
                    cvals = values[(slice(None), *sls)][(slice(None), *qidx)]
                    codes, outliers = quantize(cvals, p, ebs[s], radius,
                                               round_output=round_output)
                    recon[(slice(None), *sls)][(slice(None), *qidx)] = dequantize(
                        p, codes, outliers, ebs[s], radius).reshape(c, M)
                    parts.append(pack_stage(codes, outliers, rans_levels=lvl,
                                            rans_tables=stage_tables[s]))
            predictor.finish_sub()
            bar.update(1)
        bar.close()
    return off if decode else parts


def _compress_skeleton_iface(values, ebs, radius, round_output, predictor, edges,
                             eb, eb_ratio, order):
    """Encode with the Milestone-B interface class (see ``_skel_iface_passes``)."""
    c = values.shape[0]
    shape = values.shape[1:]
    stride, block = predictor.anchor_stride, predictor.anchor_block
    if block != 1:
        raise ValueError("skeleton codec requires anchor_block == 1")
    recon = np.zeros_like(values)
    anch = _anchor_axes(shape, stride, block)
    parts = [_code_anchor_stage(values, recon, anch, ebs[0], radius, round_output)]
    parts += _code_line_pass(values, recon, shape, stride, block, eb, eb_ratio,
                             radius, order, round_output)
    _log(f"skel(iface) encode: shape={shape} edges={edges} anchors+lines done")
    predictor.begin(shape, edges, channels=c)
    parts += _skel_iface_passes(values, recon, ebs, radius, round_output,
                                predictor, shape, stride, block)
    return b"".join(parts)


def _decompress_skeleton_iface(payload, shape, ebs, radius, predictor, edges,
                               eb, eb_ratio, order):
    """Decoder twin of ``_compress_skeleton_iface`` (identical phase/chunk/stage
    order)."""
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros((1, *shape), np.float32)
    anch = _anchor_axes(shape, stride, block)
    off = _decode_anchor_stage(payload, 0, recon, anch, ebs[0], radius)
    off = _decode_line_pass(payload, off, recon, shape, stride, block, eb,
                            eb_ratio, radius, order)
    predictor.begin(shape, edges, channels=1)
    off = _skel_iface_passes(None, recon, ebs, radius, None, predictor, shape,
                             stride, block, decode=True, payload=payload, off0=off)
    if off != len(payload):
        raise ValueError("trailing bytes in skeleton payload")
    return recon[0]


class SkeletonGNNCodec(GNNCompressorCodec):
    """Skeleton codec: anchor-grid lines coded globally with SZ cubic/linear
    interpolation, chunk interiors coded by the GNN. Always chunked (the point is
    global line context across chunk seams). ``line_order`` selects the classic
    interpolation order for the global line pass."""

    _CHUNKED_PREDICTOR = SkeletonGNNPredictor

    def __init__(self, *args, line_order: str = "cubic",
                 interfaces: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if line_order not in ("linear", "cubic"):
            raise ValueError("line_order must be 'linear' or 'cubic'")
        if self.anchor_block != 1:
            raise ValueError("skeleton codec requires anchor_block == 1")
        self.line_order = line_order
        # Milestone B: code chunk-boundary interiors (interfaces) in a global
        # phase 1 so a chunk's strict interiors see the neighbour's reconstructed
        # face. Off by default (the halo-free path is faster and ~parity on ratio).
        self.interfaces = bool(interfaces)

    def _skel_edges(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Chunk edges, forcing the chunked path (skeleton needs >=1 chunk).
        Falls back to one chunk (shape rounded up to a stride multiple) for inputs
        the base would keep whole-tensor."""
        edges = self._chunk_edges(shape)
        if edges is not None:
            return edges
        s = self.anchor_stride
        return tuple(-(-n // s) * s for n in shape)

    def compress(self, x, error_bound: float | None = None) -> bytes:
        arr = np.asarray(_as_numpy(x))
        if arr.size == 0:
            raise ValueError("cannot compress an empty tensor")
        if arr.dtype.kind not in "biuf":
            raise TypeError(f"unsupported dtype {arr.dtype}; expected numeric data")
        dtype = np.dtype(arr.dtype)
        original_shape = tuple(int(n) for n in arr.shape)
        shape = original_shape if original_shape else (1,)
        values = arr.reshape(shape).astype(np.float32, copy=False)[None, ...]
        vmin, vmax = float(values.min()), float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        eb = self.error_bound if error_bound is None else float(error_bound)
        if eb <= 0:
            raise ValueError("error_bound must be > 0")
        ratio_candidates = (
            [float(self.eb_ratio)] if self.eb_ratio is not None
            else ([1.0, 0.9, 0.8, 0.7] if self.tune == "size" else [0.8]))
        edges = self._skel_edges(shape)
        use_compile = self.compile and int(np.prod(
            [-(-n // e) for n, e in zip(shape, edges)])) >= 64
        candidates: list[tuple[int, bytes]] = []
        for ratio in ratio_candidates:
            predictor = self._chunked_predictor(vmin, vmax)
            predictor.compile = bool(use_compile)
            ebs = _chunk_stage_ebs(shape, self.levels, self.anchor_stride,
                                   self.anchor_block, eb, ratio)
            if self.interfaces:
                payload = _compress_skeleton_iface(
                    values, ebs, self.radius, dtype.kind in "bi", predictor,
                    edges, eb, ratio, self.line_order)
            else:
                payload = _compress_skeleton(
                    values, ebs, self.radius, dtype.kind in "bi", predictor,
                    edges, eb, ratio, self.line_order, self.chunk_batch)
            meta = {
                "codec": "deepsz.gnn", "skeleton": True,
                "line_order": self.line_order, "interfaces": self.interfaces,
                "shape": list(original_shape), "coded_shape": list(shape),
                "dtype": _dtype_meta(dtype), "error_bound": eb,
                "levels": self.levels, "anchor_stride": self.anchor_stride,
                "anchor_block": self.anchor_block, "radius": self.radius,
                "max_radius": self.max_radius, "agg_level": self.agg_level,
                "vmin": vmin, "vmax": vmax, "eb_ratio": ratio,
                "entropy_coder": "rans",
                "checkpoint_hash": self.checkpoint_hash.hex(),
                "chunks": list(edges),
                "chunk_batch": int(predictor.chunk_batch),
                "m_tile": int(_gp._M_TILE), "fp16": bool(self.fp16),
                "compiled": bool(use_compile),
            }
            stream = _write_stream(meta, payload, self.zstd_level, _VERSION_SKEL)
            candidates.append((len(stream), stream))
        return min(candidates, key=lambda it: it[0])[1]

    def uncompress(self, stream):
        import torch
        meta, payload = _read_stream(bytes(stream))
        if meta.get("codec") != "deepsz.gnn" or not meta.get("skeleton"):
            raise ValueError("not a DeepSZ skeleton stream")
        got_hash = meta.get("checkpoint_hash")
        if self.strict_checkpoint and got_hash != self.checkpoint_hash.hex():
            raise ValueError("checkpoint hash differs from the stream metadata")
        shape = tuple(int(n) for n in meta["coded_shape"])
        original_shape = tuple(int(n) for n in meta["shape"])
        dtype = np.dtype(meta["dtype"]["str"])
        vmin, vmax = float(meta["vmin"]), float(meta["vmax"])
        if vmax <= vmin:
            vmax = vmin + 1.0
        edges = tuple(int(e) for e in meta["chunks"])
        predictor = self._chunked_predictor(vmin, vmax, meta)
        ebs = _chunk_stage_ebs(shape, int(meta["levels"]),
                               int(meta["anchor_stride"]),
                               int(meta["anchor_block"]),
                               float(meta["error_bound"]),
                               float(meta["eb_ratio"]))
        saved_tile = _gp._M_TILE
        _gp._M_TILE = int(meta.get("m_tile", saved_tile))
        try:
            if meta.get("interfaces"):
                values = _decompress_skeleton_iface(
                    payload, shape, ebs, int(meta["radius"]), predictor, edges,
                    float(meta["error_bound"]), float(meta["eb_ratio"]),
                    meta.get("line_order", "cubic"))
            else:
                values = _decompress_skeleton(
                    payload, shape, ebs, int(meta["radius"]), predictor, edges,
                    int(meta.get("chunk_batch", 1)), float(meta["error_bound"]),
                    float(meta["eb_ratio"]), meta.get("line_order", "cubic"))
        finally:
            _gp._M_TILE = saved_tile
        out = _restore_dtype(values.reshape(original_shape), dtype)
        return torch.as_tensor(out)

    decompress = uncompress
