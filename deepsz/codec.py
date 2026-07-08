"""Closed-loop DeepSZ codec: tiling, progressive prediction, quantization.

The encoder simulates the decoder: reconstructions fed back into the predictor
are built exclusively from dequantize() outputs, never from the original data,
so decoder-side predictions match encoder-side predictions bitwise (same
platform) and the error bound holds end to end.
"""

from __future__ import annotations

import time

import numpy as np

from .bitstream import (DTYPE_CODES, DTYPE_IDS, FLAG_CUBIC, FLAG_GRAY,
                        FLAG_INTERP, FLAG_MOCK, FLAG_NOTILE, Header, pack_stage,
                        read_stream, unpack_stage, write_stream)
from .levels import stage_ebs, stage_masks
from .predictor import InterpPredictor, MockPredictor
from .quantizer import dequantize, quantize


def compress(
    img: np.ndarray,
    eb: float,
    predictor,
    levels: int = 4,
    anchor_stride: int = 16,
    anchor_block: int = 1,
    radius: int = 1 << 15,
    seed: int = 1234,
    zstd_level: int = 9,
    eb_ratio: float | None = None,
    tune: str = "fast",
    tune_size_slack: float = 1.05,
    verbose: bool = False,
) -> tuple[bytes, dict]:
    """Compress an (H, W) or (H, W, C) array. Returns (stream bytes, stats).

    ``eb_ratio`` scales the per-level error bound (coarse levels tighter, see
    ``levels.stage_ebs``): pass a value in (0, 1] to fix it. With ``eb_ratio``
    omitted, ``tune`` controls the search: ``fast`` runs one candidate, ``size``
    sweeps candidates and keeps the smallest stream, and ``rd`` keeps the
    lowest reconstruction SSE within ``tune_size_slack`` of the smallest stream.
    1.0 reproduces flat-eb classic SZ."""
    if img.ndim == 2:
        img = img[..., None]
    h, w, c = img.shape
    if c not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got {c}")
    src_dtype = DTYPE_IDS[np.dtype(img.dtype)]

    fimg = img.astype(np.float32)
    vmin = float(fimg.min())
    vmax = float(fimg.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    # A predictor with its own schedule (interp) must agree with the header
    # params, or encoder masks and the decoder's rebuilt masks silently diverge.
    for name, val in (("levels", levels), ("anchor_stride", anchor_stride),
                      ("anchor_block", anchor_block)):
        got = getattr(predictor, name, val)
        if got != val:
            raise ValueError(f"predictor {name}={got} != compress {name}={val}")

    # Tile-free predictors (interp) compress the whole image as one region: no
    # padding, no seam. Others tile into square tile_size blocks (MAT needs 512).
    notile = getattr(predictor, "tile_free", False)
    if notile:
        th = tw = 0
        ty = tx = 1
        canvas = fimg.transpose(2, 0, 1)  # (C, H, W), no padding
    else:
        th = tw = predictor.tile_size
        ty, tx = -(-h // th), -(-w // tw)
        canvas = np.pad(fimg, ((0, ty * th - h), (0, tx * tw - w), (0, 0)), mode="edge")
        canvas = canvas.transpose(2, 0, 1)

    _, ch, cw = canvas.shape
    mh, mw = (ch, cw) if notile else (th, tw)
    # interp supplies its own sub-pass split; others use the plain dyadic schedule
    make_masks = getattr(predictor, "stage_masks", stage_masks)
    masks = make_masks((mh, mw), levels, anchor_stride, anchor_block)
    flags = getattr(predictor, "stream_flag", 0) | (FLAG_GRAY if c == 1 else 0)
    flags |= FLAG_NOTILE if notile else 0
    round_output = np.issubdtype(np.dtype(img.dtype), np.integer)
    step_h, step_w = ch // ty, cw // tx

    def encode(ebs):
        st = {"predict_s": 0.0, "quantize_s": 0.0, "entropy_s": 0.0,
              "outliers": 0, "stage_codes": [0] * len(masks),
              "stage_outliers": [0] * len(masks),
              "stage_payload_bytes": [0] * len(masks),
              "stage_pred_sae": [0.0] * len(masks),
              "stage_pred_sse": [0.0] * len(masks),
              "stage_recon_sae": [0.0] * len(masks),
              "stage_recon_sse": [0.0] * len(masks),
              "stage_recon_max": [0.0] * len(masks)}
        pays, rc = [], np.empty_like(canvas)
        for i in range(ty):
            for j in range(tx):
                tile = canvas[:, i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
                payload, recon = _compress_tile(tile, masks, ebs, predictor, radius,
                                                round_output, st)
                rc[:, i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w] = recon
                pays.append(payload)
        return pays, rc, sum(len(p) for p in pays), st

    def ebs_for(ratio):
        return stage_ebs((mh, mw), levels, anchor_stride, anchor_block, eb, ratio)

    def make_header(center, ratio):
        return Header(orig_h=h, orig_w=w, channels=c, src_dtype=src_dtype,
                      eb=float(eb), levels=levels, anchor_stride=anchor_stride,
                      anchor_block=anchor_block, tile_size=th, radius=radius,
                      seed=seed, vmin=vmin, vmax=vmax,
                      ckpt_hash=getattr(predictor, "checkpoint_hash", b"\0" * 16),
                      n_tiles_y=ty, n_tiles_x=tx, flags=flags,
                      interp_center=center, eb_ratio=ratio)

    if tune not in ("fast", "size", "rd"):
        raise ValueError("tune must be 'fast', 'size', or 'rd'")

    # Encoder-side tuning: optionally sweep per-level eb ratio (coarse-error
    # damping) and, for interp, centre-combine mode. The finest level always
    # carries the full eb, so every candidate satisfies the bound.
    tunable = getattr(predictor, "tunable", False)
    has_center = hasattr(predictor, "center")
    if eb_ratio is not None:
        ratio_cands = [eb_ratio]
    elif tunable and tune != "fast":
        ratio_cands = [1.0, 0.9, 0.8, 0.7]
    else:
        ratio_cands = [1.0]
    base_center = getattr(predictor, "center", 0)
    center_cands = [base_center]
    if tunable and has_center and tune != "fast" and eb_ratio is None:
        center_cands += [c for c in (0, 1, 2) if c != base_center]

    candidates = []
    for ratio in ratio_cands:
        for center in center_cands:
            if has_center:
                predictor.center = center
            pays, rc, raw_bytes, st = encode(ebs_for(ratio))
            header = make_header(center, ratio)
            t0 = time.time()
            stream = write_stream(header, pays, zstd_level)
            st["entropy_s"] += time.time() - t0
            st["raw_payload_bytes"] = raw_bytes
            st["compressed_bytes"] = len(stream)
            st["recon_sse"] = float(sum(st["stage_recon_sse"]))
            candidates.append((len(stream), st["recon_sse"], ratio, center,
                               stream, rc, st))

    if tune == "rd" and eb_ratio is None and tunable and len(candidates) > 1:
        min_size = min(c[0] for c in candidates)
        size_limit = min_size * tune_size_slack
        feasible = [c for c in candidates if c[0] <= size_limit]
        best = min(feasible, key=lambda c: (c[1], c[0], c[2], c[3]))
    else:
        best = min(candidates, key=lambda c: (c[0], c[1], c[2], c[3]))

    _, _, chosen_ratio, chosen_center, stream, recon_canvas, stats = best
    if has_center:
        predictor.center = chosen_center
    stats["eb_ratio"] = chosen_ratio
    stats["interp_center"] = chosen_center
    stats["tune"] = tune
    stats["tune_size_slack"] = (tune_size_slack if tune == "rd"
                                and eb_ratio is None and tunable else 1.0)
    if verbose:
        print(f"tuned: eb_ratio={chosen_ratio} center={chosen_center} "
              f"compressed={len(stream)} bytes raw={stats['raw_payload_bytes']} bytes "
              f"sse={stats['recon_sse']:.6g}")

    stats["recon"] = _finalize(recon_canvas, make_header(chosen_center, chosen_ratio))
    stats["original_bytes"] = img.nbytes
    stats["ratio"] = img.nbytes / len(stream)
    return stream, stats


def decompress(stream: bytes, predictor_factory=None) -> np.ndarray:
    """Decompress a DeepSZ stream. ``predictor_factory(header) -> predictor``;
    defaults to the torch-free MockPredictor / InterpPredictor for their streams
    (GNN streams need a factory that builds a GNNPredictor from the checkpoint)."""
    header, payloads = read_stream(stream)
    if predictor_factory is None:
        if header.flags & FLAG_MOCK:
            predictor_factory = lambda hdr: MockPredictor(hdr.tile_size)
        elif header.flags & FLAG_INTERP:
            predictor_factory = lambda hdr: InterpPredictor(
                hdr.tile_size, "cubic" if hdr.flags & FLAG_CUBIC else "linear",
                hdr.levels, hdr.anchor_stride, hdr.anchor_block)
        else:
            raise ValueError("stream needs a predictor_factory (GNN checkpoint)")
    predictor = predictor_factory(header)
    if hasattr(predictor, "center"):
        predictor.center = header.interp_center

    ty, tx = header.n_tiles_y, header.n_tiles_x
    if header.flags & FLAG_NOTILE:
        step_h, step_w = header.orig_h, header.orig_w  # whole image, one region
    else:
        step_h = step_w = header.tile_size
    make_masks = getattr(predictor, "stage_masks", stage_masks)
    masks = make_masks((step_h, step_w), header.levels, header.anchor_stride,
                       header.anchor_block)
    ebs = stage_ebs((step_h, step_w), header.levels, header.anchor_stride,
                    header.anchor_block, header.eb, header.eb_ratio)
    canvas = np.empty((header.channels, ty * step_h, tx * step_w), np.float32)
    for idx, payload in enumerate(payloads):
        i, j = divmod(idx, tx)
        canvas[:, i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w] = \
            _decompress_tile(payload, masks, ebs, header, predictor)
    return _finalize(canvas, header)


def _compress_tile(tile, masks, ebs, predictor, radius, round_output, stats):
    c = tile.shape[0]
    recon = np.zeros_like(tile)
    known = np.zeros(tile.shape[1:], bool)
    parts = []
    for stage_idx, pos in enumerate(masks):
        n = int(pos.sum())
        if n == 0:
            parts.append(pack_stage(np.zeros(0, np.uint32), np.zeros(0, np.float32)))
            continue
        eb = ebs[stage_idx]
        if stage_idx == 0:
            pred = np.zeros((c, n), np.float32)
        else:
            t0 = time.time()
            pred = predictor.predict(recon, known, pos)
            stats["predict_s"] += time.time() - t0
        t0 = time.time()
        codes, outliers = quantize(tile[:, pos], pred, eb, radius,
                                   round_output=round_output)
        recon[:, pos] = dequantize(pred, codes, outliers, eb, radius).reshape(c, n)
        x_stage = tile[:, pos].astype(np.float64)
        pred_err = x_stage - pred.astype(np.float64)
        recon_err = x_stage - recon[:, pos].astype(np.float64)
        stats["stage_pred_sae"][stage_idx] += float(np.abs(pred_err).sum())
        stats["stage_pred_sse"][stage_idx] += float(np.square(pred_err).sum())
        stats["stage_recon_sae"][stage_idx] += float(np.abs(recon_err).sum())
        stats["stage_recon_sse"][stage_idx] += float(np.square(recon_err).sum())
        stats["stage_recon_max"][stage_idx] = max(
            stats["stage_recon_max"][stage_idx],
            float(np.abs(recon_err).max()))
        stats["quantize_s"] += time.time() - t0
        known |= pos
        stats["outliers"] += len(outliers)
        stats["stage_outliers"][stage_idx] += len(outliers)
        stats["stage_codes"][stage_idx] += n * c
        t0 = time.time()
        part = pack_stage(codes, outliers)
        stats["stage_payload_bytes"][stage_idx] += len(part)
        parts.append(part)
        stats["entropy_s"] += time.time() - t0
    return b"".join(parts), recon


def _decompress_tile(payload, masks, ebs, header, predictor):
    c = header.channels
    th, tw = masks[0].shape  # region dims (tile or whole image)
    recon = np.zeros((c, th, tw), np.float32)
    known = np.zeros((th, tw), bool)
    off = 0
    for stage_idx, pos in enumerate(masks):
        codes, outliers, off = unpack_stage(payload, off)
        n = int(pos.sum())
        if n == 0:
            continue
        if stage_idx == 0:
            pred = np.zeros((c, n), np.float32)
        else:
            pred = predictor.predict(recon, known, pos)
        recon[:, pos] = dequantize(pred, codes, outliers, ebs[stage_idx],
                                   header.radius).reshape(c, n)
        known |= pos
    return recon


def _finalize(canvas: np.ndarray, header: Header) -> np.ndarray:
    out = canvas.transpose(1, 2, 0)[:header.orig_h, :header.orig_w]
    dtype = DTYPE_CODES[header.src_dtype]
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    out = out.astype(dtype)
    if header.channels == 1:
        out = out[..., 0]
    return out
