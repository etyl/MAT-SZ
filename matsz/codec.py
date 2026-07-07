"""Closed-loop MAT-SZ codec: tiling, progressive prediction, quantization.

The encoder simulates the decoder: reconstructions fed back into the predictor
are built exclusively from dequantize() outputs, never from the original data,
so decoder-side predictions match encoder-side predictions bitwise (same
platform) and the error bound holds end to end.
"""

from __future__ import annotations

import time

import numpy as np

from .bitstream import (DTYPE_CODES, DTYPE_IDS, FLAG_GRAY, FLAG_MOCK, Header,
                        pack_stage, read_stream, unpack_stage, write_stream)
from .levels import stage_masks
from .predictor import MockPredictor
from .quantizer import dequantize, quantize


def compress(
    img: np.ndarray,
    eb: float,
    predictor,
    levels: int = 4,
    anchor_stride: int = 16,
    anchor_block: int = 4,
    radius: int = 1 << 15,
    seed: int = 1234,
    zstd_level: int = 9,
    verbose: bool = False,
) -> tuple[bytes, dict]:
    """Compress an (H, W) or (H, W, C) array. Returns (stream bytes, stats)."""
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

    t = predictor.tile_size
    ty, tx = -(-h // t), -(-w // t)
    canvas = np.pad(fimg, ((0, ty * t - h), (0, tx * t - w), (0, 0)), mode="edge")
    canvas = canvas.transpose(2, 0, 1)  # (C, H, W)

    masks = stage_masks(t, t, levels, anchor_stride, anchor_block)
    flags = getattr(predictor, "stream_flag", 0) | (FLAG_GRAY if c == 1 else 0)
    round_output = np.issubdtype(np.dtype(img.dtype), np.integer)

    stats = {"predict_s": 0.0, "quantize_s": 0.0, "entropy_s": 0.0,
             "outliers": 0, "stage_codes": [0] * len(masks)}
    payloads = []
    recon_canvas = np.empty_like(canvas)
    for i in range(ty):
        for j in range(tx):
            tile = canvas[:, i * t:(i + 1) * t, j * t:(j + 1) * t]
            payload, recon = _compress_tile(tile, masks, eb, predictor, radius,
                                            round_output, stats)
            recon_canvas[:, i * t:(i + 1) * t, j * t:(j + 1) * t] = recon
            payloads.append(payload)
            if verbose:
                print(f"tile ({i},{j}): {len(payload)} bytes raw payload")

    header = Header(orig_h=h, orig_w=w, channels=c, src_dtype=src_dtype,
                    eb=float(eb), levels=levels, anchor_stride=anchor_stride,
                    anchor_block=anchor_block, tile_size=t, radius=radius,
                    seed=seed, vmin=vmin, vmax=vmax,
                    ckpt_hash=getattr(predictor, "checkpoint_hash", b"\0" * 16),
                    n_tiles_y=ty, n_tiles_x=tx, flags=flags)
    t0 = time.time()
    stream = write_stream(header, payloads, zstd_level)
    stats["entropy_s"] += time.time() - t0
    stats["recon"] = _finalize(recon_canvas, header)
    stats["compressed_bytes"] = len(stream)
    stats["original_bytes"] = img.nbytes
    stats["ratio"] = img.nbytes / len(stream)
    return stream, stats


def decompress(stream: bytes, predictor_factory=None) -> np.ndarray:
    """Decompress a MAT-SZ stream. ``predictor_factory(header) -> predictor``;
    defaults to MockPredictor for mock streams (real streams need a factory
    that builds a MATPredictor from the checkpoint path)."""
    header, payloads = read_stream(stream)
    if predictor_factory is None:
        if not header.flags & FLAG_MOCK:
            raise ValueError("stream was made with the real MAT predictor; "
                             "provide predictor_factory (checkpoint needed)")
        predictor_factory = lambda hdr: MockPredictor(hdr.tile_size)
    predictor = predictor_factory(header)

    t = header.tile_size
    ty, tx = header.n_tiles_y, header.n_tiles_x
    masks = stage_masks(t, t, header.levels, header.anchor_stride, header.anchor_block)
    canvas = np.empty((header.channels, ty * t, tx * t), np.float32)
    for idx, payload in enumerate(payloads):
        i, j = divmod(idx, tx)
        canvas[:, i * t:(i + 1) * t, j * t:(j + 1) * t] = _decompress_tile(
            payload, masks, header, predictor)
    return _finalize(canvas, header)


def _compress_tile(tile, masks, eb, predictor, radius, round_output, stats):
    c = tile.shape[0]
    recon = np.zeros_like(tile)
    known = np.zeros(tile.shape[1:], bool)
    parts = []
    for stage_idx, pos in enumerate(masks):
        n = int(pos.sum())
        if n == 0:
            parts.append(pack_stage(np.zeros(0, np.uint32), np.zeros(0, np.float32)))
            continue
        if stage_idx == 0:
            pred = np.zeros((c, n), np.float32)
        else:
            t0 = time.time()
            full = predictor.predict(recon, known)
            stats["predict_s"] += time.time() - t0
            pred = full[:, pos]
        t0 = time.time()
        codes, outliers = quantize(tile[:, pos], pred, eb, radius,
                                   round_output=round_output)
        recon[:, pos] = dequantize(pred, codes, outliers, eb, radius).reshape(c, n)
        stats["quantize_s"] += time.time() - t0
        known |= pos
        stats["outliers"] += len(outliers)
        stats["stage_codes"][stage_idx] += n * c
        t0 = time.time()
        parts.append(pack_stage(codes, outliers))
        stats["entropy_s"] += time.time() - t0
    return b"".join(parts), recon


def _decompress_tile(payload, masks, header, predictor):
    c = header.channels
    t = header.tile_size
    recon = np.zeros((c, t, t), np.float32)
    known = np.zeros((t, t), bool)
    off = 0
    for stage_idx, pos in enumerate(masks):
        codes, outliers, off = unpack_stage(payload, off)
        n = int(pos.sum())
        if n == 0:
            continue
        if stage_idx == 0:
            pred = np.zeros((c, n), np.float32)
        else:
            pred = predictor.predict(recon, known)[:, pos]
        recon[:, pos] = dequantize(pred, codes, outliers, header.eb,
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
