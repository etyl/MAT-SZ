"""DeepSZ container format.

File = fixed struct-packed little-endian header, then a u32 per-tile payload
size table, then one zstd frame holding the concatenated tile payloads.

Tile payload (produced by codec, opaque here) = per stage:
  [n_codes u32][huffman blob len u64][huffman blob][n_outliers u32][outliers f32...]
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np
import zstandard

MAGIC = b"MATSZ01\0"
VERSION = 3

FLAG_MOCK = 1 << 0
FLAG_GRAY = 1 << 1
FLAG_GNN = 1 << 2
FLAG_INTERP = 1 << 3       # SZ-style interpolation baseline (torch-free)
FLAG_CUBIC = 1 << 4        # interp order: set = cubic, clear = linear
FLAG_NOTILE = 1 << 5       # whole image is one tile (no padding, no seam)

_HEADER_FMT = "<8sHHIIBBdBBBBHIQdd16sHHd"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

DTYPE_CODES = {0: np.uint8, 1: np.float32}
DTYPE_IDS = {np.dtype(np.uint8): 0, np.dtype(np.float32): 1}


@dataclass
class Header:
    orig_h: int
    orig_w: int
    channels: int
    src_dtype: int  # key into DTYPE_CODES
    eb: float  # absolute error bound, original data units
    levels: int
    anchor_stride: int
    anchor_block: int
    tile_size: int
    radius: int
    seed: int
    vmin: float
    vmax: float
    ckpt_hash: bytes = b"\0" * 16  # sha256 prefix; zeros for mock predictor
    n_tiles_y: int = 1
    n_tiles_x: int = 1
    flags: int = 0
    interp_center: int = 0  # interp multi-axis mode: 0=avg both, 1=axis0, 2=axis1
    eb_ratio: float = 1.0   # per-level error-bound decay (coarse tighter); 1=flat
    version: int = VERSION

    def pack(self) -> bytes:
        return struct.pack(
            _HEADER_FMT, MAGIC, self.version, self.flags,
            self.orig_h, self.orig_w, self.channels, self.src_dtype,
            self.eb, self.levels, self.anchor_stride, self.anchor_block,
            self.interp_center,
            self.tile_size, self.radius, self.seed, self.vmin, self.vmax,
            self.ckpt_hash, self.n_tiles_y, self.n_tiles_x, self.eb_ratio,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "Header":
        (magic, version, flags, orig_h, orig_w, channels, src_dtype, eb,
         levels, anchor_stride, anchor_block, interp_center, tile_size, radius,
         seed, vmin, vmax, ckpt_hash, n_tiles_y, n_tiles_x, eb_ratio
         ) = struct.unpack_from(_HEADER_FMT, buf, 0)
        if magic != MAGIC:
            raise ValueError(f"not a DeepSZ stream (bad magic {magic!r})")
        if version != VERSION:
            raise ValueError(f"unsupported version {version}")
        return cls(orig_h=orig_h, orig_w=orig_w, channels=channels,
                   src_dtype=src_dtype, eb=eb, levels=levels,
                   anchor_stride=anchor_stride, anchor_block=anchor_block,
                   tile_size=tile_size, radius=radius, seed=seed,
                   vmin=vmin, vmax=vmax, ckpt_hash=ckpt_hash,
                   n_tiles_y=n_tiles_y, n_tiles_x=n_tiles_x,
                   flags=flags, interp_center=interp_center,
                   eb_ratio=eb_ratio, version=version)


def write_stream(header: Header, tile_payloads: list[bytes], zstd_level: int = 9) -> bytes:
    n = header.n_tiles_y * header.n_tiles_x
    if len(tile_payloads) != n:
        raise ValueError(f"expected {n} tile payloads, got {len(tile_payloads)}")
    sizes = struct.pack(f"<{n}Q", *(len(p) for p in tile_payloads))
    body = zstandard.ZstdCompressor(level=zstd_level).compress(b"".join(tile_payloads))
    return header.pack() + sizes + body


def read_stream(data: bytes) -> tuple[Header, list[bytes]]:
    header = Header.unpack(data)
    n = header.n_tiles_y * header.n_tiles_x
    off = _HEADER_SIZE
    sizes = struct.unpack_from(f"<{n}Q", data, off)
    off += 8 * n
    body = zstandard.ZstdDecompressor().decompress(data[off:])
    payloads = []
    pos = 0
    for s in sizes:
        payloads.append(body[pos:pos + s])
        pos += s
    if pos != len(body):
        raise ValueError("tile payload sizes do not match body length")
    return header, payloads


# ---- stage-record helpers used by codec ----

def pack_stage(codes: np.ndarray, outliers: np.ndarray) -> bytes:
    blob_codes = np.asarray(codes, np.uint32)
    out = np.asarray(outliers, np.float32)
    from .huffman import huffman_encode
    hblob = huffman_encode(blob_codes)
    return (struct.pack("<IQ", len(blob_codes), len(hblob)) + hblob
            + struct.pack("<I", len(out)) + out.tobytes())


def unpack_stage(buf: bytes, off: int) -> tuple[np.ndarray, np.ndarray, int]:
    n_codes, hlen = struct.unpack_from("<IQ", buf, off)
    off += 12
    from .huffman import huffman_decode
    codes = huffman_decode(buf[off:off + hlen])
    if len(codes) != n_codes:
        raise ValueError("stage code count mismatch")
    off += hlen
    (n_out,) = struct.unpack_from("<I", buf, off)
    off += 4
    outliers = np.frombuffer(buf, np.float32, count=n_out, offset=off).copy()
    off += 4 * n_out
    return codes, outliers, off
