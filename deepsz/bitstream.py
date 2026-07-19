"""DeepSZ container format.

File = fixed struct-packed little-endian header, then a u32 per-tile payload
size table, then one zstd frame holding the concatenated tile payloads.

Tile payload (produced by codec, opaque here) = per stage:
  [n_codes u32][entropy blob len u64][entropy blob][n_outliers u32][outliers f32...]

Interpolation streams use a compiled empirical ANS coder. Legacy and mock
streams use canonical Huffman; streams whose header sets FLAG_RANS use
scale-conditioned context coding over the same code array.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import zstandard

MAGIC = b"MATSZ01\0"
VERSION = 5

FLAG_MOCK = 1 << 0
FLAG_GRAY = 1 << 1
FLAG_GNN = 1 << 2
FLAG_INTERP = 1 << 3       # SZ-style interpolation baseline (torch-free)
FLAG_CUBIC = 1 << 4        # interp order: set = cubic, clear = linear
FLAG_NOTILE = 1 << 5       # whole image is one tile (no padding, no seam)
FLAG_RANS = 1 << 6         # per-symbol scale-conditioned coder for stage bins

_EMP_ANS = b"MATSANS1"

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
    # Original (unpadded) spatial shape, any rank. Written as a variable-length
    # block by write_stream (the fixed struct only has room for 2 axes). Empty
    # falls back to (orig_h, orig_w) for legacy 2-D callers.
    spatial: tuple[int, ...] = ()

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


def _pack_spatial(header: Header) -> bytes:
    spatial = header.spatial or (header.orig_h, header.orig_w)
    return struct.pack(f"<B{len(spatial)}I", len(spatial), *spatial)


def write_stream(header: Header, tile_payloads: list[bytes], zstd_level: int = 9) -> bytes:
    n = header.n_tiles_y * header.n_tiles_x
    if len(tile_payloads) != n:
        raise ValueError(f"expected {n} tile payloads, got {len(tile_payloads)}")
    sizes = struct.pack(f"<{n}Q", *(len(p) for p in tile_payloads))
    body = zstandard.ZstdCompressor(level=zstd_level).compress(b"".join(tile_payloads))
    return header.pack() + _pack_spatial(header) + sizes + body


def read_stream(data: bytes) -> tuple[Header, list[bytes]]:
    header = Header.unpack(data)
    off = _HEADER_SIZE
    (nd,) = struct.unpack_from("<B", data, off)
    off += 1
    header.spatial = struct.unpack_from(f"<{nd}I", data, off)
    off += 4 * nd
    n = header.n_tiles_y * header.n_tiles_x
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

def pack_stage(
    codes: np.ndarray,
    outliers: np.ndarray,
    *,
    rans_levels: np.ndarray | None = None,
    rans_tables=None,
    empirical_ans: bool = False,
) -> bytes:
    blob_codes = np.asarray(codes, np.uint32)
    out = np.asarray(outliers, np.float32)
    if empirical_ans:
        hblob = _empirical_ans_encode(blob_codes)
    elif rans_levels is None:
        from .huffman import huffman_encode
        hblob = huffman_encode(blob_codes)
    else:
        from .rans import rans_encode
        if rans_tables is None:
            raise ValueError("rans_tables are required with rans_levels")
        hblob = rans_encode(blob_codes, rans_levels, rans_tables)
    return (struct.pack("<IQ", len(blob_codes), len(hblob)) + hblob
            + struct.pack("<I", len(out)) + out.tobytes())


def unpack_stage(
    buf: bytes,
    off: int,
    *,
    rans_levels: np.ndarray | None = None,
    rans_tables=None,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_codes, hlen = struct.unpack_from("<IQ", buf, off)
    off += 12
    hblob = buf[off:off + hlen]
    if hblob.startswith(_EMP_ANS):
        codes = _empirical_ans_decode(hblob)
    elif rans_levels is None:
        from .huffman import huffman_decode
        codes = huffman_decode(hblob)
    else:
        from .rans import rans_decode
        if rans_tables is None:
            raise ValueError("rans_tables are required with rans_levels")
        codes = rans_decode(buf[off:off + hlen], rans_levels, rans_tables)
    if len(codes) != n_codes:
        raise ValueError("stage code count mismatch")
    off += hlen
    (n_out,) = struct.unpack_from("<I", buf, off)
    off += 4
    outliers = np.frombuffer(buf, np.float32, count=n_out, offset=off).copy()
    off += 4 * n_out
    return codes, outliers, off


def _empirical_ans_encode(codes: np.ndarray) -> bytes:
    """Encode bins with a compiled empirical ANS model.

    Only symbols that occur are placed in the model, which avoids serializing
    the quantizer's mostly empty 65k alphabet.  The outer zstd frame compresses
    the small symbol/frequency table further.
    """
    import constriction

    codes = np.asarray(codes, np.uint32).ravel()
    if len(codes):
        frequencies = np.bincount(codes)
        symbols = np.flatnonzero(frequencies).astype(np.uint32)
        counts = frequencies[symbols]
        ranks = np.empty(len(frequencies), np.int32)
        ranks[symbols] = np.arange(len(symbols), dtype=np.int32)
        inverse = ranks[codes]
    else:
        symbols = np.zeros(0, np.uint32)
        counts = np.zeros(0, np.int64)
        inverse = np.zeros(0, np.int32)
    head = _EMP_ANS + struct.pack("<QI", len(codes), len(symbols))
    table = (symbols.astype("<u4", copy=False).tobytes()
             + counts.astype("<u4", copy=False).tobytes())
    if len(symbols) <= 1:
        return head + table
    model = constriction.stream.model.Categorical(
        counts.astype(np.float32), perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(inverse.astype(np.int32, copy=False), model)
    words = encoder.get_compressed().astype("<u4", copy=False)
    return head + table + words.tobytes()


def _empirical_ans_decode(blob: bytes) -> np.ndarray:
    import constriction

    off = len(_EMP_ANS)
    n_codes, alphabet = struct.unpack_from("<QI", blob, off)
    off += 12
    symbols = np.frombuffer(blob, "<u4", alphabet, off)
    off += 4 * alphabet
    counts = np.frombuffer(blob, "<u4", alphabet, off)
    off += 4 * alphabet
    if alphabet == 0:
        if n_codes:
            raise ValueError("empty ANS alphabet for non-empty stage")
        return np.zeros(0, np.uint32)
    if alphabet == 1:
        return np.full(n_codes, symbols[0], np.uint32)
    words = np.frombuffer(blob, "<u4", offset=off)
    model = constriction.stream.model.Categorical(
        counts.astype(np.float32), perfect=False)
    decoder = constriction.stream.stack.AnsCoder(words)
    ranks = decoder.decode(model, n_codes)
    if not decoder.is_empty():
        raise ValueError("trailing data in empirical ANS payload")
    return symbols[ranks].astype(np.uint32, copy=False)
