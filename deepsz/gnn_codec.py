"""Checkpoint-backed, tensor-shaped GNN compressor codec."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import zstandard

from .codec import _compress_tile
from .gnn_predictor import GNNPredictor
from .levels import stage_ebs, stage_masks
from .quantizer import dequantize
from .bitstream import unpack_stage
from .rans import build_laplace_tables, scale_to_level


_MAGIC = b"MATSZGNN"
_VERSION = 2
_PREFIX = "<8sII"
_PREFIX_SIZE = struct.calcsize(_PREFIX)


def _as_numpy(x: Any) -> np.ndarray:
    """Accept numpy arrays and torch tensors without importing torch eagerly."""
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    cpu = getattr(x, "cpu", None)
    numpy = getattr(x, "numpy", None)
    if detach is not None and cpu is not None:
        return x.detach().cpu().numpy()
    if numpy is not None:
        return x.numpy()
    return np.asarray(x)


def _dtype_meta(dtype: np.dtype) -> dict[str, Any]:
    dtype = np.dtype(dtype)
    return {
        "str": dtype.str,
        "kind": dtype.kind,
        "itemsize": dtype.itemsize,
    }


def _restore_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    elif dtype.kind == "b":
        values = values >= 0.5
    return values.astype(dtype, copy=False)


def _write_stream(meta: dict[str, Any], payload: bytes, zstd_level: int) -> bytes:
    header = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = zstandard.ZstdCompressor(level=zstd_level).compress(payload)
    return struct.pack(_PREFIX, _MAGIC, _VERSION, len(header)) + header + body


def _read_stream(stream: bytes) -> tuple[dict[str, Any], bytes]:
    if len(stream) < _PREFIX_SIZE:
        raise ValueError("not a DeepSZ GNN stream")
    magic, version, header_len = struct.unpack_from(_PREFIX, stream, 0)
    if magic != _MAGIC:
        raise ValueError(f"not a DeepSZ GNN stream (bad magic {magic!r})")
    if version != _VERSION:
        raise ValueError(f"unsupported DeepSZ GNN stream version {version}")
    off = _PREFIX_SIZE
    meta = json.loads(stream[off:off + header_len].decode("utf-8"))
    payload = zstandard.ZstdDecompressor().decompress(stream[off + header_len:])
    return meta, payload


def _empty_stats(n_stages: int) -> dict[str, Any]:
    return {
        "predict_s": 0.0,
        "quantize_s": 0.0,
        "entropy_s": 0.0,
        "outliers": 0,
        "stage_codes": [0] * n_stages,
        "stage_outliers": [0] * n_stages,
        "stage_payload_bytes": [0] * n_stages,
        "stage_model_bits": [0.0] * n_stages,
        "stage_pred_sae": [0.0] * n_stages,
        "stage_pred_sse": [0.0] * n_stages,
        "stage_recon_sae": [0.0] * n_stages,
        "stage_recon_sse": [0.0] * n_stages,
        "stage_recon_max": [0.0] * n_stages,
    }


def _decompress_region(
    payload: bytes,
    shape: tuple[int, ...],
    masks: list[np.ndarray],
    ebs: list[float],
    radius: int,
    predictor: GNNPredictor,
    use_rans: bool,
) -> np.ndarray:
    recon = np.zeros((1, *shape), np.float32)
    known = np.zeros(shape, bool)
    off = 0
    for stage_idx, pos in enumerate(masks):
        n = int(pos.sum())
        if n == 0:
            if use_rans:
                tables = build_laplace_tables(ebs[stage_idx], radius)
                codes, outliers, off = unpack_stage(
                    payload, off, rans_levels=np.zeros(0, np.uint8),
                    rans_tables=tables)
            else:
                codes, outliers, off = unpack_stage(payload, off)
            continue
        if stage_idx == 0:
            pred = np.zeros((1, n), np.float32)
            scale = np.full((1, n), ebs[stage_idx], np.float32)
        else:
            if use_rans:
                pred, scale = predictor.predict(recon, known, pos,
                                                eb=ebs[stage_idx])
            else:
                got = predictor.predict(recon, known, pos, eb=ebs[stage_idx])
                pred = got[0] if isinstance(got, tuple) else got
                scale = None
        if use_rans:
            tables = build_laplace_tables(ebs[stage_idx], radius)
            levels64 = scale_to_level(scale, ebs[stage_idx]).reshape(-1)
            codes, outliers, off = unpack_stage(
                payload, off, rans_levels=levels64, rans_tables=tables)
        else:
            codes, outliers, off = unpack_stage(payload, off)
        recon[:, pos] = dequantize(pred, codes, outliers, ebs[stage_idx],
                                   radius).reshape(1, n)
        known |= pos
    if off != len(payload):
        raise ValueError("trailing bytes in DeepSZ GNN payload")
    return recon[0]


class GNNCompressorCodec:
    """Usable Python codec for GNN-backed DeepSZ tensor compression.

    The codec is initialized from a GNN checkpoint path. ``compress`` accepts a
    numpy array or torch tensor of any rank and returns bytes. ``uncompress``
    accepts those bytes and returns a torch tensor with the original shape and
    dtype.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        error_bound: float = 1e-3,
        *,
        levels: int = 4,
        anchor_stride: int = 16,
        anchor_block: int = 1,
        radius: int = 1 << 15,
        max_radius: int = 64,
        device: str = "cpu",
        zstd_level: int = 9,
        eb_ratio: float | None = 1.0,
        tune: str = "fast",
        strict_checkpoint: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GNN checkpoint not found: {self.checkpoint_path}")
        if error_bound <= 0:
            raise ValueError("error_bound must be > 0")
        if tune not in ("fast", "size"):
            raise ValueError("tune must be 'fast' or 'size'")

        self.error_bound = float(error_bound)
        self.levels = int(levels)
        self.anchor_stride = int(anchor_stride)
        self.anchor_block = int(anchor_block)
        self.radius = int(radius)
        self.max_radius = int(max_radius)
        self.device = device
        self.zstd_level = int(zstd_level)
        self.eb_ratio = eb_ratio
        self.tune = tune
        self.strict_checkpoint = bool(strict_checkpoint)
        self.checkpoint_hash = self._checkpoint_hash()

    def compress(self, x: Any, error_bound: float | None = None) -> bytes:
        """Compress a numpy array or torch tensor of any rank into bytes."""
        arr = np.asarray(_as_numpy(x))
        if arr.size == 0:
            raise ValueError("cannot compress an empty tensor")
        if arr.dtype.kind not in "biuf":
            raise TypeError(f"unsupported dtype {arr.dtype}; expected numeric data")

        dtype = np.dtype(arr.dtype)
        original_shape = tuple(int(n) for n in arr.shape)
        shape = original_shape if original_shape else (1,)
        values = arr.reshape(shape).astype(np.float32, copy=False)
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        eb = self.error_bound if error_bound is None else float(error_bound)
        if eb <= 0:
            raise ValueError("error_bound must be > 0")

        ratio_candidates = (
            [float(self.eb_ratio)] if self.eb_ratio is not None
            else ([1.0, 0.9, 0.8, 0.7] if self.tune == "size" else [1.0])
        )
        candidates: list[tuple[int, bytes]] = []
        for ratio in ratio_candidates:
            payload = self._compress_payload(values, dtype, eb, vmin, vmax, ratio)
            meta = {
                "codec": "deepsz.gnn",
                "shape": list(original_shape),
                "coded_shape": list(shape),
                "dtype": _dtype_meta(dtype),
                "error_bound": eb,
                "levels": self.levels,
                "anchor_stride": self.anchor_stride,
                "anchor_block": self.anchor_block,
                "radius": self.radius,
                "max_radius": self.max_radius,
                "vmin": vmin,
                "vmax": vmax,
                "eb_ratio": ratio,
                "entropy_coder": "rans",
                "checkpoint_hash": self.checkpoint_hash.hex(),
            }
            stream = _write_stream(meta, payload, self.zstd_level)
            candidates.append((len(stream), stream))
        return min(candidates, key=lambda item: item[0])[1]

    def uncompress(self, stream: bytes | bytearray | memoryview):
        """Decompress bytes from ``compress`` and return a torch tensor."""
        import torch

        meta, payload = _read_stream(bytes(stream))
        if meta.get("codec") != "deepsz.gnn":
            raise ValueError("not a DeepSZ GNN tensor stream")
        got_hash = meta.get("checkpoint_hash")
        if self.strict_checkpoint and got_hash != self.checkpoint_hash.hex():
            raise ValueError("checkpoint hash differs from the stream metadata")

        shape = tuple(int(n) for n in meta["coded_shape"])
        original_shape = tuple(int(n) for n in meta["shape"])
        dtype = np.dtype(meta["dtype"]["str"])
        vmin = float(meta["vmin"])
        vmax = float(meta["vmax"])
        if vmax <= vmin:
            vmax = vmin + 1.0

        predictor = self._predictor(vmin, vmax, meta)
        masks = stage_masks(shape, int(meta["levels"]), int(meta["anchor_stride"]),
                            int(meta["anchor_block"]))
        ebs = stage_ebs(shape, int(meta["levels"]), int(meta["anchor_stride"]),
                        int(meta["anchor_block"]), float(meta["error_bound"]),
                        float(meta["eb_ratio"]))
        use_rans = meta.get("entropy_coder", "huffman") == "rans"
        values = _decompress_region(payload, shape, masks, ebs, int(meta["radius"]),
                                    predictor, use_rans)
        out = _restore_dtype(values.reshape(original_shape), dtype)
        return torch.as_tensor(out)

    decompress = uncompress

    def _compress_payload(
        self,
        values: np.ndarray,
        dtype: np.dtype,
        eb: float,
        vmin: float,
        vmax: float,
        eb_ratio: float,
    ) -> bytes:
        predictor = self._predictor(vmin, vmax)
        masks = stage_masks(values.shape, self.levels, self.anchor_stride,
                            self.anchor_block)
        ebs = stage_ebs(values.shape, self.levels, self.anchor_stride,
                        self.anchor_block, eb, eb_ratio)
        stats = _empty_stats(len(masks))
        payload, _ = _compress_tile(values[None, ...], masks, ebs, predictor,
                                    self.radius, dtype.kind in "bi", stats)
        return payload

    def _predictor(
        self,
        vmin: float,
        vmax: float,
        meta: dict[str, Any] | None = None,
    ) -> GNNPredictor:
        levels = self.levels if meta is None else int(meta["levels"])
        anchor_stride = self.anchor_stride if meta is None else int(meta["anchor_stride"])
        anchor_block = self.anchor_block if meta is None else int(meta["anchor_block"])
        max_radius = self.max_radius if meta is None else int(meta["max_radius"])
        return GNNPredictor(
            self.checkpoint_path,
            vmin,
            vmax,
            tile_size=0,
            max_radius=max_radius,
            device=self.device,
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=anchor_block,
        )

    def _checkpoint_hash(self) -> bytes:
        import hashlib

        return hashlib.sha256(self.checkpoint_path.read_bytes()).digest()[:16]


GNNCodec = GNNCompressorCodec
