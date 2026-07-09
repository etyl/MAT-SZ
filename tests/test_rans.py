import numpy as np
import pytest

from deepsz.rans import (build_laplace_tables, model_bits, rans_decode,
                         rans_encode, scale_to_level)


def _roundtrip(codes, levels, tables):
    blob = rans_encode(codes, levels, tables)
    got = rans_decode(blob, levels, tables)
    assert np.array_equal(got, np.asarray(codes, np.uint32).ravel())
    return blob


def test_random_codes_random_scale_levels_roundtrip():
    rng = np.random.RandomState(0)
    tables = build_laplace_tables(0.01, radius=32, precision=14)
    levels = rng.randint(0, 64, 4096).astype(np.uint8)
    codes = rng.randint(0, 64, len(levels)).astype(np.uint32)

    blob = _roundtrip(codes, levels, tables)

    assert len(blob) * 8 <= model_bits(codes, levels, tables) * 1.02 + 64


def test_skewed_distribution_tracks_model_bits():
    rng = np.random.RandomState(1)
    eb = 0.01
    radius = 64
    tables = build_laplace_tables(eb, radius=radius, precision=15)
    levels = np.full(20000, scale_to_level(np.asarray([eb]), eb)[0], np.uint8)
    residual = rng.laplace(0.0, eb, len(levels))
    q = np.rint(residual / (2 * eb)).astype(np.int64)
    codes = np.clip(q + radius, 1, 2 * radius - 1).astype(np.uint32)

    blob = _roundtrip(codes, levels, tables)

    assert abs(len(blob) * 8 - model_bits(codes, levels, tables)) / len(blob) / 8 < 0.01


def test_empty_stage_roundtrip():
    tables = build_laplace_tables(0.01, radius=8, precision=10)

    blob = _roundtrip(np.zeros(0, np.uint32), np.zeros(0, np.uint8), tables)

    assert blob == b""


def test_alphabet_edge_outlier_code_roundtrip():
    tables = build_laplace_tables(0.01, radius=16, precision=12)
    codes = np.asarray([0, 1, 31, 0, 16, 31], np.uint32)
    levels = np.asarray([0, 4, 12, 63, 32, 1], np.uint8)

    _roundtrip(codes, levels, tables)


def test_rejects_invalid_scale_level():
    tables = build_laplace_tables(0.01, radius=8, n_levels=4, precision=10)
    levels = np.asarray([4], np.uint8)

    with pytest.raises(ValueError, match="scale level"):
        rans_encode(np.asarray([1], np.uint32), levels, tables)
    with pytest.raises(ValueError, match="scale level"):
        rans_decode(b"\0\0\0\0", levels, tables)


def test_rejects_unaligned_payload():
    tables = build_laplace_tables(0.01, radius=8, precision=10)

    with pytest.raises(ValueError, match="aligned"):
        rans_decode(b"\0", np.asarray([0], np.uint8), tables)
