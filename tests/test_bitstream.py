import numpy as np
import pytest

from deepsz.bitstream import (Header, pack_stage, read_stream, unpack_stage,
                             write_stream)


def make_header(**kw):
    base = dict(orig_h=100, orig_w=130, channels=3, src_dtype=0, eb=2.0,
                levels=3, anchor_stride=8, anchor_block=4, tile_size=64,
                radius=1 << 15, seed=1234, vmin=0.0, vmax=255.0,
                ckpt_hash=bytes(range(16)), n_tiles_y=2, n_tiles_x=3, flags=1)
    base.update(kw)
    return Header(**base)


def test_header_roundtrip():
    h = make_header()
    h2 = Header.unpack(h.pack())
    assert h == h2


def test_header_roundtrip_gnn_execution_metadata():
    h = make_header(agg_level=1, gnn_prune_invalid=True)

    h2 = Header.unpack(h.pack())

    assert h2 == h


@pytest.mark.parametrize("agg_level", [0, 16])
def test_header_rejects_unrepresentable_agg_level(agg_level):
    with pytest.raises(ValueError, match="agg_level"):
        make_header(agg_level=agg_level).pack()


def test_stream_roundtrip():
    h = make_header()
    payloads = [bytes([i]) * (i * 100 + 1) for i in range(6)]
    stream = write_stream(h, payloads)
    h2, p2 = read_stream(stream)
    h.spatial = (h.orig_h, h.orig_w)  # write_stream records the spatial shape
    assert h2 == h
    assert p2 == payloads


def test_stream_roundtrip_nd():
    h = make_header(spatial=(12, 20, 24, 28), n_tiles_y=1, n_tiles_x=1)
    stream = write_stream(h, [b"x" * 33])
    h2, p2 = read_stream(stream)
    assert h2.spatial == (12, 20, 24, 28)
    assert p2 == [b"x" * 33]


def test_bad_magic():
    with pytest.raises(ValueError):
        Header.unpack(b"NOTMATSZ" + b"\0" * 100)


def test_payload_count_check():
    with pytest.raises(ValueError):
        write_stream(make_header(), [b"only-one"])


@pytest.mark.parametrize("empirical_ans", [False, True])
def test_stage_roundtrip(empirical_ans):
    rng = np.random.RandomState(0)
    codes = rng.randint(0, 1000, 500).astype(np.uint32)
    codes[::17] = 0
    outliers = rng.randn(int((codes == 0).sum())).astype(np.float32)
    buf = pack_stage(codes, outliers, empirical_ans=empirical_ans) + pack_stage(
        np.zeros(0, np.uint32), np.zeros(0, np.float32))
    c2, o2, off = unpack_stage(buf, 0)
    assert np.array_equal(c2, codes)
    assert np.array_equal(o2, outliers)
    c3, o3, off = unpack_stage(buf, off)
    assert len(c3) == 0 and len(o3) == 0 and off == len(buf)
