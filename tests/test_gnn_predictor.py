"""Tests for the dimension-agnostic GNN predictor (untrained-net smoke tests)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matsz.gnn_predictor import (build_model, build_stage_geoms,
                                 half_directions, stage_forward)
from matsz.gnn_predictor import _LegacyGeom
from matsz.levels import stage_masks


@pytest.mark.parametrize("shape,levels,stride,block,max_radius", [
    ((64, 64), 4, 16, 4, 64),
    ((64, 64), 4, 16, 1, 8),      # small radius exercises the `<= limit` gate
    ((70, 90), 3, 8, 1, 64),      # non-multiple-of-stride shape (edge clamping)
    ((16, 16, 16), 2, 4, 1, 16),  # 3-D
])
def test_geometry_matches_scan(shape, levels, stride, block, max_radius):
    """The load-bearing test: the closed-form lattice geometry must reproduce,
    bit for bit, the old step-by-step neighbour scan (`_LegacyGeom`) it replaced
    — per stage, per half-direction, for (idx, dist, valid) on both sides."""
    masks = stage_masks(shape, levels, stride, block)
    geoms, _ = build_stage_geoms(shape, levels, stride, block, max_radius, torch)
    cum = np.zeros(shape, bool)  # known before the current stage
    gi = 0
    for mask in masks:
        if mask.any():
            g = geoms[gi]
            qidx = np.ravel_multi_index(np.nonzero(mask), shape)
            ref = _LegacyGeom(cum, max_radius, torch, query_idx=qidx)
            assert np.array_equal(g.query_idx.numpy(), qidx)
            for gl, rl in zip(g.lines, ref.lines):
                for k in ("ip", "in", "dp", "dn", "vp", "vn"):
                    assert np.array_equal(gl[k].numpy(), rl[k].numpy()), (k, gi)
            gi += 1
        cum |= mask
    assert gi == len(geoms)


def test_half_directions_count():
    # one representative per line = (3^n - 1) / 2
    assert len(half_directions(2)) == 4
    assert len(half_directions(3)) == 13


def _run(model, recon, known, max_radius=64):
    """Run the propagating field once: seed the `known` points as anchors,
    then predict every point (mirrors the codec's first predict call)."""
    c = recon.shape[0]
    N = known.size
    x = torch.from_numpy(recon.reshape(c, -1).astype(np.float32))
    E = torch.zeros(c, N, model.d)
    prev = np.zeros(known.shape, bool)
    with torch.no_grad():
        values, _ = stage_forward(model, E, prev, known, x, max_radius, torch)
    return values.numpy().reshape(recon.shape)


def test_2d_smoke_and_determinism():
    model = build_model(d=16).eval()
    masks = stage_masks((64, 64), 4, 16, anchor_block=4)  # levels==log2(stride)
    known = masks[0]  # anchors known, predict everything else
    recon = np.random.RandomState(0).rand(2, 64, 64).astype(np.float32)
    recon = recon * known[None]  # only known positions meaningful
    out = _run(model, recon, known)
    assert out.shape == (2, 64, 64)
    assert np.isfinite(out).all()
    assert np.array_equal(out, _run(model, recon, known))  # deterministic


def test_3d_dimension_agnostic():
    """Same weights (trained on 2-D) evaluate on a 3-D grid -> proves the
    network is dimension-generic."""
    model = build_model(d=16).eval()
    known = np.zeros((16, 16, 16), bool)
    known[::4, ::4, ::4] = True  # sparse anchors
    recon = np.random.RandomState(1).rand(1, 16, 16, 16).astype(np.float32)
    recon = recon * known[None]
    out = _run(model, recon, known, max_radius=16)
    assert out.shape == (1, 16, 16, 16)
    assert np.isfinite(out).all()


def test_no_neighbour_is_finite():
    """A totally empty known mask still yields finite output (null token)."""
    model = build_model(d=16).eval()
    known = np.zeros((8, 8), bool)
    recon = np.zeros((1, 8, 8), np.float32)
    out = _run(model, recon, known)
    assert np.isfinite(out).all()


def test_legacy_stage_forward_accepts_predict_idx():
    """Older trainer/eval code passes a compact prediction index; keep that
    call form working while the codec uses precomputed stage geometry."""
    model = build_model(d=16).eval()
    known = np.zeros((16, 16), bool)
    known[::4, ::4] = True
    prev = np.zeros_like(known)
    pos = np.zeros_like(known)
    pos[2::4, ::4] = True
    idx = torch.from_numpy(np.nonzero(pos.reshape(-1))[0])
    x = torch.from_numpy(np.random.RandomState(2).rand(1, known.size).astype(np.float32))
    E = torch.zeros(1, known.size, model.d)

    with torch.no_grad():
        values, E2 = stage_forward(model, E, prev, known, x, 16, torch,
                                   predict_idx=idx)

    assert values.shape == (1, int(pos.sum()))
    assert E2.shape == E.shape
    assert np.isfinite(values.numpy()).all()
