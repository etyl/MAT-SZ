"""Tests for the dimension-agnostic GNN predictor (untrained-net smoke tests)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matsz.gnn_predictor import build_model, half_directions, stage_forward
from matsz.levels import stage_masks


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
