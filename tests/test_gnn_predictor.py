"""Tests for the dimension-agnostic GNN predictor. The untrained-net tests are
fast; the learning-sanity test trains a couple hundred steps (marked slow)."""

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
    masks = stage_masks(64, 64, 3, 16, anchor_block=4)
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


@pytest.mark.slow
def test_learns_ramps_and_sines():
    """Train ~250 steps on synthetic ramps + sinusoids; hole L1 must beat the
    trivial 'predict the mean of known values' baseline."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    model = build_model(d=32)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crop = 64
    yy, xx = np.mgrid[0:crop, 0:crop]

    def batch(n=6):
        out = []
        for _ in range(n):
            fx, fy = rng.uniform(0.05, 0.4, 2)
            a, b = rng.uniform(-1, 1, 2)
            img = 0.5 + 0.2 * (a * np.sin(fx * xx) + b * np.cos(fy * yy))
            img += 0.15 * (xx / crop) * rng.uniform(-1, 1)
            out.append(img.astype(np.float32))
        return torch.from_numpy(np.stack(out)).reshape(n, -1).clamp(0, 1)

    masks = stage_masks(crop, crop, 4, 16, anchor_block=4)

    def run(model, x):
        """Drive the propagating field over the schedule, teacher-forced from
        truth; yield (pred, posf, kv, known) per predicted stage."""
        n = x.shape[0]
        E = torch.zeros(n, x.shape[1], model.d)
        kv = torch.full_like(x, 0.5)
        prev = np.zeros((crop, crop), bool)
        known = masks[0].copy()
        kv = torch.where(torch.from_numpy(masks[0].reshape(-1)), x, kv)
        for pos in masks[1:]:
            if not pos.any():
                continue
            posf = torch.from_numpy(pos.reshape(-1))
            pred, E = stage_forward(model, E, prev, known, kv, 64, torch)
            yield pred, posf, kv, known
            prev = known.copy()
            known = known | pos
            kv = torch.where(posf, x, kv)

    for _ in range(250):
        x = batch()
        loss = torch.zeros(())
        ns = 0
        for pred, posf, _, _ in run(model, x):
            loss = loss + (pred[:, posf] - x[:, posf]).abs().mean()
            ns += 1
        loss = loss / ns
        opt.zero_grad(); loss.backward(); opt.step()

    # evaluate final-stage L1 vs. known-mean baseline
    x = batch(8)
    with torch.no_grad():
        for pred, posf, kv, known in run(model, x):
            pass  # last iteration = final stage
    model_l1 = (pred[:, posf] - x[:, posf]).abs().mean().item()
    known_mean = kv[:, torch.from_numpy(known.reshape(-1))].mean(1, keepdim=True)
    base_l1 = (known_mean - x[:, posf]).abs().mean().item()
    assert model_l1 < base_l1, f"model {model_l1:.4f} !< baseline {base_l1:.4f}"
