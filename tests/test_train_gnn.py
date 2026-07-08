from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts.train_gnn import sample_noise


def test_noise_range_samples_log_uniformly():
    torch.manual_seed(0)
    args = SimpleNamespace(noise=0.01, noise_range=(1e-3, 1.0))

    noise = sample_noise(8192, args, "cpu")
    log_noise = noise.log()

    assert float(noise.min()) >= args.noise_range[0]
    assert float(noise.max()) <= args.noise_range[1]
    midpoint = (torch.log(torch.tensor(args.noise_range[0]))
                + torch.log(torch.tensor(args.noise_range[1]))) / 2
    assert abs(float(log_noise.mean() - midpoint)) < 0.08


def test_noise_without_range_stays_fixed():
    args = SimpleNamespace(noise=0.02, noise_range=None)

    noise = sample_noise(4, args, "cpu")

    assert torch.equal(noise, torch.full((4,), args.noise))
