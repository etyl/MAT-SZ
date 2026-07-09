from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts.train_gnn import discretized_laplace_nll, sample_noise


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


def test_discretized_laplace_nll_stays_finite_in_tails():
    mu = torch.tensor([[0.0, 1.0, 0.5, 0.999999]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.5001, 0.0]])
    log_b = torch.full_like(mu, -8.0, requires_grad=True)
    eb = torch.tensor([1e-4])

    nll = discretized_laplace_nll(mu, log_b, target, eb)
    loss = nll.mean()
    loss.backward()

    assert torch.isfinite(nll).all()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(log_b.grad).all()
