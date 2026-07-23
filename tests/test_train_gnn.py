from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_gnn import (
    discretized_laplace_nll,
    sample_noise,
    sample_synthetic_batch,
    mixed_batch_sizes,
    normalize_tensor,
    training_autocast,
    run_chunked_scene,
)
from deepsz.gnn_predictor import build_model


def test_noise_range_samples_log_uniformly():
    torch.manual_seed(0)
    args = SimpleNamespace(noise=0.01, noise_range=(1e-3, 1.0))

    noise = sample_noise(8192, args, "cpu")
    log_noise = noise.log()

    assert float(noise.min()) >= args.noise_range[0]
    assert float(noise.max()) <= args.noise_range[1]
    midpoint = (
        torch.log(torch.tensor(args.noise_range[0]))
        + torch.log(torch.tensor(args.noise_range[1]))
    ) / 2
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


def test_synthetic_batch_is_4d_normalized_and_reproducible():
    shape = (12, 10, 8, 6)
    correlation = (4.0, 2.5, 1.5, 0.75)

    a = sample_synthetic_batch(2, shape, correlation, np.random.default_rng(123))
    b = sample_synthetic_batch(2, shape, correlation, np.random.default_rng(123))

    assert a.shape == (2, np.prod(shape))
    assert a.dtype == torch.float32
    assert float(a.min()) >= 0.0
    assert float(a.max()) <= 1.0
    assert torch.equal(a, b)


def test_eval_tensor_normalization_maps_extrema_to_unit_interval():
    tensor = np.array([[-4.0, 1.0], [6.0, -1.5]], dtype=np.float64)

    normalized = normalize_tensor(tensor)

    assert normalized.dtype == np.float32
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
    np.testing.assert_allclose(normalized, (tensor + 4.0) / 10.0)


def test_eval_tensor_normalization_handles_constant_tensor():
    normalized = normalize_tensor(np.full((2, 3), 7.0, dtype=np.float32))

    np.testing.assert_array_equal(normalized, np.zeros((2, 3), np.float32))


def test_synthetic_axis_smoothness_follows_correlation_lengths():
    shape = (24, 20, 16, 12)
    correlation = (5.0, 3.0, 1.5, 0.6)
    fields = sample_synthetic_batch(
        3, shape, correlation, np.random.default_rng(7), randomize=False
    ).reshape(3, *shape)

    # Mean adjacent differences are smaller on axes with longer correlation.
    variation = []
    for axis in range(4):
        variation.append(float(torch.diff(fields, dim=axis + 1).abs().mean()))
    assert variation[0] < variation[1] < variation[2] < variation[3]


def test_synthetic_fields_randomly_permute_correlation_axes():
    shape = (16, 16, 16, 16)
    fields = sample_synthetic_batch(
        6, shape, (6.0, 3.0, 1.5, 0.5), np.random.default_rng(19)
    ).reshape(6, *shape)

    # The smoothest axis (smallest adjacent variation) must not remain fixed
    # across the batch when correlation assignments are shuffled per field.
    smoothest_axes = []
    for field in fields:
        variation = [
            float(torch.diff(field, dim=axis).abs().mean()) for axis in range(4)
        ]
        smoothest_axes.append(int(np.argmin(variation)))
    assert len(set(smoothest_axes)) > 1


def test_mixed_batch_fraction_counts_scalar_points():
    image_batch, synthetic_batch, actual = mixed_batch_sizes(
        crop=128,
        synthetic_shape=(16, 16, 16, 16),
        synthetic_fraction=0.25,
        image_batch=8,
        synthetic_batch=1,
    )

    assert image_batch == 12
    assert synthetic_batch == 1
    image_points = image_batch * 128**2
    synthetic_points = synthetic_batch * 16**4
    assert synthetic_points / (image_points + synthetic_points) == 0.25
    assert actual == 0.25


def test_mixed_batch_sizes_preserve_config_at_endpoints():
    assert mixed_batch_sizes(128, (16,) * 4, 0, 7, 2) == (7, 0, 0.0)
    assert mixed_batch_sizes(128, (16,) * 4, 1, 7, 2) == (0, 2, 1.0)


def test_fp16_autocast_is_cuda_only_on_cpu():
    x = torch.ones(2, dtype=torch.float32)
    with training_autocast(True, torch.device("cpu")):
        y = x * 2
    assert y.dtype == torch.float32


def test_chunk_finalization_casts_autocast_output_to_field_dtype():
    """Regression for CUDA FP16 finalization copied into the FP32 field.

    CPU BF16 autocast exercises the same mixed-dtype boundary without a GPU.
    """
    model = build_model(8)
    x = torch.rand(1, 8 * 8)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        nll, npix, _ = run_chunked_scene(
            model,
            x,
            (8, 8),
            axis=1,
            order=(0, 1),
            levels=2,
            stride=4,
            d=8,
            device=torch.device("cpu"),
            eb=0.01,
            agg_level=2,
        )
    loss = nll / npix
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
