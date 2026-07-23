import numpy as np

from scripts.eval_predictors import default_error_bounds, load_image


def test_eval_defaults_are_in_normalized_image_units():
    assert np.allclose(default_error_bounds(), [1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0])


def test_eval_loader_returns_normalized_float32(tmp_path):
    from PIL import Image

    src = np.array([[0, 64], [128, 255]], np.uint8)
    path = tmp_path / "tiny.png"
    Image.fromarray(src, mode="L").save(path)

    img = load_image(path)

    assert img.dtype == np.float32
    assert img.shape == src.shape
    assert img.min() == 0.0
    assert img.max() == 1.0
