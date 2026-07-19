"""Private test doubles shared by codec integration tests."""

import numpy as np


class NearestPredictor:
    """Deterministic, torch-free predictor used only by fast tests."""

    stream_flag = 0
    checkpoint_hash = b"\0" * 16

    def predict(
        self,
        recon: np.ndarray,
        known: np.ndarray,
        pos: np.ndarray | None = None,
    ) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt, uniform_filter

        if not known.any():
            out = np.zeros_like(recon)
            return out if pos is None else out[:, pos]
        _, (ii, jj) = distance_transform_edt(~known, return_indices=True)
        filled = recon[:, ii, jj]
        smooth = uniform_filter(filled, size=(1, 3, 3), mode="nearest")
        out = np.where(known[None], filled, smooth).astype(np.float32)
        return out if pos is None else out[:, pos]
