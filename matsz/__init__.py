"""MAT-SZ: SZ-style error-bounded lossy compression with a Mask-Aware
Transformer (MAT, CVPR 2022) replacing the classic Lorenzo/spline predictor."""

from .codec import compress, decompress

__all__ = ["compress", "decompress"]
__version__ = "0.1.0"
