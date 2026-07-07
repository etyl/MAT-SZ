"""MAT-SZ: SZ-style error-bounded lossy compression with a pluggable predictor
(SZ3-style multilevel interpolation by default, or a trained GNN)."""

from .codec import compress, decompress

__all__ = ["compress", "decompress"]
__version__ = "0.1.0"
