"""DeepSZ: SZ-style error-bounded lossy compression with a pluggable predictor
(SZ3-style multilevel interpolation by default, or a trained GNN)."""

from .codec import compress, decompress
from .gnn_codec import GNNCodec, GNNCompressorCodec
from .skel_codec import SkeletonGNNCodec

__all__ = ["compress", "decompress", "GNNCompressorCodec", "GNNCodec",
           "SkeletonGNNCodec"]
__version__ = "0.1.0"
