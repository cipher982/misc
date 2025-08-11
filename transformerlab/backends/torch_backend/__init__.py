"""
PyTorch backend implementation.

This backend uses PyTorch for production-ready performance with GPU acceleration,
automatic differentiation, and modern ML best practices. It demonstrates how
the same transformer architecture looks in a real-world framework.
"""

from .attention import TorchAttention
from .feed_forward import TorchFeedForward
from .normalization import TorchLayerNorm, TorchRMSNorm
from .optimizer import TorchAdamOptimizer, TorchSGDOptimizer
from .transformer import TorchTransformer

__all__ = [
    "TorchTransformer",
    "TorchAttention",
    "TorchFeedForward",
    "TorchLayerNorm",
    "TorchRMSNorm",
    "TorchSGDOptimizer",
    "TorchAdamOptimizer",
]
