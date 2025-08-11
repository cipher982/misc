"""
Backend abstractions for different transformer implementations.

This module provides abstract base classes that define the interface
for transformer components across different backends (NumPy, PyTorch, Pure Python).
"""

from .abstract import (
    AbstractTransformer,
    AbstractTransformerBlock,
    AbstractAttention,
    AbstractFeedForward,
    AbstractNormalization,
    AbstractOptimizer,
    BackendConfig,
)

from .factory import create_transformer, list_backends, get_backend_info

__all__ = [
    "AbstractTransformer",
    "AbstractTransformerBlock", 
    "AbstractAttention",
    "AbstractFeedForward",
    "AbstractNormalization",
    "AbstractOptimizer",
    "BackendConfig",
    "create_transformer",
    "list_backends",
    "get_backend_info",
]