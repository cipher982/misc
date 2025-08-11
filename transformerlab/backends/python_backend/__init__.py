"""
Pure Python backend implementation.

This backend uses only built-in Python operations (lists, loops, basic math)
to provide maximum educational transparency. Every operation is explicit and
visible, making it ideal for understanding transformer algorithms step-by-step.
"""

from .transformer import PythonTransformer
from .attention import PythonAttention
from .feed_forward import PythonFeedForward
from .normalization import PythonLayerNorm
from .optimizer import PythonSGDOptimizer

__all__ = [
    "PythonTransformer",
    "PythonAttention", 
    "PythonFeedForward",
    "PythonLayerNorm",
    "PythonSGDOptimizer",
]