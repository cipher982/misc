"""
Transformer Intuition Lab - Interactive playground for understanding transformer architectures.
"""

__version__ = "0.1.0"
__author__ = "Transformer Intuition Lab"
__description__ = "Interactive playground for understanding transformer architectures"

from .core.transformer import Transformer
from .core.tokenizer import CharTokenizer, load_corpus

__all__ = [
    "Transformer",
    "CharTokenizer", 
    "load_corpus"
]