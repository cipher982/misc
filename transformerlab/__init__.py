"""
Transformer Intuition Lab - Interactive playground for understanding transformer architectures.
"""

__version__ = "0.1.0"
__author__ = "Transformer Intuition Lab"
__description__ = "Interactive playground for understanding transformer architectures"

from .core.tokenizer import CharTokenizer, load_corpus
from .core.transformer import Transformer

__all__ = ["Transformer", "CharTokenizer", "load_corpus"]
