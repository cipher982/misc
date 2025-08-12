"""
Schema definitions for Transformer Intuition Lab.

This module provides enhanced configuration schemas with:
- Cross-field validation and constraint checking
- Automatic parameter/memory/FLOPs estimation
- Auto-repair suggestions for invalid configurations
- Modern transformer architecture support (GQA, different activations, etc.)
"""

from .model_config import ModelConfig

__all__ = ["ModelConfig"]