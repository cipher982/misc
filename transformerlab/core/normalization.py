"""
Normalization modules for the Transformer Intuition Lab.
Pure NumPy implementations for educational purposes.
"""

import numpy as np


def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """
    Layer Normalization implementation.

    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        gamma: Scale parameter of shape (hidden_dim,)
        beta: Shift parameter of shape (hidden_dim,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as x
    """
    # Calculate mean and variance along the last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Apply scale and shift
    return gamma * x_norm + beta


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    RMS Normalization implementation (simplified LayerNorm without mean centering).

    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        gamma: Scale parameter of shape (hidden_dim,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as x
    """
    # Calculate RMS (root mean square) along the last dimension
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

    # Normalize
    x_norm = x / rms

    # Apply scale
    return gamma * x_norm


class LayerNorm:
    """Layer Normalization module."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        self.hidden_dim = hidden_dim
        self.eps = eps

        # Initialize parameters
        self.gamma = np.ones(hidden_dim)
        self.beta = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return layer_norm(x, self.gamma, self.beta, self.eps)

    def get_stats(self, x: np.ndarray) -> dict:
        """Get statistics for visualization."""
        mean = np.mean(x, axis=-1)
        var = np.var(x, axis=-1)

        return {
            "input_mean": np.mean(mean),
            "input_var": np.mean(var),
            "output_mean": np.mean(self.forward(x)),
            "output_var": np.var(self.forward(x)),
            "gamma_mean": np.mean(self.gamma),
            "beta_mean": np.mean(self.beta),
        }


class RMSNorm:
    """RMS Normalization module."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps

        # Initialize parameters
        self.gamma = np.ones(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return rms_norm(x, self.gamma, self.eps)

    def get_stats(self, x: np.ndarray) -> dict:
        """Get statistics for visualization."""
        rms = np.sqrt(np.mean(x**2, axis=-1) + self.eps)

        return {
            "input_rms": np.mean(rms),
            "output_mean": np.mean(self.forward(x)),
            "output_var": np.var(self.forward(x)),
            "gamma_mean": np.mean(self.gamma),
        }


def get_normalization_module(norm_type: str, hidden_dim: int) -> LayerNorm | RMSNorm:
    """Factory function to create normalization modules."""
    if norm_type == "LayerNorm":
        return LayerNorm(hidden_dim)
    if norm_type == "RMSNorm":
        return RMSNorm(hidden_dim)
    if norm_type == "None":
        return None
    raise ValueError(f"Unknown normalization type: {norm_type}")
