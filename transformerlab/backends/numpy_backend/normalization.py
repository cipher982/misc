"""
NumPy implementation of normalization layers.

Provides LayerNorm, RMSNorm, and other normalization functions using NumPy.
"""

import numpy as np
from typing import Optional, List


class NumPyLayerNorm:
    """NumPy Layer Normalization implementation."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(hidden_dim)  # Scale parameter
        self.beta = np.zeros(hidden_dim)  # Shift parameter
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through layer normalization.
        
        Args:
            x: Input tensor of shape (..., hidden_dim)
            
        Returns:
            Normalized tensor
        """
        # Compute mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        # Cache for backward pass
        self._cache = {
            'input': x,
            'mean': mean,
            'var': var,
            'x_normalized': x_normalized,
        }
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through layer normalization."""
        if not hasattr(self, '_cache'):
            raise RuntimeError("Forward pass must be called before backward pass")
        
        x = self._cache['input']
        mean = self._cache['mean']
        var = self._cache['var']
        x_normalized = self._cache['x_normalized']
        
        N = x.shape[-1]  # Feature dimension
        
        # Gradients w.r.t. parameters
        self.grad_gamma = np.sum(grad_output * x_normalized, axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
        
        # Gradient w.r.t. input
        grad_x_normalized = grad_output * self.gamma
        
        std_inv = 1.0 / np.sqrt(var + self.eps)
        
        grad_var = np.sum(grad_x_normalized * (x - mean), axis=-1, keepdims=True) * (-0.5) * (std_inv**3)
        grad_mean = np.sum(grad_x_normalized * (-std_inv), axis=-1, keepdims=True) + grad_var * np.sum(-2.0 * (x - mean), axis=-1, keepdims=True) / N
        
        grad_input = grad_x_normalized * std_inv + grad_var * 2.0 * (x - mean) / N + grad_mean / N
        
        return grad_input
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get trainable parameters."""
        return [self.gamma, self.beta]


class NumPyRMSNorm:
    """NumPy RMS Normalization implementation."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Learnable scale parameter (no bias in RMSNorm)
        self.gamma = np.ones(hidden_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through RMS normalization.
        
        Args:
            x: Input tensor of shape (..., hidden_dim)
            
        Returns:
            Normalized tensor
        """
        # Compute RMS
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        
        # Normalize
        x_normalized = x / rms
        
        # Scale
        output = self.gamma * x_normalized
        
        # Cache for backward pass
        self._cache = {
            'input': x,
            'rms': rms,
            'x_normalized': x_normalized,
        }
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through RMS normalization."""
        if not hasattr(self, '_cache'):
            raise RuntimeError("Forward pass must be called before backward pass")
        
        x = self._cache['input']
        rms = self._cache['rms']
        x_normalized = self._cache['x_normalized']
        
        N = x.shape[-1]  # Feature dimension
        
        # Gradient w.r.t. scale parameter
        self.grad_gamma = np.sum(grad_output * x_normalized, axis=tuple(range(grad_output.ndim - 1)))
        
        # Gradient w.r.t. input
        grad_x_normalized = grad_output * self.gamma
        
        # Gradient through RMS normalization
        grad_rms = np.sum(grad_x_normalized * x * (-1.0 / (rms**2)), axis=-1, keepdims=True)
        grad_x_squared_mean = grad_rms * (0.5 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps))
        
        grad_input = grad_x_normalized / rms + grad_x_squared_mean * (2.0 * x / N)
        
        return grad_input
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get trainable parameters."""
        return [self.gamma]


class NumPyIdentity:
    """Identity normalization (no-op) for when normalization is disabled."""
    
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Identity forward pass."""
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Identity backward pass."""
        return grad_output
    
    def get_parameters(self) -> List[np.ndarray]:
        """No parameters for identity."""
        return []


def create_numpy_normalization(norm_type: str, hidden_dim: int, eps: float = 1e-6) -> Optional[object]:
    """Factory function for creating NumPy normalization layers."""
    if norm_type == "LayerNorm":
        return NumPyLayerNorm(hidden_dim, eps)
    elif norm_type == "RMSNorm":
        return NumPyRMSNorm(hidden_dim, eps)
    elif norm_type == "None" or norm_type is None:
        return NumPyIdentity(hidden_dim)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")