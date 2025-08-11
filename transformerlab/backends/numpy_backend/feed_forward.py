"""
NumPy implementation of feed-forward network.

Provides fast vectorized feed-forward computation using NumPy operations.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, List

from ..abstract import AbstractFeedForward


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation function."""
    return (x > 0).astype(float)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function (approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU activation function (approximation)."""
    tanh_term = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
    sech2_term = 1 - tanh_term**2
    return 0.5 * (1 + tanh_term) + 0.5 * x * sech2_term * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function."""
    return x * (1 / (1 + np.exp(-x)))


def swish_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of Swish activation function."""
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid + x * sigmoid * (1 - sigmoid)


class NumPyFeedForward(AbstractFeedForward):
    """NumPy feed-forward network implementation."""
    
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim, ff_dim, activation_type, residual_type)
        
        # Linear layers
        self.W1 = np.random.randn(hidden_dim, ff_dim) * np.sqrt(2.0 / hidden_dim)
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, hidden_dim) * np.sqrt(2.0 / ff_dim)
        self.b2 = np.zeros(hidden_dim)
        
        # Activation function
        if activation_type == "ReLU":
            self.activation_fn = relu
            self.activation_derivative = relu_derivative
        elif activation_type == "GeLU":
            self.activation_fn = gelu
            self.activation_derivative = gelu_derivative
        elif activation_type == "Swish":
            self.activation_fn = swish
            self.activation_derivative = swish_derivative
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
        
        self.dropout_rate = dropout
        self.training = True
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout during training."""
        if self.training and self.dropout_rate > 0:
            # Create dropout mask
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x
    
    def forward(self, x: np.ndarray, norm_layer: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            norm_layer: Optional normalization layer (not used in this implementation)
            
        Returns:
            Tuple of (output, ff_stats)
        """
        # First linear layer
        h1 = np.matmul(x, self.W1) + self.b1  # (batch, seq, ff_dim)
        
        # Activation
        h1_activated = self.activation_fn(h1)
        
        # Apply dropout
        h1_activated = self._apply_dropout(h1_activated)
        
        # Second linear layer
        output = np.matmul(h1_activated, self.W2) + self.b2  # (batch, seq, hidden_dim)
        
        # Collect statistics
        ff_stats = {
            "hidden_mean": np.mean(h1),
            "hidden_std": np.std(h1),
            "activated_mean": np.mean(h1_activated),
            "activated_std": np.std(h1_activated),
            "output_mean": np.mean(output),
            "output_std": np.std(output),
            "activation_type": self.activation_type,
        }
        
        # Cache for backward pass
        self._cache = {
            'input': x,
            'h1': h1,
            'h1_activated': h1_activated,
        }
        
        return output, ff_stats
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Backward pass through feed-forward network.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Tuple of (grad_input, ff_gradients)
        """
        if not hasattr(self, '_cache'):
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Retrieve cached values
        x = self._cache['input']
        h1 = self._cache['h1']
        h1_activated = self._cache['h1_activated']
        
        # Backward through second linear layer
        grad_W2 = np.matmul(h1_activated.transpose(0, 2, 1), grad_output).sum(axis=0)
        grad_b2 = grad_output.sum(axis=(0, 1))
        grad_h1_activated = np.matmul(grad_output, self.W2.T)
        
        # Backward through activation function
        grad_h1 = grad_h1_activated * self.activation_derivative(h1)
        
        # Backward through first linear layer
        grad_W1 = np.matmul(x.transpose(0, 2, 1), grad_h1).sum(axis=0)
        grad_b1 = grad_h1.sum(axis=(0, 1))
        grad_input = np.matmul(grad_h1, self.W1.T)
        
        # Collect gradients
        gradients = {
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2,
        }
        
        return grad_input, gradients
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get trainable parameters."""
        return [self.W1, self.b1, self.W2, self.b2]