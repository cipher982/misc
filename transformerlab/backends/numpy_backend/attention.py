"""
NumPy implementation of multi-head attention.

Provides fast vectorized attention computation using NumPy operations.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, List

from ..abstract import AbstractAttention


class NumPyAttention(AbstractAttention):
    """NumPy multi-head attention implementation."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(hidden_dim, num_heads, dropout)
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = hidden_dim // num_heads
        
        # Linear projection weights
        self.W_q = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.W_k = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.W_v = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.W_o = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        
        # Biases
        self.b_q = np.zeros(hidden_dim)
        self.b_k = np.zeros(hidden_dim)
        self.b_v = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)
        
        self.dropout_rate = dropout
        self.training = True
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout during training."""
        if self.training and self.dropout_rate > 0:
            # Create dropout mask
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_stats)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Linear projections
        Q = np.matmul(x, self.W_q) + self.b_q  # (batch, seq, hidden)
        K = np.matmul(x, self.W_k) + self.b_k  # (batch, seq, hidden)
        V = np.matmul(x, self.W_v) + self.b_v  # (batch, seq, hidden)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        # Attention scores: (batch, num_heads, seq, seq)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply dropout to attention weights
        attention_weights = self._apply_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = np.matmul(attention_weights, V)  # (batch, num_heads, seq, head_dim)
        
        # Transpose back and reshape
        attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch, seq, num_heads, head_dim)
        attention_output = attention_output.reshape(batch_size, seq_len, hidden_dim)
        
        # Final linear projection
        output = np.matmul(attention_output, self.W_o) + self.b_o
        
        # Collect statistics
        attention_stats = {
            "attention_weights_mean": np.mean(attention_weights),
            "attention_weights_std": np.std(attention_weights),
            "attention_weights_max": np.max(attention_weights),
            "attention_weights_min": np.min(attention_weights),
            "query_mean": np.mean(Q),
            "key_mean": np.mean(K),
            "value_mean": np.mean(V),
            "output_mean": np.mean(output),
            "scores_mean": np.mean(scores),
        }
        
        # Cache for backward pass
        self._cache = {
            'input': x,
            'Q': Q,
            'K': K,
            'V': V,
            'attention_weights': attention_weights,
            'attention_output': attention_output,
            'scores': scores,
        }
        
        return output, attention_stats
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Backward pass through multi-head attention.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Tuple of (grad_input, attention_gradients)
        """
        if not hasattr(self, '_cache'):
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Retrieve cached values
        x = self._cache['input']
        Q = self._cache['Q']
        K = self._cache['K']
        V = self._cache['V']
        attention_weights = self._cache['attention_weights']
        attention_output = self._cache['attention_output']
        
        batch_size, seq_len, hidden_dim = x.shape
        
        # Backward through output projection
        grad_W_o = np.matmul(attention_output.transpose(0, 2, 1), grad_output).sum(axis=0)
        grad_b_o = grad_output.sum(axis=(0, 1))
        grad_attention_output = np.matmul(grad_output, self.W_o.T)
        
        # Reshape gradients for multi-head structure
        grad_attention_output = grad_attention_output.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        
        # Backward through attention operation
        grad_attention_weights = np.matmul(grad_attention_output, V.transpose(0, 1, 3, 2))
        grad_V = np.matmul(attention_weights.transpose(0, 1, 3, 2), grad_attention_output)
        
        # Backward through softmax
        grad_scores = attention_weights * (
            grad_attention_weights - np.sum(grad_attention_weights * attention_weights, axis=-1, keepdims=True)
        )
        
        # Backward through scaled dot-product
        grad_Q = np.matmul(grad_scores, K) / np.sqrt(self.head_dim)
        grad_K = np.matmul(grad_scores.transpose(0, 1, 3, 2), Q) / np.sqrt(self.head_dim)
        
        # Transpose back to original shape
        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
        
        # Backward through linear projections
        grad_W_q = np.matmul(x.transpose(0, 2, 1), grad_Q).sum(axis=0)
        grad_W_k = np.matmul(x.transpose(0, 2, 1), grad_K).sum(axis=0)
        grad_W_v = np.matmul(x.transpose(0, 2, 1), grad_V).sum(axis=0)
        
        grad_b_q = grad_Q.sum(axis=(0, 1))
        grad_b_k = grad_K.sum(axis=(0, 1))
        grad_b_v = grad_V.sum(axis=(0, 1))
        
        # Gradient w.r.t. input
        grad_input = (
            np.matmul(grad_Q, self.W_q.T) +
            np.matmul(grad_K, self.W_k.T) +
            np.matmul(grad_V, self.W_v.T)
        )
        
        # Collect gradients
        gradients = {
            'W_q': grad_W_q,
            'W_k': grad_W_k,
            'W_v': grad_W_v,
            'W_o': grad_W_o,
            'b_q': grad_b_q,
            'b_k': grad_b_k,
            'b_v': grad_b_v,
            'b_o': grad_b_o,
        }
        
        return grad_input, gradients
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get trainable parameters."""
        return [
            self.W_q, self.W_k, self.W_v, self.W_o,
            self.b_q, self.b_k, self.b_v, self.b_o
        ]
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get attention weights for visualization."""
        if hasattr(self, '_cache') and 'attention_weights' in self._cache:
            return self._cache['attention_weights']
        return None