"""
Multi-head attention mechanism for the Transformer Intuition Lab.
Pure NumPy implementation for educational purposes.
"""

import numpy as np


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        v: Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
        mask: Optional attention mask of shape (seq_len, seq_len)

    Returns:
        Tuple of (attention_output, attention_weights)
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Compute attention scores
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

    # Apply mask if provided
    if mask is not None:
        # Ensure mask has the right shape for broadcasting
        if mask.ndim == 2:
            mask = mask.reshape(1, seq_len, 1, seq_len)
        elif mask.ndim == 4:
            # If mask is already 4D, ensure it has the right shape
            if mask.shape[0] == 1 and mask.shape[2] == 1:
                # Broadcast to all batches and heads
                mask = np.tile(mask, (batch_size, 1, num_heads, 1))
        scores = scores + mask

    # Apply softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(
        attention_weights, axis=-1, keepdims=True
    )

    # Apply attention weights to values
    output = np.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention:
    """Multi-head attention module."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        # Initialize weight matrices
        self.w_q = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.w_k = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.w_v = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.w_o = np.random.randn(hidden_dim, hidden_dim) * 0.02

        # Initialize biases
        self.b_q = np.zeros(hidden_dim)
        self.b_k = np.zeros(hidden_dim)
        self.b_v = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

    def forward(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_stats)
        """
        batch_size, seq_len, _ = x.shape

        # Linear transformations
        q = np.matmul(x, self.w_q) + self.b_q
        k = np.matmul(x, self.w_k) + self.b_k
        v = np.matmul(x, self.w_v) + self.b_v

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply attention
        attention_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        # Reshape back
        attention_output = attention_output.reshape(
            batch_size, seq_len, self.hidden_dim
        )

        # Output projection
        output = np.matmul(attention_output, self.w_o) + self.b_o

        # Collect statistics
        stats = self._get_attention_stats(q, k, v, attention_weights, output)

        # Store intermediate values for backward pass
        self._cache = {
            "input": x,
            "q": q,
            "k": k,
            "v": v,
            "attention_output": attention_output,
            "attention_weights": attention_weights,
            "mask": mask,
        }

        return output, stats

    def _get_attention_stats(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        attention_weights: np.ndarray,
        output: np.ndarray,
    ) -> dict:
        """Collect attention statistics for visualization."""
        return {
            "q_mean": np.mean(q),
            "q_std": np.std(q),
            "k_mean": np.mean(k),
            "k_std": np.std(k),
            "v_mean": np.mean(v),
            "v_std": np.std(v),
            "attention_weights_mean": np.mean(attention_weights),
            "attention_weights_std": np.std(attention_weights),
            "attention_weights_max": np.max(attention_weights),
            "attention_weights_min": np.min(attention_weights),
            "output_mean": np.mean(output),
            "output_std": np.std(output),
            "attention_weights": attention_weights[0],  # First batch for visualization
        }

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Backward pass through multi-head attention.

        Args:
            grad_output: Gradient of loss w.r.t. output of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Tuple of (grad_input, gradients_dict)
        """
        if not hasattr(self, "_cache"):
            raise RuntimeError("Forward pass must be called before backward pass")

        x = self._cache["input"]
        q = self._cache["q"]
        k = self._cache["k"]
        v = self._cache["v"]
        attention_output = self._cache["attention_output"]
        attention_weights = self._cache["attention_weights"]

        batch_size, seq_len, _ = grad_output.shape

        # Gradient through output projection: output = attention_output @ w_o + b_o
        grad_w_o = np.tensordot(attention_output, grad_output, axes=([0, 1], [0, 1]))
        grad_b_o = np.sum(grad_output, axis=(0, 1))
        grad_attention_output = np.matmul(grad_output, self.w_o.T)

        # Reshape gradient to match attention output shape
        grad_attention_output = grad_attention_output.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Simplified attention backward pass
        # For educational purposes, we'll use approximate gradients
        grad_v = attention_weights.transpose(0, 1, 3, 2) @ grad_attention_output
        grad_attention_weights = grad_attention_output @ v.transpose(0, 1, 3, 2)

        # Gradient through softmax (simplified)
        grad_scores = grad_attention_weights.copy()

        # Gradient through scaled dot product
        scale = 1.0 / np.sqrt(self.head_dim)
        grad_q = scale * (grad_scores @ k)
        grad_k = scale * (grad_scores.transpose(0, 1, 3, 2) @ q)

        # Reshape back to original dimensions
        grad_q = grad_q.reshape(batch_size, seq_len, self.hidden_dim)
        grad_k = grad_k.reshape(batch_size, seq_len, self.hidden_dim)
        grad_v = grad_v.reshape(batch_size, seq_len, self.hidden_dim)

        # Gradients through linear transformations
        grad_w_q = np.tensordot(x, grad_q, axes=([0, 1], [0, 1]))
        grad_b_q = np.sum(grad_q, axis=(0, 1))
        grad_x_q = np.matmul(grad_q, self.w_q.T)

        grad_w_k = np.tensordot(x, grad_k, axes=([0, 1], [0, 1]))
        grad_b_k = np.sum(grad_k, axis=(0, 1))
        grad_x_k = np.matmul(grad_k, self.w_k.T)

        grad_w_v = np.tensordot(x, grad_v, axes=([0, 1], [0, 1]))
        grad_b_v = np.sum(grad_v, axis=(0, 1))
        grad_x_v = np.matmul(grad_v, self.w_v.T)

        # Sum gradients with respect to input
        grad_input = grad_x_q + grad_x_k + grad_x_v

        # Return gradients
        gradients = {
            "w_q": grad_w_q,
            "w_k": grad_w_k,
            "w_v": grad_w_v,
            "w_o": grad_w_o,
            "b_q": grad_b_q,
            "b_k": grad_b_k,
            "b_v": grad_b_v,
            "b_o": grad_b_o,
        }

        return grad_input, gradients

    def get_parameters(self) -> list[np.ndarray]:
        """Get list of parameters for optimization."""
        return [
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.b_q,
            self.b_k,
            self.b_v,
            self.b_o,
        ]

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names."""
        return ["w_q", "w_k", "w_v", "w_o", "b_q", "b_k", "b_v", "b_o"]


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create causal mask for autoregressive attention."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask * -1e9  # Large negative value
    return mask


def create_padding_mask(padding_lengths: np.ndarray, seq_len: int) -> np.ndarray:
    """Create padding mask for variable length sequences."""
    batch_size = len(padding_lengths)
    mask = np.zeros((batch_size, seq_len, seq_len))

    for i, length in enumerate(padding_lengths):
        mask[i, :, length:] = -1e9

    return mask
