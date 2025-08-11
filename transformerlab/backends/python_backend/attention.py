"""
Pure Python implementation of multi-head attention.

This implementation uses explicit loops and operations to show
every step of the attention mechanism clearly.
"""

import math
from typing import Any

from ..abstract import AbstractAttention
from .utils import (
    add_3d,
    copy_tensor,
    get_shape,
    matmul_3d,
    randn,
    zeros,
)


class PythonAttention(AbstractAttention):
    """Pure Python multi-head attention implementation."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, verbose: bool = True):
        super().__init__(hidden_dim, num_heads, dropout)
        self.verbose = verbose

        # Initialize weight matrices (hidden_dim, hidden_dim)
        self.w_q = randn((hidden_dim, hidden_dim))
        self.w_k = randn((hidden_dim, hidden_dim))
        self.w_v = randn((hidden_dim, hidden_dim))
        self.w_o = randn((hidden_dim, hidden_dim))

        # Initialize biases (hidden_dim,)
        self.b_q = [0.0] * hidden_dim
        self.b_k = [0.0] * hidden_dim
        self.b_v = [0.0] * hidden_dim
        self.b_o = [0.0] * hidden_dim

        # Cache for backward pass
        self._cache = {}

    def forward(self, x: list[list[list[float]]], mask: Any | None = None) -> tuple[list[list[list[float]]], dict[str, Any]]:
        """Forward pass through attention with explicit steps."""
        batch_size, seq_len, _ = get_shape(x)

        if self.verbose:
            print(f"[PythonAttention] Forward pass: batch_size={batch_size}, seq_len={seq_len}")

        # Step 1: Linear transformations to get Q, K, V
        if self.verbose:
            print("  Step 1: Computing Q, K, V projections...")
        q = add_3d(matmul_3d(x, self.w_q), self.b_q)  # (batch, seq, hidden)
        k = add_3d(matmul_3d(x, self.w_k), self.b_k)  # (batch, seq, hidden)
        v = add_3d(matmul_3d(x, self.w_v), self.b_v)  # (batch, seq, hidden)

        # Step 2: Reshape for multi-head attention
        if self.verbose:
            print(f"  Step 2: Reshaping for {self.num_heads} heads...")
        q_heads = self._reshape_for_heads(q)  # (batch, seq, num_heads, head_dim)
        k_heads = self._reshape_for_heads(k)
        v_heads = self._reshape_for_heads(v)

        # Step 3: Compute attention scores
        if self.verbose:
            print("  Step 3: Computing attention scores...")
        scores = self._compute_attention_scores(q_heads, k_heads)  # (batch, num_heads, seq, seq)

        # Step 4: Apply mask if provided
        if mask is not None:
            if self.verbose:
                print("  Step 4: Applying attention mask...")
            scores = self._apply_mask(scores, mask)

        # Step 5: Apply softmax
        if self.verbose:
            print("  Step 5: Applying softmax to get attention weights...")
        attention_weights = self._apply_softmax(scores)  # (batch, num_heads, seq, seq)

        # Step 6: Apply attention to values
        if self.verbose:
            print("  Step 6: Applying attention weights to values...")
        attention_output = self._apply_attention_to_values(attention_weights, v_heads)  # (batch, seq, num_heads, head_dim)

        # Step 7: Concatenate heads
        if self.verbose:
            print("  Step 7: Concatenating attention heads...")
        concat_output = self._concatenate_heads(attention_output)  # (batch, seq, hidden)

        # Step 8: Final output projection
        if self.verbose:
            print("  Step 8: Final output projection...")
        output = add_3d(matmul_3d(concat_output, self.w_o), self.b_o)

        # Cache intermediate values for backward pass
        self._cache = {
            'input': copy_tensor(x),
            'q': copy_tensor(q),
            'k': copy_tensor(k),
            'v': copy_tensor(v),
            'attention_weights': copy_tensor(attention_weights),
            'concat_output': copy_tensor(concat_output),
        }

        # Collect statistics
        stats = self._compute_stats(q, k, v, attention_weights, output)

        return output, stats

    def _reshape_for_heads(self, x: list[list[list[float]]]) -> list[list[list[list[float]]]]:
        """Reshape tensor for multi-head attention: (batch, seq, hidden) -> (batch, seq, num_heads, head_dim)."""
        batch_size, seq_len, hidden_dim = get_shape(x)

        # Create 4D tensor
        result = []
        for batch in range(batch_size):
            batch_result = []
            for seq in range(seq_len):
                head_result = []
                for head in range(self.num_heads):
                    head_values = []
                    start_idx = head * self.head_dim
                    end_idx = start_idx + self.head_dim
                    for dim in range(start_idx, end_idx):
                        head_values.append(x[batch][seq][dim])
                    head_result.append(head_values)
                batch_result.append(head_result)
            result.append(batch_result)

        return result

    def _compute_attention_scores(
        self,
        q: list[list[list[list[float]]]],
        k: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Compute scaled dot-product attention scores."""
        batch_size, seq_len, num_heads, head_dim = len(q), len(q[0]), len(q[0][0]), len(q[0][0][0])
        scale = 1.0 / math.sqrt(head_dim)

        # Result shape: (batch, num_heads, seq, seq)
        scores = [[[[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
                   for _ in range(num_heads)] for _ in range(batch_size)]

        for batch in range(batch_size):
            for head in range(num_heads):
                for i in range(seq_len):  # Query position
                    for j in range(seq_len):  # Key position
                        # Dot product between q[batch][i][head] and k[batch][j][head]
                        dot_product = 0.0
                        for dim in range(head_dim):
                            dot_product += q[batch][i][head][dim] * k[batch][j][head][dim]

                        scores[batch][head][i][j] = dot_product * scale

        return scores

    def _apply_mask(self, scores: list[list[list[list[float]]]], mask: Any) -> list[list[list[list[float]]]]:
        """Apply attention mask to scores."""
        # For simplicity, assume mask is a 2D matrix (seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = len(scores), len(scores[0]), len(scores[0][0]), len(scores[0][0][0])

        result = copy_tensor(scores)

        for batch in range(batch_size):
            for head in range(num_heads):
                for i in range(seq_len):
                    for j in range(seq_len):
                        if mask[i][j] < 0:  # Mask value indicates blocking
                            result[batch][head][i][j] = -1e9  # Large negative value

        return result

    def _apply_softmax(self, scores: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
        """Apply softmax to attention scores."""
        batch_size, num_heads, seq_len, _ = len(scores), len(scores[0]), len(scores[0][0]), len(scores[0][0][0])

        result = [[[[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
                   for _ in range(num_heads)] for _ in range(batch_size)]

        for batch in range(batch_size):
            for head in range(num_heads):
                for i in range(seq_len):
                    # Apply softmax to scores[batch][head][i] (attention weights for position i)
                    row_scores = scores[batch][head][i]

                    # Find max for numerical stability
                    max_score = max(row_scores)

                    # Compute exponentials
                    exp_scores = [math.exp(score - max_score) for score in row_scores]

                    # Compute sum
                    exp_sum = sum(exp_scores)

                    # Normalize
                    for j in range(seq_len):
                        result[batch][head][i][j] = exp_scores[j] / exp_sum

        return result

    def _apply_attention_to_values(
        self,
        attention_weights: list[list[list[list[float]]]],
        v: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Apply attention weights to values."""
        batch_size, seq_len, num_heads, head_dim = len(v), len(v[0]), len(v[0][0]), len(v[0][0][0])

        # Result shape: (batch, seq, num_heads, head_dim)
        result = [[[[0.0 for _ in range(head_dim)] for _ in range(num_heads)]
                   for _ in range(seq_len)] for _ in range(batch_size)]

        for batch in range(batch_size):
            for head in range(num_heads):
                for i in range(seq_len):  # Output position
                    for dim in range(head_dim):  # Output dimension
                        weighted_sum = 0.0
                        for j in range(seq_len):  # Input position
                            weight = attention_weights[batch][head][i][j]
                            value = v[batch][j][head][dim]
                            weighted_sum += weight * value
                        result[batch][i][head][dim] = weighted_sum

        return result

    def _concatenate_heads(self, x: list[list[list[list[float]]]]) -> list[list[list[float]]]:
        """Concatenate attention heads: (batch, seq, num_heads, head_dim) -> (batch, seq, hidden_dim)."""
        batch_size, seq_len, num_heads, head_dim = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        hidden_dim = num_heads * head_dim

        result = zeros((batch_size, seq_len, hidden_dim))

        for batch in range(batch_size):
            for seq in range(seq_len):
                for head in range(num_heads):
                    start_idx = head * head_dim
                    for dim in range(head_dim):
                        result[batch][seq][start_idx + dim] = x[batch][seq][head][dim]

        return result

    def _compute_stats(
        self,
        q: list[list[list[float]]],
        k: list[list[list[float]]],
        v: list[list[list[float]]],
        attention_weights: list[list[list[list[float]]]],
        output: list[list[list[float]]]
    ) -> dict[str, Any]:
        """Compute attention statistics."""
        from .utils import mean, std

        return {
            "q_mean": mean(q),
            "q_std": std(q),
            "k_mean": mean(k),
            "k_std": std(k),
            "v_mean": mean(v),
            "v_std": std(v),
            "attention_weights_mean": mean(attention_weights),
            "attention_weights_std": std(attention_weights),
            "output_mean": mean(output),
            "output_std": std(output),
            "attention_weights": attention_weights[0],  # First batch for visualization
        }

    def backward(self, grad_output: list[list[list[float]]]) -> tuple[list[list[list[float]]], dict[str, Any]]:
        """Backward pass (simplified for educational purposes)."""
        if not self._cache:
            raise RuntimeError("Forward pass must be called before backward pass")

        # For pure Python backend, we implement simplified gradients
        # In practice, this would be much more complex

        # Return zero gradients for now (placeholder)
        input_shape = get_shape(self._cache['input'])
        grad_input = zeros(input_shape)

        gradients = {
            'w_q': zeros(get_shape(self.w_q)),
            'w_k': zeros(get_shape(self.w_k)),
            'w_v': zeros(get_shape(self.w_v)),
            'w_o': zeros(get_shape(self.w_o)),
            'b_q': [0.0] * len(self.b_q),
            'b_k': [0.0] * len(self.b_k),
            'b_v': [0.0] * len(self.b_v),
            'b_o': [0.0] * len(self.b_o),
        }

        return grad_input, gradients

    def get_parameters(self) -> list[Any]:
        """Get trainable parameters."""
        return [self.w_q, self.w_k, self.w_v, self.w_o, self.b_q, self.b_k, self.b_v, self.b_o]

    def get_attention_weights(self) -> Any:
        """Get attention weights for visualization."""
        if 'attention_weights' in self._cache:
            return self._cache['attention_weights']
        return None
