"""
Feed-forward network module for the Transformer Intuition Lab.
Pure NumPy implementation for educational purposes.
"""


import numpy as np

from .activations import get_activation_module


class FeedForward:
    """Feed-forward network with configurable activation and residual connections."""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
    ):
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.activation_type = activation_type
        self.residual_type = residual_type

        # Initialize weight matrices
        self.w1 = np.random.randn(hidden_dim, ff_dim) * 0.02
        self.w2 = np.random.randn(ff_dim, hidden_dim) * 0.02

        # Initialize biases
        self.b1 = np.zeros(ff_dim)
        self.b2 = np.zeros(hidden_dim)

        # Get activation module
        self.activation = get_activation_module(activation_type)

    def forward(self, x: np.ndarray, norm_module=None) -> tuple[np.ndarray, dict]:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            norm_module: Optional normalization module

        Returns:
            Tuple of (output, ff_stats)
        """
        batch_size, seq_len, _ = x.shape

        # Apply normalization if provided and it's Pre-LN
        if norm_module is not None and self.residual_type == "Pre-LN":
            x_norm = norm_module.forward(x)
        else:
            x_norm = x

        # First linear transformation
        h = np.matmul(x_norm, self.w1) + self.b1

        # Apply activation
        if self.activation_type == "SwiGLU":
            # For SwiGLU, we need to split the input and create a gate
            h_dim = h.shape[-1] // 2
            h_main = h[:, :, :h_dim]
            h_gate = h[:, :, h_dim:]
            h_activated = self.activation.forward(h_main, h_gate)
        else:
            h_activated = self.activation.forward(h)

        # Second linear transformation
        output = np.matmul(h_activated, self.w2) + self.b2

        # Apply residual connection
        if self.residual_type == "Pre-LN":
            # Pre-LN: residual connection is applied after the FFN
            final_output = x + output
        elif self.residual_type == "Post-LN":
            # Post-LN: residual connection is applied before normalization
            final_output = x + output
            if norm_module is not None:
                final_output = norm_module.forward(final_output)
        else:  # Sandwich
            # Sandwich: residual connection is applied in the middle
            final_output = x + output

        # Collect statistics
        stats = self._get_ff_stats(x, h, h_activated, output, final_output)

        return final_output, stats

    def _get_ff_stats(
        self,
        x: np.ndarray,
        h: np.ndarray,
        h_activated: np.ndarray,
        output: np.ndarray,
        final_output: np.ndarray,
    ) -> dict:
        """Collect feed-forward statistics for visualization."""
        return {
            "input_mean": np.mean(x),
            "input_std": np.std(x),
            "hidden_mean": np.mean(h),
            "hidden_std": np.std(h),
            "activated_mean": np.mean(h_activated),
            "activated_std": np.std(h_activated),
            "output_mean": np.mean(output),
            "output_std": np.std(output),
            "final_output_mean": np.mean(final_output),
            "final_output_std": np.std(final_output),
            "activation_type": self.activation_type,
            "residual_type": self.residual_type,
        }


def get_residual_type(residual_type: str) -> str:
    """Validate and return residual connection type."""
    valid_types = ["Pre-LN", "Post-LN", "Sandwich"]
    if residual_type not in valid_types:
        raise ValueError(
            f"Invalid residual type: {residual_type}. Must be one of {valid_types}"
        )
    return residual_type
