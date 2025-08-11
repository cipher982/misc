"""
Positional encoding modules for the Transformer Intuition Lab.
Pure NumPy implementations for educational purposes.
"""


import numpy as np


def sinusoidal_positional_encoding(
    seq_len: int, hidden_dim: int, max_len: int = 10000
) -> np.ndarray:
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".

    Args:
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        max_len: Maximum sequence length for frequency calculation

    Returns:
        Positional encoding of shape (seq_len, hidden_dim)
    """
    pe = np.zeros((seq_len, hidden_dim))

    for pos in range(seq_len):
        for i in range(0, hidden_dim, 2):
            # Even indices: sin
            pe[pos, i] = np.sin(pos / (max_len ** (i / hidden_dim)))
            # Odd indices: cos
            if i + 1 < hidden_dim:
                pe[pos, i + 1] = np.cos(pos / (max_len ** (i / hidden_dim)))

    return pe


def rope_positional_encoding(
    x: np.ndarray, seq_len: int, head_dim: int, max_len: int = 10000
) -> np.ndarray:
    """
    Rotary Positional Encoding (RoPE) implementation.

    Args:
        x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
        seq_len: Sequence length
        head_dim: Dimension per attention head
        max_len: Maximum sequence length for frequency calculation

    Returns:
        Position-encoded tensor of same shape as x
    """
    batch_size, seq_len, num_heads, head_dim = x.shape

    # Create position indices
    positions = np.arange(seq_len).reshape(1, -1, 1, 1)

    # Create frequency bands
    freqs = 1.0 / (max_len ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = freqs.reshape(1, 1, 1, -1)

    # Calculate rotation angles
    angles = positions * freqs

    # Split input into even and odd dimensions
    x_even = x[:, :, :, ::2]
    x_odd = x[:, :, :, 1::2]

    # Apply rotation
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    x_rotated_even = x_even * cos_angles - x_odd * sin_angles
    x_rotated_odd = x_even * sin_angles + x_odd * cos_angles

    # Reconstruct output
    output = np.zeros_like(x)
    output[:, :, :, ::2] = x_rotated_even
    output[:, :, :, 1::2] = x_rotated_odd

    return output


def alibi_positional_encoding(
    seq_len: int, num_heads: int, head_dim: int
) -> np.ndarray:
    """
    Attention with Linear Biases (ALiBi) implementation.

    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per attention head

    Returns:
        ALiBi bias of shape (num_heads, seq_len, seq_len)
    """
    # Create position indices
    pos_indices = np.arange(seq_len)

    # Create head-specific slopes
    slopes = 1 / (2 ** (np.arange(num_heads) * 8 / num_heads))
    slopes = slopes.reshape(-1, 1, 1)

    # Calculate relative positions
    relative_positions = pos_indices.reshape(1, 1, -1) - pos_indices.reshape(1, -1, 1)

    # Apply ALiBi bias
    alibi_bias = slopes * relative_positions

    return alibi_bias


class PositionalEncoding:
    """Base class for positional encoding modules."""

    def __init__(self, encoding_type: str = "Sinusoidal"):
        self.encoding_type = encoding_type

    def forward(self, x: np.ndarray, seq_len: int = None) -> np.ndarray:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

    def get_stats(self, x: np.ndarray, seq_len: int = None) -> dict:
        """Get encoding statistics for visualization."""
        output = self.forward(x, seq_len)

        return {
            "input_shape": x.shape,
            "output_shape": output.shape,
            "encoding_type": self.encoding_type,
            "output_mean": np.mean(output),
            "output_std": np.std(output),
            "output_min": np.min(output),
            "output_max": np.max(output),
        }


class SinusoidalPE(PositionalEncoding):
    """Sinusoidal positional encoding."""

    def __init__(self, hidden_dim: int, max_len: int = 10000):
        super().__init__("Sinusoidal")
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.pe = None

    def forward(self, x: np.ndarray, seq_len: int = None) -> np.ndarray:
        """Add sinusoidal positional encoding to input."""
        if seq_len is None:
            seq_len = x.shape[1]

        # Create or update positional encoding
        if self.pe is None or self.pe.shape[0] < seq_len:
            self.pe = sinusoidal_positional_encoding(
                seq_len, self.hidden_dim, self.max_len
            )

        # Add positional encoding
        return x + self.pe[:seq_len]


class RoPE(PositionalEncoding):
    """Rotary Positional Encoding."""

    def __init__(self, head_dim: int, max_len: int = 10000):
        super().__init__("RoPE")
        self.head_dim = head_dim
        self.max_len = max_len

    def forward(self, x: np.ndarray, seq_len: int = None) -> np.ndarray:
        """Apply RoPE to input."""
        if seq_len is None:
            seq_len = x.shape[1]

        return rope_positional_encoding(x, seq_len, self.head_dim, self.max_len)


class ALiBi(PositionalEncoding):
    """Attention with Linear Biases."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__("ALiBi")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bias = None

    def forward(self, x: np.ndarray, seq_len: int = None) -> np.ndarray:
        """Return ALiBi bias for attention computation."""
        if seq_len is None:
            seq_len = x.shape[1]

        # Create or update bias
        if self.bias is None or self.bias.shape[1] != seq_len:
            self.bias = alibi_positional_encoding(
                seq_len, self.num_heads, self.head_dim
            )

        return self.bias


def get_positional_encoding(encoding_type: str, **kwargs) -> PositionalEncoding:
    """Factory function to create positional encoding modules."""
    if encoding_type == "Sinusoidal":
        return SinusoidalPE(kwargs.get("hidden_dim", 512))
    elif encoding_type == "RoPE":
        return RoPE(kwargs.get("head_dim", 64))
    elif encoding_type == "ALiBi":
        return ALiBi(kwargs.get("num_heads", 8), kwargs.get("head_dim", 64))
    else:
        raise ValueError(f"Unknown positional encoding type: {encoding_type}")
