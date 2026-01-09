"""
Simplified Transformer Configuration - Educational Focus

This replaces the complex Pydantic schema system with a simple dataclass
that provides clear educational error messages for constraint violations.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TransformerConfig:
    """Simple transformer configuration with educational constraint validation."""

    # Core architecture
    vocab_size: int = 1000
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    seq_len: int = 1024

    # Feed forward network
    ff_dim: int | None = None  # Defaults to 4 * hidden_dim

    # Component types
    norm_type: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    activation_type: Literal["ReLU", "GELU", "SwiGLU"] = "ReLU"
    residual_type: Literal["Pre-LN", "Post-LN"] = "Pre-LN"
    pos_encoding_type: Literal["Sinusoidal", "Learned"] = "Sinusoidal"

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration with educational error messages."""

        # Set ff_dim default
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_dim

        # Core constraint: hidden_dim must be divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"🎓 Educational Error: hidden_dim ({self.hidden_dim}) must be evenly divisible by num_heads ({self.num_heads})\\n"
                f"   This is because attention splits the hidden dimension across heads.\\n"
                f"   Each head gets hidden_dim ÷ num_heads = {self.hidden_dim} ÷ {self.num_heads} = {self.hidden_dim / self.num_heads:.1f} dimensions.\\n"
                f"   Try: hidden_dim={self.hidden_dim + (self.num_heads - self.hidden_dim % self.num_heads)} or num_heads={self._find_valid_heads()}"
            )

        # Reasonable bounds checking
        if self.hidden_dim < self.num_heads:
            raise ValueError(
                f"🎓 Educational Error: hidden_dim ({self.hidden_dim}) should be at least as large as num_heads ({self.num_heads})\\n"
                f"   Each attention head needs at least 1 dimension to be meaningful.\\n"
                f"   Try increasing hidden_dim to at least {self.num_heads}."
            )

        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

    def _find_valid_heads(self) -> int:
        """Find the largest number of heads that divides hidden_dim."""
        for heads in range(self.num_heads, 0, -1):
            if self.hidden_dim % heads == 0:
                return heads
        return 1

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads

    @property
    def total_params(self) -> int:
        """Rough parameter count estimate for educational purposes."""
        # Token embeddings: vocab_size * hidden_dim
        token_embed = self.vocab_size * self.hidden_dim

        # Per layer: attention (4 * hidden_dim^2) + ffn (2 * hidden_dim * ff_dim) + norms (4 * hidden_dim)
        per_layer = (4 * self.hidden_dim**2) + (2 * self.hidden_dim * self.ff_dim) + (4 * self.hidden_dim)

        # Output projection: hidden_dim * vocab_size
        output_proj = self.hidden_dim * self.vocab_size

        return token_embed + (self.num_layers * per_layer) + output_proj

    def summary(self) -> str:
        """Human-readable configuration summary."""
        return f"""
🔧 Transformer Configuration:
   Architecture: {self.num_layers} layers × {self.num_heads} heads × {self.head_dim} dims
   Vocabulary: {self.vocab_size:,} tokens
   Sequence Length: {self.seq_len:,} positions  
   Parameters: ~{self.total_params:,} ({self.total_params/1e6:.1f}M)
   
   Components: {self.norm_type} + {self.activation_type} + {self.residual_type}
   Training: lr={self.learning_rate}, batch={self.batch_size}, dropout={self.dropout}
        """.strip()


# Convenience functions for common configurations
def tiny_transformer() -> TransformerConfig:
    """Tiny configuration for testing and experimentation."""
    return TransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        seq_len=32,
        batch_size=4
    )

def small_transformer() -> TransformerConfig:
    """Small configuration similar to GPT-2 small."""
    return TransformerConfig(
        vocab_size=10000,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        seq_len=512,
        batch_size=16
    )

def medium_transformer() -> TransformerConfig:
    """Medium configuration for more realistic experiments."""
    return TransformerConfig(
        vocab_size=32000,
        hidden_dim=512,
        num_heads=8,
        num_layers=12,
        seq_len=1024,
        batch_size=8
    )


if __name__ == "__main__":
    # Test configurations
    print("Testing configurations...")

    try:
        config = tiny_transformer()
        print("✅ Tiny config:", config.summary())
    except Exception as e:
        print("❌ Tiny config failed:", e)

    try:
        # This should fail with educational error
        bad_config = TransformerConfig(hidden_dim=100, num_heads=7)
    except ValueError as e:
        print("✅ Validation caught error:", e)

    print("\\nConfiguration tests complete!")
