"""
Enhanced ModelConfig with corrected mathematics and modern transformer support.
"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)


class ModelConfig(BaseModel):
    """
    Production-ready transformer configuration with:
    - Cross-field validation ensuring mathematical consistency
    - Modern architecture support (GQA/MQA, SwiGLU, RoPE, etc.)
    - Accurate parameter/memory/FLOPs estimation
    - Hardware-aware optimization hints
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Core architecture - these define the model size and capacity
    hidden_dim: int
    num_heads: int
    num_kv_heads: int | None = None  # For GQA/MQA - None means standard MHA
    ff_mult: float = 4.0  # FFN multiplier instead of raw ff_dim for flexibility
    num_layers: int

    # Implementation configuration
    tie_weights: bool = True  # Tie input/output embeddings (reduces params ~30%)
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    attention_impl: Literal["flash", "eager"] = "eager"

    # Architecture choices - modern defaults
    activation_type: Literal["ReLU", "GeLU", "Swish", "SwiGLU"] = "SwiGLU"
    norm_type: Literal["LayerNorm", "RMSNorm", "None"] = "LayerNorm"
    pos_encoding_type: Literal["Sinusoidal", "RoPE", "ALiBi"] = "RoPE"

    # RoPE-specific parameters (used when pos_encoding_type="RoPE")
    rope_theta: float = 10000.0
    rope_fraction: float = 1.0  # Fraction of head_dim to apply RoPE to

    # Training/inference configuration
    seq_len: int = 2048
    vocab_size: int = 32000

    @computed_field
    @property
    def head_dim(self) -> int:
        """Dimension per attention head - must be integer."""
        return self.hidden_dim // self.num_heads

    @computed_field
    @property
    def effective_kv_heads(self) -> int:
        """Effective number of KV heads (for GQA/MQA support)."""
        return self.num_kv_heads or self.num_heads

    @computed_field
    @property
    def ff_dim(self) -> int:
        """
        Computed FFN dimension, aligned to 64 for tensor efficiency.
        Uses ff_mult * hidden_dim, then rounds up to nearest 64.
        """
        raw_ff = int(self.ff_mult * self.hidden_dim)
        return ((raw_ff + 63) // 64) * 64

    @computed_field
    @property
    def aligned_vocab_size(self) -> int:
        """Vocab size aligned to 64 for tensor efficiency."""
        return ((self.vocab_size + 63) // 64) * 64

    @field_validator("ff_mult")
    @classmethod
    def validate_ff_mult(cls, v):
        """Ensure reasonable FFN multiplier range."""
        if v < 1.0 or v > 8.0:
            raise ValueError("ff_mult should be between 1.0 and 8.0")
        return v

    @model_validator(mode="after")
    def validate_architecture(self):
        """
        Cross-field validation after all fields are set.
        Aggregates all constraint violations into a single clear error.
        """
        errors = []

        # Head divisibility - fundamental mathematical requirement
        if self.hidden_dim % self.num_heads != 0:
            valid_heads = [h for h in range(1, 33) if self.hidden_dim % h == 0]
            errors.append(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads}). Valid options: {valid_heads}"
            )

        # GQA/MQA validation - ensures proper attention computation
        if self.num_kv_heads is not None:
            if self.num_kv_heads < 1 or self.num_kv_heads > self.num_heads:
                errors.append(
                    f"num_kv_heads ({self.num_kv_heads}) must be between "
                    f"1 and num_heads ({self.num_heads})"
                )
            elif self.num_heads % self.num_kv_heads != 0:
                valid_kv = [
                    k for k in range(1, self.num_heads + 1) if self.num_heads % k == 0
                ]
                errors.append(
                    f"num_heads ({self.num_heads}) must be divisible by "
                    f"num_kv_heads ({self.num_kv_heads}). Valid: {valid_kv}"
                )

        # Head dimension alignment for hardware efficiency
        head_dim = self.hidden_dim // self.num_heads
        if self.dtype in ["float16", "bfloat16"] and head_dim % 8 != 0:
            errors.append(
                f"head_dim ({head_dim}) should be multiple of 8 for "
                f"{self.dtype} efficiency"
            )
        if self.attention_impl == "flash" and head_dim % 16 != 0:
            errors.append(
                f"FlashAttention requires head_dim ({head_dim}) multiple of 16"
            )

        # RoPE validation - ensure even dimension for complex number pairs
        if self.pos_encoding_type == "RoPE":
            rope_dim = int(self.rope_fraction * head_dim)
            if rope_dim % 2 != 0:
                errors.append(
                    f"RoPE dimension ({rope_dim}) must be even. "
                    f"Adjust rope_fraction or head_dim"
                )

        # Device/dtype compatibility
        if self.device == "cpu" and self.dtype == "float16":
            errors.append(
                "float16 not well supported on CPU. Consider bfloat16 or float32"
            )

        if errors:
            raise ValueError(" | ".join(errors))

        return self

    def get_warnings(self) -> list[str]:
        """
        Generate performance warnings (non-blocking).
        These don't prevent the config from being valid but indicate suboptimal choices.
        """
        warnings = []

        # Vocab alignment warning (performance, not correctness)
        if self.vocab_size % 64 != 0:
            warnings.append(
                f"vocab_size ({self.vocab_size}) not aligned to 64. "
                f"Using {self.aligned_vocab_size} internally"
            )

        # Performance warnings
        if self.dtype == "float32" and self.device != "cpu":
            warnings.append("float32 uses 2x memory vs float16/bfloat16 on GPU")

        if self.attention_impl == "eager" and self.seq_len > 2048:
            warnings.append(
                "Eager attention O(L²) scaling. Consider flash attention for long sequences"
            )

        if self.head_dim > 128:
            warnings.append(
                f"Large head_dim ({self.head_dim}) may hurt attention quality"
            )

        if self.head_dim < 32:
            warnings.append(
                f"Small head_dim ({self.head_dim}) may underutilize compute"
            )

        if self.ff_mult < 2.5 and self.activation_type == "SwiGLU":
            warnings.append("SwiGLU typically uses ff_mult >= 2.5 for good performance")

        return warnings

    def estimate_parameters(self) -> int:
        """
        Calculate total parameters with corrected math.

        Components:
        - Token embeddings: vocab_size * hidden_dim
        - Output projection: hidden_dim * vocab_size (if not tied)
        - Per layer: attention + FFN + norms
        """
        d = self.hidden_dim
        vocab = self.aligned_vocab_size

        # Token embeddings
        embed_params = vocab * d

        # Output projection (tied or separate)
        out_proj_params = 0 if self.tie_weights else d * vocab

        # Per layer parameters
        # Attention: Q(d²) + K(d*kv_d) + V(d*kv_d) + O(d²) where kv_d = d*kv_heads/heads
        kv_dim = d * self.effective_kv_heads // self.num_heads
        attn_params = d * d + 2 * d * kv_dim + d * d  # Q + K + V + O

        # FFN: depends on activation type
        if self.activation_type == "SwiGLU":
            # SwiGLU has gate, up, and down projections
            ffn_params = 3 * d * self.ff_dim  # gate, up, down
        else:
            # Standard activations have up and down projections
            ffn_params = 2 * d * self.ff_dim  # up, down

        # Layer norms: weight + bias per norm (2 norms per layer typically)
        norm_params = 4 * d if self.norm_type != "None" else 0

        layer_params = attn_params + ffn_params + norm_params
        total = embed_params + out_proj_params + self.num_layers * layer_params

        return total

    def estimate_memory_mb(
        self, batch_size: int = 1, include_optimizer: bool = False
    ) -> dict:
        """
        Fixed KV-cache math and memory estimation.

        Returns breakdown of memory usage in MB.
        """
        bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2}[self.dtype]

        # Parameters
        param_count = self.estimate_parameters()
        param_memory = param_count * bytes_per_param / (1024 * 1024)

        # KV cache: 2 tensors (K,V) * layers * batch * seq * kv_heads * head_dim
        # This was the key bug fix - proper KV head dimensionality
        kv_memory = (
            2
            * self.num_layers
            * batch_size
            * self.seq_len
            * self.effective_kv_heads
            * self.head_dim
            * bytes_per_param
        ) / (1024 * 1024)

        # Activation memory (rough estimate assuming gradient checkpointing)
        activation_memory = param_memory * 0.3

        total = param_memory + kv_memory + activation_memory
        if include_optimizer:
            # Adam: parameters + momentum + variance + copy
            total += param_memory * 4

        return {
            "parameters_mb": param_memory,
            "kv_cache_mb": kv_memory,
            "activations_mb": activation_memory,
            "optimizer_mb": param_memory * 4 if include_optimizer else 0,
            "total_mb": total,
        }

    def estimate_flops_per_token(
        self, mode: Literal["prefill", "decode"] = "prefill"
    ) -> dict:
        """
        Corrected FLOPs estimation with clearer token-wise computation.

        Args:
            mode: "prefill" uses full sequence length, "decode" uses length 1

        Returns:
            Dictionary with FLOPs breakdown per token.
        """
        d = self.hidden_dim
        ff = self.ff_dim
        vocab = self.aligned_vocab_size
        kv_heads = self.effective_kv_heads

        # Context length for attention computation
        ctx_len = self.seq_len if mode == "prefill" else 1

        # Per layer, per token FLOPs
        # QKV projections: Q(d²) + K(d*kv_d) + V(d*kv_d) where kv_d = d*kv_heads/heads
        kv_dim = d * kv_heads // self.num_heads
        qkv_flops = d * d + 2 * d * kv_dim

        # Attention computation: QK^T + softmax*V
        # This scales with context length (key difference between prefill/decode)
        attn_compute_flops = 2 * ctx_len * d * self.num_heads

        # Output projection
        attn_out_flops = d * d

        # FFN computation
        if self.activation_type == "SwiGLU":
            ffn_flops = 3 * d * ff  # gate, up, down
        else:
            ffn_flops = 2 * d * ff  # up, down

        # Per layer total
        layer_flops = qkv_flops + attn_compute_flops + attn_out_flops + ffn_flops

        # All layers + final projection to vocabulary
        total_flops = self.num_layers * layer_flops + d * vocab

        return {
            f"total_flops_per_token_{mode}": total_flops,
            f"attention_flops_{mode}": (qkv_flops + attn_compute_flops + attn_out_flops)
            * self.num_layers,
            f"ffn_flops_{mode}": ffn_flops * self.num_layers,
            "context_length": ctx_len,
        }
