"""
Configuration management for Transformer Intuition Lab using Pydantic.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for transformer model architecture."""

    vocab_size: int = Field(..., gt=0, description="Vocabulary size")
    hidden_dim: int = Field(..., gt=0, description="Hidden dimension size")
    num_layers: int = Field(..., gt=0, description="Number of transformer layers")
    num_heads: int = Field(..., gt=0, description="Number of attention heads")
    ff_dim: int | None = Field(
        None, gt=0, description="Feed-forward dimension (defaults to 4 * hidden_dim)"
    )
    norm_type: Literal["LayerNorm", "RMSNorm", "None"] = Field(
        "LayerNorm", description="Normalization type"
    )
    activation_type: Literal["ReLU", "GeLU", "Swish", "SwiGLU"] = Field(
        "ReLU", description="Activation function"
    )
    residual_type: Literal["Pre-LN", "Post-LN", "Sandwich"] = Field(
        "Pre-LN", description="Residual connection type"
    )
    pos_encoding_type: Literal["Sinusoidal", "RoPE", "ALiBi"] = Field(
        "Sinusoidal", description="Positional encoding type"
    )
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")

    @validator("ff_dim", pre=True, always=True)
    def set_ff_dim(cls, v, values):
        """Set default ff_dim to 4 * hidden_dim if not specified."""
        if v is None and "hidden_dim" in values:
            return 4 * values["hidden_dim"]
        return v

    @validator("num_heads")
    def validate_num_heads(cls, v, values):
        """Ensure num_heads divides hidden_dim evenly."""
        if "hidden_dim" in values and values["hidden_dim"] % v != 0:
            raise ValueError(
                f"num_heads ({v}) must divide hidden_dim ({values['hidden_dim']}) evenly"
            )
        return v


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    batch_size: int = Field(..., gt=0, description="Batch size")
    seq_len: int = Field(..., gt=0, description="Sequence length")
    num_steps: int = Field(..., gt=0, description="Number of training steps")
    learning_rate: float = Field(..., gt=0, description="Learning rate")
    warmup_steps: int = Field(0, ge=0, description="Number of warmup steps")
    gradient_clip: float | None = Field(
        None, gt=0, description="Gradient clipping value"
    )
    save_every: int = Field(100, gt=0, description="Save checkpoint every N steps")
    eval_every: int = Field(50, gt=0, description="Evaluate every N steps")


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    max_length: int = Field(..., gt=0, description="Maximum generation length")
    temperature: float = Field(0.8, gt=0, le=2.0, description="Generation temperature")
    top_k: int | None = Field(None, gt=0, description="Top-k sampling")
    top_p: float | None = Field(
        None, gt=0, le=1.0, description="Top-p (nucleus) sampling"
    )
    do_sample: bool = Field(True, description="Whether to use sampling")
    pad_token_id: int = Field(0, description="Padding token ID")
    eos_token_id: int | None = Field(None, description="End-of-sequence token ID")


class ExperimentConfig(BaseModel):
    """Configuration for experiments and comparisons."""

    name: str = Field(..., description="Experiment name")
    description: str | None = Field(None, description="Experiment description")
    model: ModelConfig = Field(..., description="Model configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    generation: GenerationConfig = Field(..., description="Generation configuration")
    corpus_path: Path = Field(..., description="Path to training corpus")
    output_dir: Path = Field(
        Path("experiments"), description="Output directory for results"
    )

    class Config:
        json_encoders = {Path: str}


class AppConfig(BaseModel):
    """Configuration for the Streamlit application."""

    page_title: str = Field("Transformer Intuition Lab", description="Page title")
    page_icon: str = Field("🧠", description="Page icon")
    layout: Literal["wide", "centered"] = Field("wide", description="Page layout")
    initial_sidebar_state: Literal["expanded", "collapsed"] = Field(
        "expanded", description="Initial sidebar state"
    )
    max_upload_size: int = Field(200, description="Maximum file upload size in MB")
    enable_experiment_comparison: bool = Field(
        True, description="Enable experiment comparison features"
    )
    enable_advanced_options: bool = Field(
        False, description="Show advanced configuration options"
    )


def load_config(config_path: Path) -> ExperimentConfig:
    """Load configuration from a YAML file."""
    import yaml

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    return ExperimentConfig(**config_data)


def save_config(config: ExperimentConfig, config_path: Path) -> None:
    """Save configuration to a YAML file."""
    import yaml

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig(
        name="Default Experiment",
        description="Default transformer configuration for learning",
        model=ModelConfig(
            vocab_size=100,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
        ),
        training=TrainingConfig(
            batch_size=2,
            seq_len=128,
            num_steps=100,
            learning_rate=0.001,
        ),
        generation=GenerationConfig(
            max_length=50,
            temperature=0.8,
        ),
        corpus_path=Path("transformerlab/data/tiny_shakespeare.txt"),
    )


def validate_config(config: ExperimentConfig) -> list[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []

    # Check if corpus file exists
    if not config.corpus_path.exists():
        warnings.append(f"Corpus file not found: {config.corpus_path}")

    # Check model size
    total_params = (
        config.model.vocab_size * config.model.hidden_dim  # embeddings
        + config.model.num_layers
        * (
            4 * config.model.hidden_dim * config.model.hidden_dim  # attention
            + 4 * config.model.hidden_dim * config.model.ff_dim  # feed-forward
            + 4 * config.model.hidden_dim  # layer norms
        )
    )

    if total_params > 10_000_000:  # 10M parameters
        warnings.append(f"Large model size: {total_params:,} parameters")

    # Check training configuration
    if config.training.batch_size * config.training.seq_len > 10000:
        warnings.append("Large effective batch size, may be slow on CPU")

    return warnings
