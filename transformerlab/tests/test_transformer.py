"""
Tests for the transformer model.
"""

import numpy as np

from transformerlab.core.transformer import Transformer, TransformerBlock


def test_transformer_initialization():
    """Test transformer initialization."""
    vocab_size = 100
    hidden_dim = 64
    num_layers = 2
    num_heads = 4

    model = Transformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    assert model.vocab_size == vocab_size
    assert model.hidden_dim == hidden_dim
    assert model.num_layers == num_layers
    assert model.num_heads == num_heads
    assert len(model.blocks) == num_layers


def test_transformer_forward_pass():
    """Test transformer forward pass."""
    vocab_size = 50
    hidden_dim = 32
    num_layers = 2
    num_heads = 4

    model = Transformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    batch_size, seq_len = 2, 10
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

    logits, stats = model.forward(x, targets)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert "loss" in stats
    assert "layer_stats" in stats
    assert len(stats["layer_stats"]) == num_layers


def test_transformer_block():
    """Test transformer block."""
    hidden_dim = 64
    num_heads = 4
    ff_dim = 128

    block = TransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim)

    batch_size, seq_len = 2, 10
    x = np.random.randn(batch_size, seq_len, hidden_dim)

    output, stats = block.forward(x)

    assert output.shape == x.shape
    assert "attention" in stats
    assert "feed_forward" in stats


def test_transformer_generation():
    """Test text generation."""
    vocab_size = 50
    hidden_dim = 32
    num_layers = 2
    num_heads = 4

    model = Transformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    batch_size = 1
    seq_len = 5
    prompt = np.random.randint(0, vocab_size, (batch_size, seq_len))

    generated = model.generate(prompt, max_length=10)

    assert generated.shape[0] == batch_size
    assert generated.shape[1] == seq_len + 10  # Original + generated


def test_transformer_model_stats():
    """Test model statistics collection."""
    vocab_size = 50
    hidden_dim = 32
    num_layers = 2
    num_heads = 4

    model = Transformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    stats = model.get_model_stats()

    assert "loss_history" in stats
    assert "step_count" in stats
    assert "config" in stats
    assert stats["config"]["vocab_size"] == vocab_size
    assert stats["config"]["hidden_dim"] == hidden_dim


def test_transformer_different_configs():
    """Test transformer with different configurations."""
    vocab_size = 50
    hidden_dim = 32
    num_layers = 2
    num_heads = 4

    # Test different normalization types
    for norm_type in ["LayerNorm", "RMSNorm", "None"]:
        model = Transformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            norm_type=norm_type,
        )

        batch_size, seq_len = 2, 10
        x = np.random.randint(0, vocab_size, (batch_size, seq_len))
        targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

        logits, stats = model.forward(x, targets)
        assert logits.shape == (batch_size, seq_len, vocab_size)

    # Test different activation types
    for activation_type in ["ReLU", "GeLU", "Swish"]:
        model = Transformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            activation_type=activation_type,
        )

        batch_size, seq_len = 2, 10
        x = np.random.randint(0, vocab_size, (batch_size, seq_len))
        targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

        logits, stats = model.forward(x, targets)
        assert logits.shape == (batch_size, seq_len, vocab_size)
