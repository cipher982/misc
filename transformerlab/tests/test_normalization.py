"""
Tests for the normalization module.
"""

import numpy as np

from transformerlab.core.normalization import LayerNorm, RMSNorm, layer_norm, rms_norm


def test_layer_norm_function():
    """Test layer normalization function."""
    batch_size, seq_len, hidden_dim = 2, 3, 4
    x = np.random.randn(batch_size, seq_len, hidden_dim)
    gamma = np.ones(hidden_dim)
    beta = np.zeros(hidden_dim)

    output = layer_norm(x, gamma, beta)

    assert output.shape == x.shape
    assert not np.allclose(output, x)  # Should be different after normalization


def test_rms_norm_function():
    """Test RMS normalization function."""
    batch_size, seq_len, hidden_dim = 2, 3, 4
    x = np.random.randn(batch_size, seq_len, hidden_dim)
    gamma = np.ones(hidden_dim)

    output = rms_norm(x, gamma)

    assert output.shape == x.shape
    assert not np.allclose(output, x)  # Should be different after normalization


def test_layer_norm_class():
    """Test LayerNorm class."""
    hidden_dim = 64
    layer_norm_module = LayerNorm(hidden_dim)

    batch_size, seq_len = 2, 10
    x = np.random.randn(batch_size, seq_len, hidden_dim)

    output = layer_norm_module.forward(x)
    assert output.shape == x.shape

    stats = layer_norm_module.get_stats(x)
    assert "input_mean" in stats
    assert "output_mean" in stats


def test_rms_norm_class():
    """Test RMSNorm class."""
    hidden_dim = 64
    rms_norm_module = RMSNorm(hidden_dim)

    batch_size, seq_len = 2, 10
    x = np.random.randn(batch_size, seq_len, hidden_dim)

    output = rms_norm_module.forward(x)
    assert output.shape == x.shape

    stats = rms_norm_module.get_stats(x)
    assert "input_rms" in stats
    assert "output_mean" in stats


def test_normalization_statistics():
    """Test that normalization produces expected statistics."""
    hidden_dim = 64
    layer_norm_module = LayerNorm(hidden_dim)

    batch_size, seq_len = 2, 10
    x = np.random.randn(batch_size, seq_len, hidden_dim)

    output = layer_norm_module.forward(x)

    # Check that output has reasonable statistics
    assert np.abs(np.mean(output)) < 1.0
    assert np.std(output) > 0.5  # Should have some variance
