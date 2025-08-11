"""
Pytest configuration and fixtures for transformer testing.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Any

from transformerlab.backends.factory import create_transformer, list_backends
from transformerlab.core.tokenizer import CharTokenizer


@pytest.fixture
def small_config():
    """Small transformer configuration for fast testing."""
    return {
        "vocab_size": 20,
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 2,
        "ff_dim": 32,
        "max_seq_len": 50,
        "norm_type": "LayerNorm",
        "activation_type": "ReLU",
        "residual_type": "Pre-LN",
        "pos_encoding_type": "Sinusoidal",
    }


@pytest.fixture
def medium_config():
    """Medium transformer configuration for thorough testing."""
    return {
        "vocab_size": 50,
        "hidden_dim": 64,
        "num_layers": 3,
        "num_heads": 4,
        "ff_dim": 128,
        "max_seq_len": 100,
        "norm_type": "LayerNorm",
        "activation_type": "GeLU",
        "residual_type": "Pre-LN",
        "pos_encoding_type": "Sinusoidal",
    }


@pytest.fixture(params=["python", "numpy", "torch"])
def backend_name(request):
    """Parameterized fixture for testing all backends."""
    return request.param


@pytest.fixture
def available_backends():
    """List of available backends."""
    return list_backends()


@pytest.fixture
def sample_input_data():
    """Sample input data for testing."""
    return {
        "input_tokens": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]],
        "target_tokens": [[2, 3, 4, 5, 1], [3, 4, 5, 1, 2]],
        "single_input": [[1, 2, 3, 4, 5]],
        "single_target": [[2, 3, 4, 5, 1]],
        "prompt_tokens": [1, 2, 3],
    }


@pytest.fixture
def simple_tokenizer():
    """Simple character tokenizer for testing."""
    # Create a small vocabulary with sample text
    text = "abcdefghijklmnopqrstuvwxyz hello world test"
    tokenizer = CharTokenizer(text)
    return tokenizer


@pytest.fixture
def temp_model_dir():
    """Temporary directory for saving/loading models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def trained_transformer(backend_name, small_config, sample_input_data):
    """Fixture that provides a transformer with some training."""
    transformer = create_transformer(backend_name, **small_config)
    
    # Train for a few steps if backend supports it
    if hasattr(transformer, 'train_step'):
        try:
            # Create simple optimizer
            if backend_name == 'numpy':
                from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
                optimizer = create_numpy_optimizer('sgd', learning_rate=0.01)
            else:
                # For other backends, we'll skip training
                return transformer
            
            # Train for 2 steps
            for _ in range(2):
                inputs = np.array(sample_input_data["single_input"])
                targets = np.array(sample_input_data["single_target"])
                transformer.train_step(inputs, targets, optimizer)
        except Exception:
            # If training fails, just return the initialized transformer
            pass
    
    return transformer


@pytest.fixture
def performance_test_configs():
    """Different configurations for performance testing."""
    return [
        {
            "name": "tiny",
            "vocab_size": 10,
            "hidden_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "ff_dim": 16,
        },
        {
            "name": "small", 
            "vocab_size": 20,
            "hidden_dim": 16,
            "num_layers": 2,
            "num_heads": 2,
            "ff_dim": 32,
        },
        {
            "name": "medium",
            "vocab_size": 50,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 64,
        },
    ]


class TestDataGenerator:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_sequence_data(vocab_size: int, seq_len: int, batch_size: int = 1) -> dict[str, np.ndarray]:
        """Create random sequence data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        inputs = np.random.randint(1, vocab_size, size=(batch_size, seq_len))
        targets = np.random.randint(1, vocab_size, size=(batch_size, seq_len))
        
        return {
            "inputs": inputs,
            "targets": targets,
        }
    
    @staticmethod
    def create_text_data(tokenizer, text: str = "hello world test") -> dict[str, np.ndarray]:
        """Create tokenized text data."""
        tokens = tokenizer.encode(text)
        
        return {
            "text": text,
            "tokens": tokens,
            "input_array": np.array([tokens[:-1]]),
            "target_array": np.array([tokens[1:]]),
        }


@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    return TestDataGenerator()


# Mark integration tests
@pytest.fixture
def integration_test_marker():
    """Marker for integration tests."""
    return pytest.mark.integration


# Mark slow tests
@pytest.fixture  
def slow_test_marker():
    """Marker for slow tests."""
    return pytest.mark.slow


# Skip if backend not available
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow tests") 
    config.addinivalue_line("markers", "backend_specific: marks tests that are backend-specific")


def skip_if_backend_unavailable(backend_name: str):
    """Skip test if backend is not available."""
    try:
        create_transformer(backend_name, vocab_size=10, hidden_dim=8, num_layers=1, num_heads=2, ff_dim=16)
        return False
    except Exception:
        return True


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Seed random number generators for reproducible tests
    np.random.seed(42)
    
    # You could add more environment setup here
    yield
    
    # Cleanup after test if needed