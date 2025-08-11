"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from typing import Dict, Any
import tempfile
import shutil
from pathlib import Path

from transformerlab.core.tokenizer import load_corpus, CharTokenizer


@pytest.fixture
def sample_text():
    """Small sample text for testing."""
    return "Hello world! This is a test. How are you? Fine, thanks."


@pytest.fixture
def sample_tokenizer(sample_text):
    """Tokenizer with sample vocabulary."""
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(sample_text)
    return tokenizer


@pytest.fixture
def sample_tokens(sample_tokenizer, sample_text):
    """Tokenized sample text."""
    return sample_tokenizer.encode(sample_text)


@pytest.fixture
def tiny_shakespeare():
    """Load the tiny Shakespeare dataset."""
    text, tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")
    return text, tokenizer


@pytest.fixture
def model_config():
    """Standard model configuration for testing."""
    return {
        "vocab_size": 50,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "max_seq_len": 100,
        "norm_type": "LayerNorm",
        "activation_type": "ReLU",
        "residual_type": "Pre-LN",
        "pos_encoding_type": "Sinusoidal",
    }


@pytest.fixture
def training_data():
    """Sample training data batch."""
    batch_size, seq_len = 2, 10
    vocab_size = 50
    
    # Create simple sequential data
    x = np.random.randint(1, vocab_size, (batch_size, seq_len))
    targets = np.roll(x, -1, axis=1)  # Shift by one for next token prediction
    
    return x, targets


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def browser():
    """Playwright browser instance for UI tests."""
    pytest_plugins = ["playwright.pytest_plugin"]
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()
    except ImportError:
        pytest.skip("Playwright not available")


# Performance test configurations
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "min_rounds": 3,
        "max_time": 1.0,
        "warmup": True,
    }


# Backend comparison fixtures
@pytest.fixture(params=["numpy", "python", "torch"])
def backend_name(request):
    """Parametrize tests across all backends."""
    return request.param


@pytest.fixture
def skip_if_no_torch():
    """Skip test if PyTorch is not available."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark")
    config.addinivalue_line("markers", "backend: mark test as backend-specific")
    config.addinivalue_line("markers", "ui: mark test as UI test")


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests."""
    slow_keywords = ["train", "benchmark", "performance", "large", "integration"]
    
    for item in items:
        # Mark tests with slow keywords
        if any(keyword in item.name.lower() for keyword in slow_keywords):
            item.add_marker(pytest.mark.slow)
        
        # Mark UI tests
        if "ui" in item.name.lower() or "streamlit" in item.name.lower():
            item.add_marker(pytest.mark.ui)
        
        # Mark integration tests  
        if "integration" in item.name.lower() or item.fspath.basename.startswith("test_integration"):
            item.add_marker(pytest.mark.integration)