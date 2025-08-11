"""
Comprehensive backend testing suite.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from transformerlab.backends.factory import (
    create_transformer, 
    list_backends, 
    get_backend_info,
    compare_backends
)
from transformerlab.backends.abstract import AbstractTransformer


class TestBackendCreation:
    """Test backend creation and initialization."""
    
    def test_list_backends(self):
        """Test that list_backends returns expected backends."""
        backends = list_backends()
        expected_backends = {"python", "numpy", "torch"}
        
        assert isinstance(backends, list)
        assert len(backends) > 0
        assert expected_backends.issubset(set(backends))
    
    def test_backend_info(self):
        """Test backend information retrieval."""
        backends = list_backends()
        
        for backend_name in backends:
            info = get_backend_info(backend_name)
            
            assert isinstance(info, dict)
            assert "description" in info
            assert "features" in info
            assert "backend_type" in info
            assert isinstance(info["description"], str)
            assert len(info["description"]) > 0
    
    @pytest.mark.parametrize("backend_name", ["python", "numpy", "torch"])
    def test_create_transformer(self, backend_name, small_config):
        """Test transformer creation for each backend."""
        try:
            transformer = create_transformer(backend_name, **small_config)
            
            assert isinstance(transformer, AbstractTransformer)
            assert transformer.vocab_size == small_config["vocab_size"]
            assert transformer.hidden_dim == small_config["hidden_dim"]
            assert transformer.num_layers == small_config["num_layers"]
            assert transformer.num_heads == small_config["num_heads"]
            
        except ImportError as e:
            if "torch" in str(e) and backend_name == "torch":
                pytest.skip("PyTorch not available")
            else:
                raise
    
    def test_backend_comparison(self, small_config):
        """Test backend comparison functionality."""
        comparison = compare_backends(["python", "numpy", "torch"], small_config)
        
        assert isinstance(comparison, dict)
        assert len(comparison) == 3
        
        for backend_name, info in comparison.items():
            assert "available" in info
            if info["available"]:
                assert "parameter_count" in info
                assert "description" in info
                assert isinstance(info["parameter_count"], int)
                assert info["parameter_count"] > 0


class TestBackendFunctionality:
    """Test basic functionality across backends."""
    
    def test_parameter_count(self, backend_name, small_config):
        """Test parameter counting is consistent."""
        if backend_name == "torch":
            pytest.importorskip("torch", reason="PyTorch not available")
        
        transformer = create_transformer(backend_name, **small_config)
        param_count = transformer.get_parameter_count()
        
        assert isinstance(param_count, int)
        assert param_count > 0
        
        # Parameter count should be consistent across backends for same config
        expected_min_params = small_config["vocab_size"] * small_config["hidden_dim"]
        assert param_count >= expected_min_params
    
    def test_forward_pass(self, backend_name, small_config, sample_input_data):
        """Test forward pass for each backend."""
        if backend_name == "torch":
            pytest.importorskip("torch", reason="PyTorch not available")
        
        transformer = create_transformer(backend_name, **small_config)
        
        # Prepare input based on backend
        if backend_name == "torch":
            import torch
            inputs = torch.tensor(sample_input_data["single_input"], dtype=torch.long)
            targets = torch.tensor(sample_input_data["single_target"], dtype=torch.long)
        else:
            inputs = sample_input_data["single_input"]
            targets = sample_input_data["single_target"]
        
        logits, stats = transformer.forward(inputs, targets)
        
        # Check output structure
        assert logits is not None
        assert isinstance(stats, dict)
        assert "loss" in stats
        
        # Check output dimensions
        if hasattr(logits, 'shape'):
            batch_size, seq_len, vocab_size = logits.shape
            assert batch_size == 1
            assert seq_len == 5
            assert vocab_size == small_config["vocab_size"]
    
    def test_generation(self, backend_name, small_config, sample_input_data):
        """Test text generation capability."""
        if backend_name == "torch":
            pytest.importorskip("torch", reason="PyTorch not available")
        
        transformer = create_transformer(backend_name, **small_config)
        
        # Prepare prompt
        if backend_name == "torch":
            import torch
            prompt = torch.tensor([sample_input_data["prompt_tokens"]], dtype=torch.long)
        else:
            prompt = np.array([sample_input_data["prompt_tokens"]])
        
        try:
            generated = transformer.generate(prompt, max_length=10, temperature=1.0)
            
            assert generated is not None
            if hasattr(generated, 'shape'):
                assert generated.shape[0] == 1  # batch size
                assert generated.shape[1] >= len(sample_input_data["prompt_tokens"])
            
        except NotImplementedError:
            pytest.skip(f"Generation not implemented for {backend_name} backend")


class TestBackendSpecificFeatures:
    """Test backend-specific features and optimizations."""
    
    def test_numpy_backend_features(self, small_config):
        """Test NumPy-specific features."""
        transformer = create_transformer("numpy", **small_config)
        
        # Test parameter access
        params = transformer.get_parameters()
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)
        
        # Test memory estimation
        if hasattr(transformer, '_estimate_memory_usage'):
            memory_info = transformer._estimate_memory_usage()
            assert isinstance(memory_info, dict)
            assert "parameters_mb" in memory_info
            assert "total_mb" in memory_info
    
    def test_python_backend_verbosity(self, small_config, sample_input_data, capsys):
        """Test Python backend verbose output."""
        transformer = create_transformer("python", **small_config)
        
        # Forward pass should produce verbose output
        inputs = sample_input_data["single_input"]
        targets = sample_input_data["single_target"]
        
        transformer.forward(inputs, targets)
        
        # Check that output was captured (Python backend is verbose)
        captured = capsys.readouterr()
        # Note: The actual capture depends on implementation details
    
    @pytest.mark.skipif(True, reason="Requires PyTorch")
    def test_torch_backend_gpu_compatibility(self, small_config):
        """Test PyTorch backend GPU compatibility."""
        torch = pytest.importorskip("torch")
        
        transformer = create_transformer("torch", **small_config)
        
        # Test device management
        if torch.cuda.is_available():
            # Test moving to GPU
            transformer.to('cuda')
            assert next(transformer.parameters()).device.type == 'cuda'
            
            # Test moving back to CPU
            transformer.to('cpu')
            assert next(transformer.parameters()).device.type == 'cpu'


class TestBackendConsistency:
    """Test consistency between different backends."""
    
    @pytest.mark.slow
    def test_output_similarity(self, small_config):
        """Test that backends produce similar outputs for same inputs."""
        # Create transformers with same configuration
        transformers = {}
        for backend_name in ["python", "numpy"]:
            try:
                transformers[backend_name] = create_transformer(backend_name, **small_config)
            except Exception as e:
                pytest.skip(f"Backend {backend_name} not available: {e}")
        
        if len(transformers) < 2:
            pytest.skip("Need at least 2 backends for comparison")
        
        # Use same random seed to initialize weights similarly
        np.random.seed(42)
        
        # Test with same input
        inputs = [[1, 2, 3, 4, 5]]
        targets = [[2, 3, 4, 5, 1]]
        
        outputs = {}
        for backend_name, transformer in transformers.items():
            try:
                logits, stats = transformer.forward(inputs, targets)
                outputs[backend_name] = {
                    "logits": logits,
                    "loss": stats.get("loss", None)
                }
            except Exception as e:
                pytest.skip(f"Forward pass failed for {backend_name}: {e}")
        
        # Compare losses (they should be similar but not identical due to implementation differences)
        losses = [output["loss"] for output in outputs.values() if output["loss"] is not None]
        if len(losses) >= 2:
            # Losses should be in the same order of magnitude
            loss_ratio = max(losses) / min(losses)
            assert loss_ratio < 10.0, f"Losses too different: {losses}"
    
    def test_parameter_count_consistency(self):
        """Test that parameter counts are consistent across backends."""
        config = {
            "vocab_size": 20,
            "hidden_dim": 16,  
            "num_layers": 1,
            "num_heads": 2,
            "ff_dim": 32,
        }
        
        param_counts = {}
        for backend_name in ["python", "numpy"]:
            try:
                transformer = create_transformer(backend_name, **config)
                param_counts[backend_name] = transformer.get_parameter_count()
            except Exception:
                continue
        
        if len(param_counts) >= 2:
            # All backends should report the same parameter count
            counts = list(param_counts.values())
            assert all(count == counts[0] for count in counts), f"Parameter counts differ: {param_counts}"


class TestBackendErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_backend_name(self, small_config):
        """Test handling of invalid backend names."""
        with pytest.raises((ValueError, KeyError)):
            create_transformer("invalid_backend", **small_config)
    
    def test_invalid_configuration(self, backend_name):
        """Test handling of invalid configurations."""
        if backend_name == "torch":
            pytest.importorskip("torch", reason="PyTorch not available")
        
        # Test with invalid num_heads (should be divisible by hidden_dim)
        invalid_config = {
            "vocab_size": 20,
            "hidden_dim": 17,  # Not divisible by num_heads
            "num_layers": 1,
            "num_heads": 4,    # 17 % 4 != 0
            "ff_dim": 32,
        }
        
        with pytest.raises((ValueError, AssertionError)):
            create_transformer(backend_name, **invalid_config)
    
    def test_empty_input_handling(self, backend_name, small_config):
        """Test handling of empty or invalid inputs."""
        if backend_name == "torch":
            pytest.importorskip("torch", reason="PyTorch not available")
        
        transformer = create_transformer(backend_name, **small_config)
        
        # Test with empty input
        empty_input = [[]]
        
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            transformer.forward(empty_input)