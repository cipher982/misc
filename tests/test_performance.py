"""
Performance benchmarking tests for different backends.
"""

import pytest
import numpy as np
import time
from typing import Dict, Any, List
import psutil
import gc

from transformerlab.core.tokenizer import load_corpus


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance comparison tests across backends."""

    def setup_method(self):
        """Setup for each test method."""
        # Ensure clean memory state
        gc.collect()
        
        # Load consistent test data
        self.text, self.tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")
        self.tokens = self.tokenizer.encode(self.text)
        
        # Test configurations
        self.configs = [
            {"hidden_dim": 32, "num_layers": 1, "seq_len": 10, "batch_size": 1},
            {"hidden_dim": 64, "num_layers": 2, "seq_len": 20, "batch_size": 2},
            {"hidden_dim": 128, "num_layers": 3, "seq_len": 30, "batch_size": 4},
        ]

    def create_model(self, backend: str, config: Dict[str, Any]):
        """Create model for specified backend."""
        base_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "num_heads": 4,
            "ff_dim": config["hidden_dim"] * 2,
            "norm_type": "LayerNorm",
            "activation_type": "ReLU",
            "residual_type": "Pre-LN",
            "pos_encoding_type": "Sinusoidal",
        }
        base_config.update(config)
        
        if backend == "numpy":
            from transformerlab.core.transformer import Transformer
            return Transformer(**base_config)
        elif backend == "python":
            # Will implement after creating Python backend
            pytest.skip("Python backend not yet implemented")
        elif backend == "torch":
            # Will implement after creating PyTorch backend  
            pytest.skip("PyTorch backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def create_batch(self, config: Dict[str, Any]):
        """Create training batch."""
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        
        x = np.zeros((batch_size, seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, seq_len), dtype=np.int32)
        
        for i in range(batch_size):
            start_idx = np.random.randint(0, len(self.tokens) - seq_len - 1)
            x[i] = self.tokens[start_idx:start_idx + seq_len]
            targets[i] = self.tokens[start_idx + 1:start_idx + seq_len + 1]
            
        return x, targets

    @pytest.mark.parametrize("backend", ["numpy"])  # Will add others later
    @pytest.mark.parametrize("config_idx", [0, 1, 2])
    def test_forward_pass_performance(self, benchmark, backend, config_idx):
        """Benchmark forward pass performance."""
        config = self.configs[config_idx]
        model = self.create_model(backend, config)
        x, targets = self.create_batch(config)
        
        def forward_pass():
            return model.forward(x, targets)
        
        result = benchmark(forward_pass)
        
        # Verify the forward pass worked
        logits, stats = result
        assert logits is not None
        assert stats["loss"] is not None

    @pytest.mark.parametrize("backend", ["numpy"])
    def test_training_step_performance(self, benchmark, backend):
        """Benchmark training step performance."""
        config = self.configs[1]  # Medium size
        model = self.create_model(backend, config)
        x, targets = self.create_batch(config)
        
        from transformerlab.core.optimizer import SGDOptimizer
        optimizer = SGDOptimizer(learning_rate=0.01)
        
        def training_step():
            return model.train_step(x, targets, optimizer)
        
        loss = benchmark(training_step)
        assert isinstance(loss, (int, float))
        assert not np.isnan(loss)

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        process = psutil.Process()
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory
        execution_time = end_time - start_time
        
        return result, {
            "memory_used_mb": memory_used,
            "execution_time_s": execution_time,
            "peak_memory_mb": peak_memory,
        }

    @pytest.mark.parametrize("backend", ["numpy"])
    def test_memory_usage(self, backend):
        """Test memory usage across different model sizes."""
        results = {}
        
        for i, config in enumerate(self.configs):
            model = self.create_model(backend, config)
            x, targets = self.create_batch(config)
            
            # Test forward pass memory
            _, forward_stats = self.measure_memory_usage(
                model.forward, x, targets
            )
            
            # Test training step memory
            from transformerlab.core.optimizer import SGDOptimizer
            optimizer = SGDOptimizer(learning_rate=0.01)
            
            _, training_stats = self.measure_memory_usage(
                model.train_step, x, targets, optimizer
            )
            
            results[f"config_{i}"] = {
                "model_params": len(model.get_parameters()),
                "forward_memory_mb": forward_stats["memory_used_mb"],
                "forward_time_s": forward_stats["execution_time_s"],
                "training_memory_mb": training_stats["memory_used_mb"],
                "training_time_s": training_stats["execution_time_s"],
            }
        
        # Basic assertions - memory should increase with model size
        assert results["config_1"]["forward_memory_mb"] >= results["config_0"]["forward_memory_mb"]
        assert results["config_2"]["forward_memory_mb"] >= results["config_1"]["forward_memory_mb"]
        
        # Print results for manual inspection
        print("\nMemory Usage Results:")
        for config_name, stats in results.items():
            print(f"{config_name}: {stats}")

    @pytest.mark.slow
    def test_convergence_comparison(self):
        """Compare convergence rates across backends."""
        config = self.configs[1]  # Medium size
        num_steps = 20
        
        results = {}
        
        # Test numpy backend
        model_numpy = self.create_model("numpy", config)
        from transformerlab.core.optimizer import SGDOptimizer
        optimizer_numpy = SGDOptimizer(learning_rate=0.01)
        
        x, targets = self.create_batch(config)
        losses_numpy = []
        
        start_time = time.time()
        for step in range(num_steps):
            loss = model_numpy.train_step(x, targets, optimizer_numpy)
            losses_numpy.append(loss)
        numpy_time = time.time() - start_time
        
        results["numpy"] = {
            "losses": losses_numpy,
            "total_time": numpy_time,
            "final_loss": losses_numpy[-1],
            "loss_reduction": losses_numpy[0] - losses_numpy[-1],
        }
        
        # Verify convergence (loss should generally decrease)
        assert results["numpy"]["loss_reduction"] >= 0, "Model should be learning"
        
        print("\nConvergence Results:")
        for backend, stats in results.items():
            print(f"{backend}: final_loss={stats['final_loss']:.4f}, "
                  f"reduction={stats['loss_reduction']:.4f}, "
                  f"time={stats['total_time']:.2f}s")

    def teardown_method(self):
        """Cleanup after each test method."""
        gc.collect()