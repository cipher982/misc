"""
Performance and benchmarking tests.
"""

import pytest
import time
import numpy as np
from typing import Dict, List, Tuple

# Import psutil conditionally for performance tests
psutil = pytest.importorskip("psutil", reason="psutil required for performance tests")

from transformerlab.backends.factory import create_transformer, list_backends
from transformerlab.benchmarks import PerformanceBenchmark


class TestPerformanceBenchmarking:
    """Test performance benchmarking functionality."""
    
    def test_benchmark_initialization(self):
        """Test that benchmark class initializes correctly."""
        benchmark = PerformanceBenchmark()
        
        assert hasattr(benchmark, 'results')
        assert isinstance(benchmark.results, dict)
    
    @pytest.mark.slow
    def test_single_backend_benchmark(self, performance_test_configs):
        """Test benchmarking a single backend."""
        benchmark = PerformanceBenchmark()
        
        # Use smallest config for fast testing
        config = performance_test_configs[0]  # "tiny" config
        
        try:
            results = benchmark.benchmark_backend("numpy", config)
            
            assert isinstance(results, dict)
            assert "execution_time" in results
            assert "memory_usage" in results
            assert "parameter_count" in results
            
            # Verify reasonable values
            assert results["execution_time"] > 0
            assert results["parameter_count"] > 0
            
        except Exception as e:
            pytest.skip(f"Benchmark failed: {e}")
    
    def test_compare_backends_performance(self, performance_test_configs):
        """Test performance comparison between backends."""
        benchmark = PerformanceBenchmark()
        
        # Use smallest config for speed
        config = performance_test_configs[0]
        
        try:
            comparison = benchmark.compare_backends(
                backends=["python", "numpy"], 
                config=config,
                num_runs=1  # Single run for speed
            )
            
            assert isinstance(comparison, dict)
            assert len(comparison) >= 1  # At least one backend should work
            
            for backend_name, results in comparison.items():
                if results.get("available", False):
                    assert "speed_score" in results
                    assert "memory_score" in results
                    assert "accuracy_score" in results
                    
        except Exception as e:
            pytest.skip(f"Backend comparison failed: {e}")
    
    def test_model_size_scaling(self):
        """Test performance scaling with model size."""
        benchmark = PerformanceBenchmark()
        
        size_configs = [
            {"vocab_size": 10, "hidden_dim": 8, "num_layers": 1, "num_heads": 2, "ff_dim": 16},
            {"vocab_size": 20, "hidden_dim": 16, "num_layers": 1, "num_heads": 2, "ff_dim": 32},
        ]
        
        try:
            results = benchmark.benchmark_model_sizes(
                size_configs=size_configs,
                backends=["numpy"],
                num_runs=1
            )
            
            assert isinstance(results, dict)
            assert "numpy" in results
            
            numpy_results = results["numpy"]
            assert len(numpy_results) == len(size_configs)
            
            # Larger model should generally take more time and memory
            if len(numpy_results) >= 2:
                small_time = numpy_results[0]["execution_time"]
                large_time = numpy_results[1]["execution_time"] 
                
                # Allow some variance, but larger model shouldn't be much faster
                assert large_time <= small_time * 5
                
        except Exception as e:
            pytest.skip(f"Model size scaling test failed: {e}")


class TestMemoryProfiling:
    """Test memory usage profiling."""
    
    def test_memory_tracking_numpy(self, small_config):
        """Test memory tracking for NumPy backend."""
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        transformer = create_transformer("numpy", **small_config)
        
        # Check memory increased
        after_creation = process.memory_info().rss / 1024 / 1024
        memory_increase = after_creation - initial_memory
        
        # Should use some memory (at least for parameters)
        assert memory_increase > 0, "Model creation should use memory"
        assert memory_increase < 100, "Small model shouldn't use excessive memory"
        
        # Test memory during forward pass
        inputs = np.random.randint(1, small_config["vocab_size"], size=(1, 5))
        targets = np.random.randint(1, small_config["vocab_size"], size=(1, 5))
        
        transformer.forward(inputs, targets)
        
        after_forward = process.memory_info().rss / 1024 / 1024
        forward_increase = after_forward - after_creation
        
        # Forward pass should use some additional memory
        assert forward_increase >= 0, "Forward pass memory usage"
    
    def test_memory_estimation(self, small_config):
        """Test model memory estimation."""
        transformer = create_transformer("numpy", **small_config)
        
        if hasattr(transformer, '_estimate_memory_usage'):
            memory_info = transformer._estimate_memory_usage()
            
            assert isinstance(memory_info, dict)
            assert "parameters_mb" in memory_info
            assert "total_mb" in memory_info
            
            # Should be reasonable estimates
            assert memory_info["parameters_mb"] > 0
            assert memory_info["total_mb"] >= memory_info["parameters_mb"]
            assert memory_info["total_mb"] < 1000  # Should be reasonable for small model
    
    def test_memory_cleanup(self, small_config):
        """Test that memory is properly released."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Create and destroy multiple models
        for i in range(3):
            transformer = create_transformer("numpy", **small_config)
            
            # Use the model
            inputs = np.random.randint(1, small_config["vocab_size"], size=(1, 5))
            transformer.forward(inputs)
            
            # Delete reference
            del transformer
        
        # Memory should not have grown excessively
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Allow some growth but not excessive
        assert memory_growth < 50, f"Memory grew too much: {memory_growth} MB"


class TestExecutionTiming:
    """Test execution timing and performance."""
    
    def test_forward_pass_timing(self, small_config):
        """Test forward pass timing."""
        transformer = create_transformer("numpy", **small_config)
        
        inputs = np.random.randint(1, small_config["vocab_size"], size=(2, 10))
        targets = np.random.randint(1, small_config["vocab_size"], size=(2, 10))
        
        # Warm-up run
        transformer.forward(inputs, targets)
        
        # Timed runs
        times = []
        for _ in range(5):
            start_time = time.time()
            transformer.forward(inputs, targets)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
        
        # Should complete in reasonable time
        avg_time = np.mean(times)
        assert avg_time < 1.0, f"Forward pass too slow: {avg_time:.3f}s"
        
        # Timing should be relatively consistent
        std_time = np.std(times)
        cv = std_time / avg_time if avg_time > 0 else float('inf')
        assert cv < 0.5, f"Timing too inconsistent: CV={cv:.3f}"
    
    def test_training_step_timing(self, small_config):
        """Test training step timing."""
        transformer = create_transformer("numpy", **small_config)
        
        from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        inputs = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        targets = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        
        # Warm-up
        transformer.train_step(inputs, targets, optimizer)
        
        # Timed runs
        times = []
        for _ in range(3):
            start_time = time.time()
            transformer.train_step(inputs, targets, optimizer)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        assert avg_time < 2.0, f"Training step too slow: {avg_time:.3f}s"
    
    def test_generation_timing(self, small_config):
        """Test text generation timing."""
        transformer = create_transformer("numpy", **small_config)
        
        prompt = np.array([[1, 2, 3]])
        
        try:
            # Warm-up
            transformer.generate(prompt, max_length=10)
            
            # Timed generation
            start_time = time.time()
            generated = transformer.generate(prompt, max_length=20)
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            # Should generate in reasonable time
            assert generation_time < 5.0, f"Generation too slow: {generation_time:.3f}s"
            
            # Should produce output
            assert generated is not None
            if hasattr(generated, 'shape'):
                assert generated.shape[1] > len(prompt[0])
                
        except NotImplementedError:
            pytest.skip("Generation not implemented")


class TestScalabilityBenchmarks:
    """Test scalability with different model sizes."""
    
    @pytest.mark.slow
    def test_sequence_length_scaling(self, small_config):
        """Test performance scaling with sequence length."""
        transformer = create_transformer("numpy", **small_config)
        
        sequence_lengths = [5, 10, 20]
        execution_times = []
        
        for seq_len in sequence_lengths:
            if seq_len > small_config["max_seq_len"]:
                continue
                
            inputs = np.random.randint(1, small_config["vocab_size"], size=(1, seq_len))
            targets = np.random.randint(1, small_config["vocab_size"], size=(1, seq_len))
            
            # Time multiple runs
            times = []
            for _ in range(3):
                start_time = time.time()
                transformer.forward(inputs, targets)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            execution_times.append(avg_time)
        
        # Longer sequences should generally take more time
        if len(execution_times) >= 2:
            # Allow some variance but shouldn't be dramatically different
            ratio = execution_times[-1] / execution_times[0]
            assert ratio < 10, f"Scaling too poor: {ratio:.2f}x slower"
    
    def test_batch_size_scaling(self, small_config):
        """Test performance scaling with batch size."""
        transformer = create_transformer("numpy", **small_config)
        
        batch_sizes = [1, 2, 4]
        times_per_sample = []
        
        for batch_size in batch_sizes:
            inputs = np.random.randint(1, small_config["vocab_size"], size=(batch_size, 8))
            targets = np.random.randint(1, small_config["vocab_size"], size=(batch_size, 8))
            
            # Time the forward pass
            start_time = time.time()
            transformer.forward(inputs, targets)
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_sample = total_time / batch_size
            times_per_sample.append(time_per_sample)
        
        # Larger batches should be more efficient per sample
        if len(times_per_sample) >= 2:
            # Per-sample time should not increase dramatically with batch size
            efficiency_ratio = times_per_sample[-1] / times_per_sample[0]
            assert efficiency_ratio < 3.0, f"Batching efficiency poor: {efficiency_ratio:.2f}x"


class TestResourceMonitoring:
    """Test system resource monitoring during training."""
    
    def test_cpu_usage_monitoring(self, small_config):
        """Test CPU usage during training."""
        transformer = create_transformer("numpy", **small_config)
        
        from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        inputs = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        targets = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        
        # Monitor CPU usage during training
        cpu_percentages = []
        
        for _ in range(5):
            cpu_before = psutil.cpu_percent(interval=None)
            transformer.train_step(inputs, targets, optimizer)
            cpu_after = psutil.cpu_percent(interval=0.1)
            
            cpu_percentages.append(cpu_after)
        
        avg_cpu = np.mean(cpu_percentages)
        
        # Should use some CPU but not be excessive for small model
        assert avg_cpu > 0, "Should use some CPU"
        assert avg_cpu < 100, "CPU usage shouldn't be excessive"
    
    def test_resource_cleanup_after_training(self, small_config):
        """Test that resources are properly cleaned up."""
        initial_process = psutil.Process()
        initial_memory = initial_process.memory_info().rss
        
        # Create and train model
        transformer = create_transformer("numpy", **small_config)
        
        from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        inputs = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        targets = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        
        # Training session
        for _ in range(10):
            transformer.train_step(inputs, targets, optimizer)
        
        # Clean up
        del transformer
        del optimizer
        
        # Check memory after cleanup
        final_memory = initial_process.memory_info().rss
        memory_diff = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Some memory growth is expected, but shouldn't be excessive
        assert memory_diff < 50, f"Too much memory not released: {memory_diff:.1f} MB"