#!/usr/bin/env python3
"""
Performance benchmarking for different transformer backends.

This script compares the performance characteristics of NumPy, Python, and PyTorch
implementations across different model sizes and operations.
"""

import time
import tracemalloc
import json
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict

from transformerlab.backends.factory import create_transformer, list_backends


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    operation: str
    model_config: Dict[str, Any]
    forward_time_ms: float
    backward_time_ms: Optional[float]
    training_time_ms: Optional[float]
    memory_usage_mb: float
    parameter_count: int
    accuracy_loss: float
    success: bool
    error_message: Optional[str] = None


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.backends = list_backends()
    
    def benchmark_model_sizes(
        self,
        size_configs: List[Dict[str, Any]],
        backends: Optional[List[str]] = None,
        num_runs: int = 3
    ) -> List[BenchmarkResult]:
        """Benchmark different model sizes across backends."""
        backends = backends or self.backends
        all_results = []
        
        print("🚀 Performance Benchmarking Suite")
        print("=" * 60)
        
        for size_name, config in size_configs:
            print(f"\n📊 Testing {size_name} model:")
            print(f"   Config: {config}")
            
            for backend in backends:
                print(f"\n  🔧 Backend: {backend}")
                
                # Run multiple times for averaging
                run_results = []
                for run in range(num_runs):
                    result = self._benchmark_single_run(backend, config, run + 1)
                    run_results.append(result)
                    
                    if result.success:
                        print(f"    Run {run + 1}: Forward={result.forward_time_ms:.2f}ms, "
                              f"Memory={result.memory_usage_mb:.1f}MB")
                    else:
                        print(f"    Run {run + 1}: ❌ {result.error_message}")
                
                # Average successful runs
                successful_runs = [r for r in run_results if r.success]
                if successful_runs:
                    averaged_result = self._average_results(successful_runs, backend, config)
                    all_results.append(averaged_result)
                    
                    print(f"    ✅ Average: Forward={averaged_result.forward_time_ms:.2f}ms, "
                          f"Memory={averaged_result.memory_usage_mb:.1f}MB")
                else:
                    # Add failed result
                    failed_result = run_results[0]  # Use first failed result
                    all_results.append(failed_result)
                    print(f"    ❌ All runs failed")
        
        self.results.extend(all_results)
        return all_results
    
    def _benchmark_single_run(
        self, 
        backend: str, 
        config: Dict[str, Any], 
        run_number: int
    ) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        try:
            # Start memory tracking
            tracemalloc.start()
            
            # Create transformer
            transformer = create_transformer(backend, **config)
            
            # Prepare test data
            batch_size = 2
            seq_len = min(10, config.get('max_seq_len', 10))
            vocab_size = config['vocab_size']
            
            # Generate random input
            np.random.seed(42 + run_number)  # Consistent but varied seeds
            sample_input = np.random.randint(1, vocab_size - 1, size=(batch_size, seq_len))
            sample_targets = np.random.randint(1, vocab_size - 1, size=(batch_size, seq_len))
            
            # Convert to backend format if needed
            if backend == "torch":
                import torch
                x = torch.tensor(sample_input, dtype=torch.long)
                targets = torch.tensor(sample_targets, dtype=torch.long)
            else:
                x = sample_input.tolist()
                targets = sample_targets.tolist()
            
            # Benchmark forward pass
            start_time = time.perf_counter()
            logits, stats = transformer.forward(x, targets)
            forward_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Extract loss
            loss = stats.get('loss', 0.0)
            
            # Benchmark backward pass (if supported)
            backward_time = None
            try:
                start_time = time.perf_counter()
                if backend == "torch":
                    _, gradients = transformer.backward(logits, targets)
                else:
                    _, gradients = transformer.backward(logits, targets)
                backward_time = (time.perf_counter() - start_time) * 1000
            except Exception as e:
                print(f"    Warning: Backward pass failed: {e}")
            
            # Benchmark training step (if optimizer available)
            training_time = None
            try:
                if backend == "numpy":
                    from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
                    optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
                elif backend == "python":
                    from transformerlab.backends.python_backend.optimizer import create_python_optimizer  
                    optimizer = create_python_optimizer("sgd", learning_rate=0.01)
                elif backend == "torch":
                    from transformerlab.backends.torch_backend.optimizer import create_torch_optimizer
                    params = transformer.get_pytorch_parameters()
                    optimizer = create_torch_optimizer("sgd", params, learning_rate=0.01)
                
                start_time = time.perf_counter()
                loss_value = transformer.train_step(x, targets, optimizer)
                training_time = (time.perf_counter() - start_time) * 1000
            except Exception as e:
                print(f"    Warning: Training step failed: {e}")
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage_mb = peak / 1024 / 1024  # Convert to MB
            
            # Get parameter count
            param_count = transformer.get_parameter_count()
            
            return BenchmarkResult(
                backend=backend,
                operation="forward_backward_training",
                model_config=config,
                forward_time_ms=forward_time,
                backward_time_ms=backward_time,
                training_time_ms=training_time,
                memory_usage_mb=memory_usage_mb,
                parameter_count=param_count,
                accuracy_loss=float(loss) if isinstance(loss, (int, float, np.number)) else 0.0,
                success=True
            )
            
        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                backend=backend,
                operation="forward_backward_training",
                model_config=config,
                forward_time_ms=0.0,
                backward_time_ms=None,
                training_time_ms=None,
                memory_usage_mb=0.0,
                parameter_count=0,
                accuracy_loss=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _average_results(
        self, 
        results: List[BenchmarkResult], 
        backend: str, 
        config: Dict[str, Any]
    ) -> BenchmarkResult:
        """Average multiple benchmark results."""
        n = len(results)
        
        avg_forward = sum(r.forward_time_ms for r in results) / n
        
        backward_times = [r.backward_time_ms for r in results if r.backward_time_ms is not None]
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else None
        
        training_times = [r.training_time_ms for r in results if r.training_time_ms is not None]
        avg_training = sum(training_times) / len(training_times) if training_times else None
        
        avg_memory = sum(r.memory_usage_mb for r in results) / n
        avg_loss = sum(r.accuracy_loss for r in results) / n
        
        # Use values from first result for consistent fields
        first_result = results[0]
        
        return BenchmarkResult(
            backend=backend,
            operation="forward_backward_training",
            model_config=config,
            forward_time_ms=avg_forward,
            backward_time_ms=avg_backward,
            training_time_ms=avg_training,
            memory_usage_mb=avg_memory,
            parameter_count=first_result.parameter_count,
            accuracy_loss=avg_loss,
            success=True
        )
    
    def compare_backends(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comparison analysis between backends."""
        if not results:
            return {}
        
        # Group results by model config
        config_groups = {}
        for result in results:
            config_key = json.dumps(result.model_config, sort_keys=True)
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
        
        analysis = {}
        
        for config_key, group_results in config_groups.items():
            config = json.loads(config_key)
            config_name = f"{config['hidden_dim']}d_{config['num_layers']}l_{config['num_heads']}h"
            
            successful_results = [r for r in group_results if r.success]
            if not successful_results:
                continue
            
            # Find fastest backend
            fastest = min(successful_results, key=lambda r: r.forward_time_ms)
            
            # Calculate speedups
            backend_comparison = {}
            for result in successful_results:
                speedup = fastest.forward_time_ms / result.forward_time_ms
                efficiency = result.forward_time_ms / result.memory_usage_mb  # ms per MB
                
                backend_comparison[result.backend] = {
                    "forward_time_ms": result.forward_time_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                    "parameter_count": result.parameter_count,
                    "speedup_vs_fastest": speedup,
                    "efficiency_ms_per_mb": efficiency,
                    "accuracy_loss": result.accuracy_loss,
                }
            
            analysis[config_name] = {
                "config": config,
                "fastest_backend": fastest.backend,
                "backend_comparison": backend_comparison,
            }
        
        return analysis
    
    def generate_report(self, results: List[BenchmarkResult], save_path: Optional[str] = None) -> str:
        """Generate a comprehensive benchmark report."""
        analysis = self.compare_backends(results)
        
        report = []
        report.append("🚀 Transformer Backend Performance Report")
        report.append("=" * 60)
        report.append("")
        
        if not analysis:
            report.append("❌ No successful benchmark results to analyze.")
            return "\n".join(report)
        
        # Summary table
        report.append("📊 Performance Summary")
        report.append("-" * 40)
        
        headers = ["Model", "Backend", "Forward (ms)", "Memory (MB)", "Speedup", "Efficiency"]
        report.append(f"{'Model':<15} {'Backend':<8} {'Forward':<12} {'Memory':<10} {'Speedup':<8} {'Efficiency':<10}")
        report.append("-" * 75)
        
        for config_name, data in analysis.items():
            first = True
            for backend, metrics in data["backend_comparison"].items():
                model_name = config_name if first else ""
                report.append(
                    f"{model_name:<15} {backend:<8} "
                    f"{metrics['forward_time_ms']:<12.2f} "
                    f"{metrics['memory_usage_mb']:<10.1f} "
                    f"{metrics['speedup_vs_fastest']:<8.2f}x "
                    f"{metrics['efficiency_ms_per_mb']:<10.3f}"
                )
                first = False
            report.append("")
        
        # Detailed analysis
        report.append("\n🔍 Detailed Analysis")
        report.append("-" * 40)
        
        for config_name, data in analysis.items():
            report.append(f"\n📈 {config_name}:")
            report.append(f"   Fastest: {data['fastest_backend']}")
            
            for backend, metrics in data["backend_comparison"].items():
                report.append(f"   {backend}:")
                report.append(f"     • Forward time: {metrics['forward_time_ms']:.2f}ms")
                report.append(f"     • Memory usage: {metrics['memory_usage_mb']:.1f}MB")
                report.append(f"     • Parameters: {metrics['parameter_count']:,}")
                report.append(f"     • Loss: {metrics['accuracy_loss']:.4f}")
                report.append(f"     • Speedup: {metrics['speedup_vs_fastest']:.2f}x")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\n📄 Report saved to: {save_path}")
        
        return report_text
    
    def save_raw_results(self, results: List[BenchmarkResult], path: str):
        """Save raw benchmark results as JSON."""
        json_data = [asdict(result) for result in results]
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"📊 Raw results saved to: {path}")


def main():
    """Run comprehensive benchmarks."""
    
    # Define model configurations to test
    test_configs = [
        ("tiny", {
            "vocab_size": 20,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ff_dim": 32,
            "max_seq_len": 50,
        }),
        ("small", {
            "vocab_size": 50,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 64,
            "max_seq_len": 100,
        }),
        ("medium", {
            "vocab_size": 100,
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 8,
            "ff_dim": 128,
            "max_seq_len": 200,
        }),
    ]
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    results = benchmark.benchmark_model_sizes(test_configs, num_runs=3)
    
    # Generate and display report
    report = benchmark.generate_report(results, "benchmark_report.txt")
    print("\n" + report)
    
    # Save raw results
    benchmark.save_raw_results(results, "benchmark_results.json")
    
    print(f"\n✅ Benchmarking completed!")
    print(f"   Results: {len([r for r in results if r.success])} successful, {len([r for r in results if not r.success])} failed")


if __name__ == "__main__":
    main()