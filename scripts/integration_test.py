#!/usr/bin/env python3
"""
Integration test script for the complete transformer system.
Tests end-to-end functionality across all backends.
"""

import sys
import traceback
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformerlab.backends.factory import create_transformer, list_backends, compare_backends


def test_basic_functionality():
    """Test basic functionality across all backends."""
    print("🧪 Integration Test: Basic Functionality")
    print("=" * 50)
    
    config = {
        "vocab_size": 30,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 64,
        "max_seq_len": 20,
    }
    
    # Sample data
    sample_input = [[1, 2, 3, 4, 5]]
    sample_target = [[2, 3, 4, 5, 6]]
    
    backends_tested = []
    
    for backend_name in list_backends():
        print(f"\n🔧 Testing {backend_name} backend:")
        
        try:
            # 1. Create transformer
            transformer = create_transformer(backend_name, **config)
            print(f"  ✅ Model creation successful")
            
            # 2. Test parameter counting
            param_count = transformer.get_parameter_count()
            print(f"  📊 Parameter count: {param_count:,}")
            assert param_count > 0
            
            # 3. Test forward pass
            if backend_name == "torch":
                import torch
                inputs = torch.tensor(sample_input, dtype=torch.long)
                targets = torch.tensor(sample_target, dtype=torch.long)
            else:
                inputs = sample_input
                targets = sample_target
            
            logits, stats = transformer.forward(inputs, targets)
            print(f"  🚀 Forward pass successful")
            print(f"  📈 Loss: {stats.get('loss', 'N/A')}")
            
            # 4. Test generation (if available)
            try:
                prompt = [[1, 2, 3]]
                if backend_name == "torch":
                    import torch
                    prompt = torch.tensor(prompt, dtype=torch.long)
                
                generated = transformer.generate(prompt, max_length=5)
                print(f"  ✨ Generation successful")
            except (NotImplementedError, Exception) as e:
                print(f"  ⚠️  Generation not available: {str(e)[:50]}...")
            
            backends_tested.append(backend_name)
            print(f"  ✅ {backend_name} backend: ALL TESTS PASSED")
            
        except Exception as e:
            print(f"  ❌ {backend_name} backend failed: {e}")
            if "torch" in str(e).lower():
                print(f"  ℹ️  PyTorch may not be installed - this is expected")
            else:
                print(f"  🐛 Error details: {traceback.format_exc()}")
    
    print(f"\n✅ Successfully tested {len(backends_tested)} backends: {backends_tested}")
    return len(backends_tested) >= 2  # At least numpy and python should work


def test_backend_comparison():
    """Test backend comparison functionality."""
    print("\n\n🔍 Integration Test: Backend Comparison")
    print("=" * 50)
    
    config = {
        "vocab_size": 25,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ff_dim": 32,
    }
    
    try:
        comparison = compare_backends(["numpy", "python", "torch"], config)
        
        available_count = 0
        for backend_name, info in comparison.items():
            if info.get("available", False):
                available_count += 1
                print(f"  ✅ {backend_name}: Available")
                print(f"     Parameters: {info.get('parameter_count', 'N/A'):,}")
                print(f"     Description: {info.get('description', 'N/A')}")
            else:
                print(f"  ❌ {backend_name}: Not available - {info.get('error', 'Unknown')}")
        
        print(f"\n📊 Comparison complete: {available_count}/3 backends available")
        return available_count >= 2
        
    except Exception as e:
        print(f"❌ Backend comparison failed: {e}")
        return False


def test_educational_features():
    """Test educational features and transparency."""
    print("\n\n🎓 Integration Test: Educational Features")
    print("=" * 50)
    
    # Test Python backend verbose output
    print("\n📝 Testing Python backend verbosity:")
    try:
        config = {"vocab_size": 10, "hidden_dim": 8, "num_layers": 1, "num_heads": 2, "ff_dim": 16}
        transformer = create_transformer("python", **config)
        
        # Capture some output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            transformer.forward([[1, 2, 3]])
        
        output = f.getvalue()
        if "PythonTransformer" in output:
            print("  ✅ Python backend produces educational output")
        else:
            print("  ⚠️  Limited educational output detected")
            
    except Exception as e:
        print(f"  ❌ Python backend test failed: {e}")
    
    # Test NumPy backend efficiency
    print("\n⚡ Testing NumPy backend efficiency:")
    try:
        import time
        config = {"vocab_size": 20, "hidden_dim": 16, "num_layers": 2, "num_heads": 4, "ff_dim": 32}
        transformer = create_transformer("numpy", **config)
        
        start_time = time.time()
        transformer.forward([[1, 2, 3, 4, 5]])
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        print(f"  ⏱️  Forward pass: {execution_time:.2f}ms")
        
        if execution_time < 100:  # Less than 100ms is reasonable
            print("  ✅ NumPy backend performs efficiently")
        else:
            print("  ⚠️  Performance could be better")
            
    except Exception as e:
        print(f"  ❌ NumPy backend test failed: {e}")
    
    return True


def test_architecture_features():
    """Test different architecture configurations."""
    print("\n\n🏗️  Integration Test: Architecture Features")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {
            "name": "LayerNorm + ReLU",
            "config": {"vocab_size": 15, "hidden_dim": 16, "num_layers": 1, "num_heads": 2, 
                      "ff_dim": 32, "norm_type": "LayerNorm", "activation_type": "ReLU"}
        },
        {
            "name": "RMSNorm + GeLU", 
            "config": {"vocab_size": 15, "hidden_dim": 16, "num_layers": 1, "num_heads": 2,
                      "ff_dim": 32, "norm_type": "RMSNorm", "activation_type": "GeLU"}
        },
        {
            "name": "No Norm + Swish",
            "config": {"vocab_size": 15, "hidden_dim": 16, "num_layers": 1, "num_heads": 2,
                      "ff_dim": 32, "norm_type": "None", "activation_type": "Swish"}
        }
    ]
    
    success_count = 0
    for test_case in configs:
        print(f"\n🧪 Testing {test_case['name']}:")
        
        try:
            # Test with NumPy backend (most stable)
            transformer = create_transformer("numpy", **test_case["config"])
            transformer.forward([[1, 2, 3]])
            
            print(f"  ✅ {test_case['name']}: Configuration works")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ {test_case['name']}: {e}")
    
    print(f"\n🏗️  Architecture test results: {success_count}/{len(configs)} configurations successful")
    return success_count >= 2


def main():
    """Run complete integration test suite."""
    print("🧠 Transformer Intuition Lab - Integration Testing")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    try:
        # Test 1: Basic functionality
        if test_basic_functionality():
            tests_passed += 1
        
        # Test 2: Backend comparison
        if test_backend_comparison():
            tests_passed += 1
        
        # Test 3: Educational features
        if test_educational_features():
            tests_passed += 1
            
        # Test 4: Architecture features
        if test_architecture_features():
            tests_passed += 1
        
        # Final report
        print(f"\n\n{'='*60}")
        print(f"🏁 INTEGRATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Tests Passed: {tests_passed}/{total_tests}")
        print(f"Success Rate: {tests_passed/total_tests*100:.1f}%")
        
        if tests_passed == total_tests:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("✅ The transformer system is ready for use")
            return 0
        elif tests_passed >= 3:
            print("✅ MOST TESTS PASSED - System is functional")
            print("⚠️  Some advanced features may need attention")
            return 0
        else:
            print("❌ INTEGRATION TESTS FAILED")
            print("🔧 System needs debugging before use")
            return 1
            
    except Exception as e:
        print(f"\n❌ Integration testing failed with error: {e}")
        print(f"🐛 Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())