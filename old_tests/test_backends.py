#!/usr/bin/env python3
"""
Test script to verify all backends work correctly.
"""

import sys
import traceback
from transformerlab.backends.factory import create_transformer, list_backends, compare_backends

def test_backend_creation():
    """Test creating transformers with different backends."""
    print("🧪 Testing Backend Creation")
    print("=" * 50)
    
    # Test configuration
    config = {
        "vocab_size": 50,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "max_seq_len": 100,
    }
    
    backends = list_backends()
    print(f"Available backends: {backends}")
    
    for backend_name in backends:
        print(f"\\n🔧 Testing {backend_name} backend:")
        
        try:
            # Create transformer
            transformer = create_transformer(backend_name, **config)
            print(f"  ✅ Created {backend_name} transformer successfully")
            
            # Test parameter count
            param_count = transformer.get_parameter_count()
            print(f"  📊 Parameter count: {param_count:,}")
            
            # Test backend info
            backend_info = transformer.get_backend_info()
            print(f"  ℹ️  Backend type: {backend_info['backend_type']}")
            
        except Exception as e:
            print(f"  ❌ Failed to create {backend_name} transformer: {e}")
            print(f"     Error details: {traceback.format_exc()}")

def test_backend_comparison():
    """Test backend comparison functionality."""
    print("\\n\\n📊 Testing Backend Comparison")
    print("=" * 50)
    
    config = {
        "vocab_size": 30,
        "hidden_dim": 32,
        "num_layers": 1,
        "num_heads": 2,
        "ff_dim": 64,
    }
    
    try:
        comparison = compare_backends(["numpy", "python", "torch"], config)
        
        for backend_name, info in comparison.items():
            print(f"\\n{backend_name.upper()}:")
            if info.get("available", False):
                print(f"  ✅ Available")
                print(f"  📊 Parameters: {info.get('parameter_count', 'N/A'):,}")
                if 'description' in info:
                    print(f"  📝 Description: {info['description']}")
            else:
                print(f"  ❌ Not available: {info.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Backend comparison failed: {e}")

def test_simple_forward_pass():
    """Test simple forward pass for available backends."""
    print("\\n\\n🚀 Testing Forward Pass")
    print("=" * 50)
    
    # Small configuration for testing
    config = {
        "vocab_size": 20,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ff_dim": 32,
    }
    
    # Sample input (batch_size=1, seq_len=5)
    sample_input = [[1, 2, 3, 4, 5]]
    sample_targets = [[2, 3, 4, 5, 6]]
    
    for backend_name in ["numpy", "python", "torch"]:  # Test all backends
        print(f"\\n🔧 Testing {backend_name} forward pass:")
        
        try:
            transformer = create_transformer(backend_name, **config)
            
            # Import numpy for conversion if needed
            if backend_name == "torch":
                import torch
                x = torch.tensor(sample_input, dtype=torch.long)
                targets = torch.tensor(sample_targets, dtype=torch.long)
            else:
                x = sample_input
                targets = sample_targets
            
            # Forward pass
            logits, stats = transformer.forward(x, targets)
            
            print(f"  ✅ Forward pass successful")
            print(f"  📊 Loss: {stats.get('loss', 'N/A')}")
            
            if hasattr(logits, 'shape'):
                print(f"  📏 Output shape: {logits.shape}")
            elif hasattr(logits, '__len__'):
                print(f"  📏 Output shape: {len(logits)} x {len(logits[0])} x {len(logits[0][0])}")
            
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            print(f"     Details: {traceback.format_exc()}")

def main():
    """Run all backend tests."""
    print("🧠 Transformer Intuition Lab - Backend Testing")
    print("=" * 60)
    
    try:
        test_backend_creation()
        test_backend_comparison() 
        test_simple_forward_pass()
        
        print("\\n\\n✅ Backend testing completed!")
        
    except Exception as e:
        print(f"\\n\\n❌ Backend testing failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()