#!/usr/bin/env python3
"""
Simple test script for the Transformer Intuition Lab.
"""

import numpy as np
from transformerlab.core.tokenizer import load_corpus
from transformerlab.core.transformer import Transformer
from transformerlab.viz.plots import plot_loss_history
import matplotlib.pyplot as plt

def test_basic_functionality():
    """Test basic transformer functionality."""
    print("🧠 Testing Transformer Intuition Lab...")
    
    # Load corpus
    print("📚 Loading corpus...")
    text, tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")
    print(f"   Corpus length: {len(text)} characters")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    print("🏗️ Creating transformer model...")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        norm_type="LayerNorm",
        activation_type="ReLU",
        residual_type="Pre-LN",
        pos_encoding_type="Sinusoidal"
    )
    print(f"   Model created with {model.num_layers} layers")
    
    # Test forward pass
    print("⚡ Testing forward pass...")
    batch_size, seq_len = 2, 10
    tokens = tokenizer.encode(text[:seq_len * batch_size])
    
    # Create batches with proper padding
    x = np.zeros((batch_size, seq_len), dtype=np.int32)
    targets = np.zeros((batch_size, seq_len), dtype=np.int32)
    
    for i in range(batch_size):
        start_idx = i * seq_len
        end_idx = min(start_idx + seq_len, len(tokens))
        x[i, :end_idx - start_idx] = tokens[start_idx:end_idx]
        if end_idx < len(tokens):
            targets[i, :end_idx - start_idx] = tokens[start_idx + 1:end_idx + 1]
    
    logits, stats = model.forward(x, targets)
    print(f"   Forward pass successful! Loss: {stats['loss']:.4f}")
    
    # Test training
    print("🎯 Testing training...")
    for step in range(5):
        logits, stats = model.forward(x, targets)
        print(f"   Step {step + 1}: Loss = {stats['loss']:.4f}")
    
    # Test generation
    print("✍️ Testing text generation...")
    prompt = tokenizer.encode("First Citizen:")
    prompt_array = np.array([prompt])
    generated = model.generate(prompt_array, max_length=20, temperature=0.8)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"   Generated: {generated_text}")
    
    # Test visualization
    print("📊 Testing visualization...")
    if model.loss_history:
        fig = plot_loss_history(model.loss_history, "Test Training Loss")
        plt.savefig("test_loss.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   Loss plot saved as test_loss.png")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()