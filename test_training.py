#!/usr/bin/env python3
"""
Test script for training functionality.
"""

import numpy as np
from transformerlab.core.tokenizer import load_corpus
from transformerlab.core.transformer import Transformer
from transformerlab.core.optimizer import SGDOptimizer

def test_training():
    """Test the training functionality."""
    print("🧠 Testing Transformer Training")
    print("=" * 40)

    # Load corpus
    text, tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")
    print(f"Loaded corpus: {len(text)} characters, vocab size: {tokenizer.vocab_size}")

    # Create model
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,  # Small for testing
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        norm_type="LayerNorm",
        activation_type="ReLU",
        residual_type="Pre-LN"
    )

    # Create optimizer
    optimizer = SGDOptimizer(learning_rate=0.01)

    # Create training data
    tokens = tokenizer.encode(text)
    seq_len = 20
    batch_size = 2

    print(f"Model parameters: {len(model.get_parameters())}")
    print(f"Training with seq_len={seq_len}, batch_size={batch_size}")
    print()

    # Training loop
    num_steps = 10
    for step in range(num_steps):
        # Create random batch
        batch_tokens = []
        batch_targets = []

        for _ in range(batch_size):
            start_idx = np.random.randint(0, len(tokens) - seq_len - 1)
            batch_tokens.append(tokens[start_idx : start_idx + seq_len])
            batch_targets.append(tokens[start_idx + 1 : start_idx + seq_len + 1])

        # Convert to numpy arrays
        batch_tokens = np.array(batch_tokens)
        batch_targets = np.array(batch_targets)

        # Training step
        try:
            loss = model.train_step(batch_tokens, batch_targets, optimizer)
            print(f"Step {step + 1}: Loss = {loss:.4f}")
        except Exception as e:
            print(f"Step {step + 1}: Error - {e}")
            import traceback
            traceback.print_exc()
            break

    print()
    print("✅ Training test completed!")

if __name__ == "__main__":
    test_training()