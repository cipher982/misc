#!/usr/bin/env python3
"""
Demo script for the Transformer Intuition Lab.
Shows key features and capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np

from transformerlab.core.tokenizer import load_corpus
from transformerlab.core.transformer import Transformer
from transformerlab.core.optimizer import SGDOptimizer
from transformerlab.viz.plots import plot_layer_statistics, plot_loss_history


def demo_basic_training():
    """Demonstrate basic training with different configurations."""
    print("🎯 Demo: Basic Training Comparison")
    print("=" * 50)

    # Load corpus
    text, tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")

    # Test configurations
    configs = [
        {
            "name": "LayerNorm + ReLU",
            "norm": "LayerNorm",
            "activation": "ReLU",
            "residual": "Pre-LN",
        },
        {
            "name": "RMSNorm + GeLU",
            "norm": "RMSNorm",
            "activation": "GeLU",
            "residual": "Pre-LN",
        },
        {
            "name": "No Norm + Swish",
            "norm": "None",
            "activation": "Swish",
            "residual": "Post-LN",
        },
    ]

    results = []

    for config in configs:
        print(f"\n🏗️ Testing: {config['name']}")

        # Create model
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ff_dim=128,
            norm_type=config["norm"],
            activation_type=config["activation"],
            residual_type=config["residual"],
            pos_encoding_type="Sinusoidal",
        )

        # Train for a few steps
        batch_size, seq_len = 2, 20
        tokens = tokenizer.encode(text[: seq_len * batch_size * 2])

        x = np.zeros((batch_size, seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, seq_len), dtype=np.int32)

        for i in range(batch_size):
            start_idx = i * seq_len
            end_idx = min(start_idx + seq_len, len(tokens))
            x[i, : end_idx - start_idx] = tokens[start_idx:end_idx]
            if end_idx < len(tokens):
                targets[i, : end_idx - start_idx] = tokens[start_idx + 1 : end_idx + 1]

        # Create optimizer
        optimizer = SGDOptimizer(learning_rate=0.01)

        # Train
        losses = []
        for step in range(10):
            loss = model.train_step(x, targets, optimizer)
            losses.append(loss)
            if step % 3 == 0:
                print(f"   Step {step}: Loss = {loss:.4f}")

        results.append(
            {"config": config["name"], "losses": losses, "final_loss": losses[-1]}
        )

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(result["losses"], label=result["config"], linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    configs = [r["config"] for r in results]
    final_losses = [r["final_loss"] for r in results]
    bars = plt.bar(configs, final_losses, color=["skyblue", "lightgreen", "lightcoral"])
    plt.ylabel("Final Loss")
    plt.title("Final Loss by Configuration")
    plt.xticks(rotation=45)

    for bar, loss in zip(bars, final_losses, strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{loss:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("demo_training_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Results saved to demo_training_comparison.png")
    print(f"Best performing: {min(results, key=lambda x: x['final_loss'])['config']}")


def demo_text_generation():
    """Demonstrate text generation capabilities."""
    print("\n✍️ Demo: Text Generation")
    print("=" * 50)

    # Load corpus and create model
    text, tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        ff_dim=256,
        norm_type="LayerNorm",
        activation_type="GeLU",
        residual_type="Pre-LN",
        pos_encoding_type="Sinusoidal",
    )

    # Train briefly
    print("🎯 Training model for generation...")
    batch_size, seq_len = 2, 30
    tokens = tokenizer.encode(text[: seq_len * batch_size * 3])

    x = np.zeros((batch_size, seq_len), dtype=np.int32)
    targets = np.zeros((batch_size, seq_len), dtype=np.int32)

    for i in range(batch_size):
        start_idx = i * seq_len
        end_idx = min(start_idx + seq_len, len(tokens))
        x[i, : end_idx - start_idx] = tokens[start_idx:end_idx]
        if end_idx < len(tokens):
            targets[i, : end_idx - start_idx] = tokens[start_idx + 1 : end_idx + 1]

    # Create optimizer
    optimizer = SGDOptimizer(learning_rate=0.01)

    for step in range(20):
        loss = model.train_step(x, targets, optimizer)
        if step % 5 == 0:
            print(f"   Step {step}: Loss = {loss:.4f}")

    # Generate text
    print("\n🎲 Generating text...")
    prompts = ["First Citizen:", "All:", "Second Citizen:"]

    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        prompt_tokens = tokenizer.encode(prompt)
        prompt_array = np.array([prompt_tokens])

        # Generate with different temperatures
        for temp in [0.5, 0.8, 1.2]:
            generated = model.generate(prompt_array, max_length=30, temperature=temp)
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"   T={temp}: {generated_text}")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n📊 Demo: Visualization Features")
    print("=" * 50)

    # Load corpus and create model
    text, tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=96,
        num_layers=3,
        num_heads=6,
        ff_dim=192,
        norm_type="LayerNorm",
        activation_type="ReLU",
        residual_type="Pre-LN",
        pos_encoding_type="Sinusoidal",
    )

    # Train and collect statistics
    print("🎯 Training with statistics collection...")
    batch_size, seq_len = 2, 25
    tokens = tokenizer.encode(text[: seq_len * batch_size * 4])

    x = np.zeros((batch_size, seq_len), dtype=np.int32)
    targets = np.zeros((batch_size, seq_len), dtype=np.int32)

    for i in range(batch_size):
        start_idx = i * seq_len
        end_idx = min(start_idx + seq_len, len(tokens))
        x[i, : end_idx - start_idx] = tokens[start_idx:end_idx]
        if end_idx < len(tokens):
            targets[i, : end_idx - start_idx] = tokens[start_idx + 1 : end_idx + 1]

    # Create optimizer
    optimizer = SGDOptimizer(learning_rate=0.01)

    # Train and collect layer stats
    layer_stats_history = []
    losses = []

    for step in range(15):
        loss = model.train_step(x, targets, optimizer)
        losses.append(loss)
        
        # Get layer stats from forward pass for visualization
        logits, stats = model.forward(x, targets)
        layer_stats_history.append(stats["layer_stats"])

        if step % 5 == 0:
            print(f"   Step {step}: Loss = {loss:.4f}")

    # Create visualizations
    print("\n📈 Creating visualizations...")

    # Loss plot (use our collected losses instead of model.loss_history which has duplicates)
    fig = plot_loss_history(losses, "Training Loss Over Time")
    plt.savefig("demo_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Layer statistics (from last step)
    if layer_stats_history:
        last_layer_stats = layer_stats_history[-1]
        fig = plot_layer_statistics(
            last_layer_stats,
            "attention_weights_mean",
            "Mean Attention Weights Across Layers",
        )
        plt.savefig("demo_layer_stats.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("📊 Visualizations saved:")
    print("   - demo_loss.png")
    print("   - demo_layer_stats.png")


def main():
    """Run all demos."""
    print("🧠 Transformer Intuition Lab - Demo Suite")
    print("=" * 60)

    try:
        demo_basic_training()
        demo_text_generation()
        demo_visualization()

        print("\n✅ All demos completed successfully!")
        print("\n🎉 Key Features Demonstrated:")
        print("   • Different normalization strategies")
        print("   • Various activation functions")
        print("   • Text generation with temperature control")
        print("   • Training visualization")
        print("   • Layer statistics analysis")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
