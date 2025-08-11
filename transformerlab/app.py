"""
Main Streamlit application for the Transformer Intuition Lab.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from transformerlab.core.tokenizer import load_corpus
from transformerlab.core.transformer import Transformer
from transformerlab.viz.plots import plot_loss_history


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Transformer Intuition Lab",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧠 Transformer Intuition Lab")
    st.markdown("Interactive playground for understanding transformer architectures")

    # Initialize session state
    if "model" not in st.session_state:
        st.session_state.model = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "text" not in st.session_state:
        st.session_state.text = ""
    if "experiment_history" not in st.session_state:
        st.session_state.experiment_history = []

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Model Configuration")

        # Corpus selection
        st.subheader("📚 Corpus")
        corpus_file = st.selectbox(
            "Select Corpus", ["tiny_shakespeare.txt"], help="Choose the training corpus"
        )

        # Model architecture
        st.subheader("🏗️ Architecture")
        hidden_dim = st.slider("Hidden Dimension", 64, 512, 256, 64)
        num_layers = st.slider("Number of Layers", 2, 12, 6, 1)
        num_heads = st.slider("Number of Heads", 2, 16, 8, 2)
        ff_dim = st.slider("Feed-forward Dimension", 128, 2048, 1024, 128)

        # Normalization
        st.subheader("📏 Normalization")
        norm_type = st.radio(
            "Normalization Type",
            ["LayerNorm", "RMSNorm", "None"],
            help="Choose the normalization method",
        )

        # Residual connections
        st.subheader("🔄 Residual Connections")
        residual_type = st.radio(
            "Residual Type",
            ["Pre-LN", "Post-LN", "Sandwich"],
            help="Choose the residual connection pattern",
        )

        # Activation functions
        st.subheader("⚡ Activation Functions")
        activation_type = st.selectbox(
            "Activation Type",
            ["ReLU", "GeLU", "Swish", "SwiGLU"],
            help="Choose the activation function",
        )

        # Positional encoding
        st.subheader("📍 Positional Encoding")
        pos_encoding_type = st.selectbox(
            "Positional Encoding Type",
            ["Sinusoidal", "RoPE", "ALiBi"],
            help="Choose the positional encoding method",
        )

        # Training controls
        st.subheader("🎯 Training")
        batch_size = st.slider("Batch Size", 1, 8, 2, 1)
        seq_len = st.slider("Sequence Length", 32, 256, 128, 32)
        num_steps = st.slider("Training Steps", 1, 50, 10, 1)

        # Action buttons
        st.subheader("🚀 Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Initialize Model"):
                initialize_model(
                    corpus_file,
                    hidden_dim,
                    num_layers,
                    num_heads,
                    ff_dim,
                    norm_type,
                    residual_type,
                    activation_type,
                    pos_encoding_type,
                )

        with col2:
            if st.button("🎯 Train Model"):
                if st.session_state.model is not None:
                    train_model(batch_size, seq_len, num_steps)
                else:
                    st.error("Please initialize the model first!")

        if st.button("🗑️ Reset"):
            reset_session()

        # Experiment management
        st.subheader("📊 Experiments")
        if st.button("💾 Save Experiment"):
            save_experiment()

        if st.button("📈 Compare Experiments"):
            compare_experiments()

    # Main content area
    if st.session_state.model is not None:
        # Model info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Model Size", f"{get_model_size(st.session_state.model):.1f}K params"
            )

        with col2:
            st.metric("Training Steps", st.session_state.model.step_count)

        with col3:
            if st.session_state.model.loss_history:
                st.metric(
                    "Current Loss", f"{st.session_state.model.loss_history[-1]:.4f}"
                )
            else:
                st.metric("Current Loss", "N/A")

        # Plots
        st.subheader("📊 Training Metrics")

        if st.session_state.model.loss_history:
            # Loss plot
            fig = plot_loss_history(st.session_state.model.loss_history)
            st.pyplot(fig)
            plt.close(fig)

            # Attention heatmap
            if st.session_state.model.step_count > 0:
                # Get attention weights from the last forward pass
                # This would require storing the last forward pass results
                st.info("Attention heatmap will be available after training")

        # Model statistics
        st.subheader("📈 Model Statistics")
        model_stats = st.session_state.model.get_model_stats()

        col1, col2 = st.columns(2)

        with col1:
            st.json(model_stats["config"])

        with col2:
            if model_stats["layer_stats"]:
                # Layer statistics
                layer_data = []
                for i, layer_stat in enumerate(model_stats["layer_stats"]):
                    layer_data.append(
                        {
                            "Layer": i,
                            "Attention Mean": layer_stat["attention"][
                                "attention_weights_mean"
                            ],
                            "FF Output Mean": layer_stat["feed_forward"]["output_mean"],
                        }
                    )

                st.dataframe(layer_data)

        # Text generation
        st.subheader("✍️ Text Generation")

        prompt = st.text_input("Enter a prompt:", "First Citizen:")

        if st.button("🎲 Generate"):
            if prompt:
                generated_text = generate_text(prompt)
                st.text_area("Generated Text:", generated_text, height=200)

    else:
        st.info(
            "👈 Use the sidebar to configure and initialize your transformer model!"
        )

        # Show corpus info
        if st.session_state.text:
            st.subheader("📚 Corpus Information")
            st.text(f"Corpus length: {len(st.session_state.text)} characters")
            st.text(f"Vocabulary size: {st.session_state.tokenizer.vocab_size}")
            st.text(f"Sample characters: {st.session_state.tokenizer.chars[:20]}")

            st.subheader("📖 Corpus Preview")
            st.text(st.session_state.text[:500] + "...")


def initialize_model(
    corpus_file: str,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    norm_type: str,
    residual_type: str,
    activation_type: str,
    pos_encoding_type: str,
):
    """Initialize the transformer model."""
    try:
        # Load corpus
        corpus_path = os.path.join(os.path.dirname(__file__), "data", corpus_file)
        text, tokenizer = load_corpus(corpus_path)

        # Store in session state
        st.session_state.text = text
        st.session_state.tokenizer = tokenizer

        # Create model
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            norm_type=norm_type,
            activation_type=activation_type,
            residual_type=residual_type,
            pos_encoding_type=pos_encoding_type,
        )

        st.session_state.model = model

        st.success(
            f"✅ Model initialized with {get_model_size(model):.1f}K parameters!"
        )

    except Exception as e:
        st.error(f"❌ Error initializing model: {str(e)}")


def train_model(batch_size: int, seq_len: int, num_steps: int):
    """Train the model for the specified number of steps."""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        st.error("Model not initialized!")
        return

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    text = st.session_state.text

    # Create training data
    tokens = tokenizer.encode(text)

    progress_bar = st.progress(0)
    status_text = st.empty()

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

        # Forward pass
        logits, stats = model.forward(batch_tokens, batch_targets)

        # Update progress
        progress = (step + 1) / num_steps
        progress_bar.progress(progress)
        status_text.text(
            f"Training step {step + 1}/{num_steps} - Loss: {stats['loss']:.4f}"
        )

    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Training completed! Final loss: {model.loss_history[-1]:.4f}")


def generate_text(prompt: str, max_length: int = 100) -> str:
    """Generate text from a prompt."""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        return "Model not initialized!"

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = np.array([prompt_tokens])

    # Generate
    generated_tokens = model.generate(
        prompt_array, max_length=max_length, temperature=0.8
    )

    # Decode
    generated_text = tokenizer.decode(generated_tokens[0].tolist())

    return generated_text


def get_model_size(model: Transformer) -> float:
    """Calculate model size in thousands of parameters."""
    # This is a rough estimate
    total_params = (
        model.vocab_size * model.hidden_dim  # token embeddings
        + model.hidden_dim * model.vocab_size  # output projection
        + model.num_layers
        * (
            4 * model.hidden_dim * model.hidden_dim  # attention weights
            + 2 * model.hidden_dim * model.ff_dim  # feed-forward weights
            + 2 * model.hidden_dim  # normalization parameters
        )
    )
    return total_params / 1000


def save_experiment():
    """Save current experiment configuration and results."""
    if st.session_state.model is None:
        st.error("No model to save!")
        return

    model_stats = st.session_state.model.get_model_stats()

    experiment = {
        "config": model_stats["config"],
        "loss_history": model_stats["loss_history"],
        "step_count": model_stats["step_count"],
    }

    st.session_state.experiment_history.append(experiment)
    st.success(
        f"✅ Experiment saved! Total experiments: {len(st.session_state.experiment_history)}"
    )


def compare_experiments():
    """Compare saved experiments."""
    if not st.session_state.experiment_history:
        st.warning("No experiments to compare!")
        return

    st.subheader("📊 Experiment Comparison")

    # Create comparison table
    comparison_data = []
    for i, exp in enumerate(st.session_state.experiment_history):
        row = {
            "Experiment": f"Exp {i+1}",
            "Norm Type": exp["config"]["norm_type"],
            "Activation": exp["config"]["activation_type"],
            "Residual": exp["config"]["residual_type"],
            "Pos Encoding": exp["config"]["pos_encoding_type"],
            "Final Loss": exp["loss_history"][-1] if exp["loss_history"] else "N/A",
            "Steps": exp["step_count"],
        }
        comparison_data.append(row)

    st.dataframe(comparison_data)

    # Plot loss comparisons
    if len(st.session_state.experiment_history) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, exp in enumerate(st.session_state.experiment_history):
            if exp["loss_history"]:
                ax.plot(exp["loss_history"], label=f"Exp {i+1}")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Comparison Across Experiments")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)


def reset_session():
    """Reset the session state."""
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.text = ""
    st.session_state.experiment_history = []
    st.success("✅ Session reset!")


if __name__ == "__main__":
    main()
