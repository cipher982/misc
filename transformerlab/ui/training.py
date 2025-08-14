"""
Training and inference UI components for the Transformer Intuition Lab.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from transformerlab.backends.factory import create_transformer
from transformerlab.core.tokenizer import load_corpus
from transformerlab.viz.plots import plot_loss_history


def initialize_model(config: dict[str, Any]):
    """Initialize the transformer model with given configuration."""
    try:
        # Load corpus
        corpus_path = os.path.join(
            os.path.dirname(__file__), "..", "data", config["corpus_file"]
        )
        text, tokenizer = load_corpus(corpus_path)

        # Store in session state
        st.session_state.text = text
        st.session_state.tokenizer = tokenizer

        # Get current backend
        backend_name = st.session_state.get("current_backend", "numpy")

        # Create model with selected backend
        # Use verbose=False for python backend in web interface to keep console clean
        verbose = backend_name != "python"

        model = create_transformer(
            backend_name=backend_name,
            vocab_size=tokenizer.vocab_size,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            norm_type=config["norm_type"],
            activation_type=config["activation_type"],
            residual_type=config["residual_type"],
            pos_encoding_type=config["pos_encoding_type"],
            verbose=verbose,
        )

        st.session_state.model = model

        st.success(
            f"✅ Model initialized with {get_model_size(model):.1f}K parameters!"
        )

    except Exception as e:
        st.error(f"❌ Error initializing model: {str(e)}")


def train_model(config: dict[str, Any]):
    """Train the model for the specified number of steps."""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        st.error("Model not initialized!")
        return

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    text = st.session_state.text

    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    num_steps = config["num_steps"]

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

        # Initialize optimizer if not exists or backend changed
        backend_name = st.session_state.get("current_backend", "numpy")
        if (
            not hasattr(st.session_state, "optimizer")
            or st.session_state.optimizer is None
            or st.session_state.get("optimizer_backend") != backend_name
        ):
            st.session_state.optimizer = create_backend_optimizer(backend_name, model)
            st.session_state.optimizer_backend = backend_name

        # Training step
        loss = model.train_step(batch_tokens, batch_targets, st.session_state.optimizer)

        # Get stats for display
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


def create_backend_optimizer(
    backend_name: str, model, optimizer_type: str = "adam", learning_rate: float = 0.001
):
    """Create optimizer appropriate for the backend."""
    if backend_name == "numpy":
        from transformerlab.backends.numpy_backend.optimizer import (
            create_numpy_optimizer,
        )

        return create_numpy_optimizer(optimizer_type, learning_rate=learning_rate)

    if backend_name == "python":
        from transformerlab.backends.python_backend.optimizer import (
            PythonAdamOptimizer,
            PythonSGDOptimizer,
        )

        if optimizer_type.lower() == "adam":
            return PythonAdamOptimizer(learning_rate=learning_rate)
        return PythonSGDOptimizer(learning_rate=learning_rate)

    if backend_name == "torch":
        from transformerlab.backends.torch_backend.optimizer import (
            create_torch_optimizer,
        )

        # Get PyTorch parameters from the model
        if hasattr(model, "parameters"):
            return create_torch_optimizer(
                optimizer_type, model.parameters(), learning_rate=learning_rate
            )
        # Fallback to numpy optimizer if torch parameters not available
        from transformerlab.backends.numpy_backend.optimizer import (
            create_numpy_optimizer,
        )

        return create_numpy_optimizer(optimizer_type, learning_rate=learning_rate)

    # Default fallback
    from transformerlab.backends.numpy_backend.optimizer import (
        create_numpy_optimizer,
    )

    return create_numpy_optimizer(optimizer_type, learning_rate=learning_rate)


def generate_text(prompt: str, max_length: int = 100) -> str:
    """Generate text from a prompt."""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        return "Model not initialized!"

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    try:
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
    except Exception as e:
        return f"Generation error: {str(e)}"


def get_model_size(model) -> float:
    """Calculate model size in thousands of parameters."""
    try:
        return model.get_parameter_count() / 1000
    except:
        # Fallback calculation
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


def render_model_info():
    """Render model information and metrics."""
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


def render_training_plots():
    """Render training plots and metrics."""
    if st.session_state.model is not None and st.session_state.model.loss_history:
        st.subheader("📊 Training Metrics")

        # Loss plot
        fig = plot_loss_history(st.session_state.model.loss_history)
        st.pyplot(fig)
        plt.close(fig)

        # Attention heatmap
        if st.session_state.model.step_count > 0:
            # Get attention weights from the last forward pass
            # This would require storing the last forward pass results
            st.info("Attention heatmap will be available after training")


def render_model_statistics():
    """Render detailed model statistics."""
    if st.session_state.model is not None:
        st.subheader("📈 Model Statistics")
        model_stats = st.session_state.model.get_model_stats()

        col1, col2 = st.columns(2)

        with col1:
            st.json(model_stats["config"])

        with col2:
            if model_stats.get("layer_stats"):
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


def render_text_generation():
    """Render text generation interface."""
    if st.session_state.model is not None:
        st.subheader("✍️ Text Generation")

        prompt = st.text_input("Enter a prompt:", "First Citizen:")

        if st.button("🎲 Generate"):
            if prompt:
                generated_text = generate_text(prompt)
                st.text_area("Generated Text:", generated_text, height=200)


def render_corpus_info():
    """Render corpus information when no model is loaded."""
    if st.session_state.text:
        st.subheader("📚 Corpus Information")
        st.text(f"Corpus length: {len(st.session_state.text)} characters")
        st.text(f"Vocabulary size: {st.session_state.tokenizer.vocab_size}")
        st.text(f"Sample characters: {st.session_state.tokenizer.chars[:20]}")

        st.subheader("📖 Corpus Preview")
        st.text(st.session_state.text[:500] + "...")


def reset_session():
    """Reset the session state."""
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.text = ""
    st.session_state.experiment_history = []
    st.success("✅ Session reset!")
