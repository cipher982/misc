"""
Configuration sidebar UI components for the Transformer Intuition Lab.
"""

import streamlit as st

from transformerlab.backends.factory import list_backends, get_backend_info


def render_config_sidebar():
    """Render the configuration sidebar with all model settings."""
    with st.sidebar:
        st.header("🎛️ Model Configuration")
        
        # Backend selection
        st.subheader("🔧 Backend")
        backend_options = list_backends()
        backend_descriptions = {name: get_backend_info(name)['description'] for name in backend_options}
        
        selected_backend = st.selectbox(
            "Select Backend",
            backend_options,
            help="Choose the transformer implementation backend"
        )
        
        # Show backend info
        backend_info = get_backend_info(selected_backend)
        st.info(f"**{backend_info['description']}**\n\n{backend_info['features']}")
        
        if st.session_state.get('current_backend') != selected_backend:
            st.session_state.current_backend = selected_backend
            st.session_state.model = None  # Reset model when backend changes
            st.session_state.optimizer = None  # Reset optimizer when backend changes

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

    return {
        'corpus_file': corpus_file,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'ff_dim': ff_dim,
        'norm_type': norm_type,
        'residual_type': residual_type,
        'activation_type': activation_type,
        'pos_encoding_type': pos_encoding_type,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'num_steps': num_steps,
    }


def render_action_buttons(config):
    """Render action buttons for model initialization and training."""
    with st.sidebar:
        st.subheader("🚀 Actions")
        col1, col2 = st.columns(2)

        action_taken = None

        with col1:
            if st.button("🔄 Initialize Model"):
                action_taken = ('initialize', config)

        with col2:
            if st.button("🎯 Train Model"):
                if st.session_state.model is not None:
                    action_taken = ('train', config)
                else:
                    st.error("Please initialize the model first!")

        if st.button("🗑️ Reset"):
            action_taken = ('reset', None)

        return action_taken


def render_experiment_controls():
    """Render experiment management controls."""
    with st.sidebar:
        st.subheader("📊 Experiments")
        
        action_taken = None
        
        if st.button("💾 Save Experiment"):
            action_taken = ('save_experiment', None)

        if st.button("📈 Compare Experiments"):
            action_taken = ('compare_experiments', None)

        return action_taken