"""
Simple Educational Transformer Demo

A streamlined demonstration of the simplified transformer architecture
focused purely on educational value without complex abstractions.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from config import TransformerConfig, tiny_transformer, small_transformer
from transformer import SimpleTransformer


def main():
    """Main Streamlit application."""
    st.title("🎓 Simple Transformer Lab")
    st.markdown("**Educational transformer implementation with maximum transparency**")
    
    # Configuration Section
    st.sidebar.header("🔧 Configuration")
    
    config_preset = st.sidebar.selectbox(
        "Choose preset:",
        ["Custom", "Tiny (Fast)", "Small (Realistic)"]
    )
    
    if config_preset == "Tiny (Fast)":
        config = tiny_transformer()
    elif config_preset == "Small (Realistic)":
        config = small_transformer()
    else:
        # Custom configuration with educational constraints
        st.sidebar.markdown("**Architecture:**")
        vocab_size = st.sidebar.number_input("Vocabulary Size", value=100, min_value=10, max_value=10000)
        hidden_dim = st.sidebar.number_input("Hidden Dimension", value=64, min_value=8, max_value=512)
        num_heads = st.sidebar.number_input("Number of Heads", value=4, min_value=1, max_value=16)
        num_layers = st.sidebar.number_input("Number of Layers", value=2, min_value=1, max_value=8)
        
        try:
            config = TransformerConfig(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                seq_len=32,
                batch_size=2
            )
        except ValueError as e:
            st.sidebar.error(f"Configuration Error:\\n{e}")
            config = tiny_transformer()  # Fallback
    
    # Display configuration summary
    st.sidebar.markdown("**Current Configuration:**")
    st.sidebar.text(config.summary())
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Forward Pass", "🎯 Generation", "📊 Training", "🔍 Architecture"])
    
    with tab1:
        demo_forward_pass(config)
    
    with tab2:
        demo_generation(config)
    
    with tab3:
        demo_training(config)
    
    with tab4:
        show_architecture(config)


def demo_forward_pass(config: TransformerConfig):
    """Demonstrate forward pass with detailed logging."""
    st.header("📚 Forward Pass Demonstration")
    st.markdown("See every step of the transformer forward pass with detailed logging.")
    
    # Initialize model
    if 'model' not in st.session_state or st.session_state.config != config:
        with st.spinner("Initializing transformer..."):
            st.session_state.model = SimpleTransformer(config, verbose=False)
            st.session_state.config = config
    
    model = st.session_state.model
    
    # Input configuration
    st.subheader("Input Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        seq_length = st.slider("Sequence Length", 1, min(16, config.seq_len), 4)
        batch_size = st.slider("Batch Size", 1, 4, 2)
    
    with col2:
        use_random = st.checkbox("Random Input", value=True)
        if not use_random:
            manual_input = st.text_input("Manual Input (comma-separated token IDs)", "1,2,3,4")
    
    if st.button("🔄 Run Forward Pass", type="primary"):
        # Create input data
        if use_random:
            input_ids = [[random.randint(1, config.vocab_size-1) for _ in range(seq_length)] 
                        for _ in range(batch_size)]
        else:
            try:
                tokens = [int(x.strip()) for x in manual_input.split(",")][:seq_length]
                input_ids = [tokens + [0] * (seq_length - len(tokens))]  # Pad if needed
                batch_size = 1
            except:
                st.error("Invalid manual input. Using random tokens.")
                input_ids = [[random.randint(1, config.vocab_size-1) for _ in range(seq_length)]]
                batch_size = 1
        
        # Display input
        st.subheader("Input Tokens")
        for i, batch in enumerate(input_ids):
            st.write(f"Batch {i+1}: {batch}")
        
        # Forward pass with detailed output
        st.subheader("Forward Pass Steps")
        
        # Capture the verbose output
        model.verbose = True
        
        with st.spinner("Processing..."):
            logits, stats = model.forward(input_ids)
        
        model.verbose = False
        
        # Display results
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Output Shape", f"{len(logits)} × {len(logits[0])} × {len(logits[0][0])}")
        
        with col2:
            st.metric("Final Logits Mean", f"{stats['final_logits_mean']:.4f}")
        
        with col3:
            st.metric("Final Logits Std", f"{stats['final_logits_std']:.4f}")
        
        # Show attention statistics
        if 'layer_stats' in stats:
            st.subheader("Layer Statistics")
            for i, layer_stat in enumerate(stats['layer_stats']):
                st.write(f"**Layer {i+1}:** Attention Mean: {layer_stat['attn_output_mean']:.4f}, FF Mean: {layer_stat['ff_output_mean']:.4f}")


def demo_generation(config: TransformerConfig):
    """Demonstrate text generation."""
    st.header("🎯 Generation Demonstration")
    st.markdown("Watch the transformer generate tokens one by one autoregressively.")
    
    # Initialize model
    if 'model' not in st.session_state or st.session_state.config != config:
        with st.spinner("Initializing transformer..."):
            st.session_state.model = SimpleTransformer(config, verbose=False)
            st.session_state.config = config
    
    model = st.session_state.model
    
    # Generation settings
    st.subheader("Generation Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prompt_text = st.text_input("Prompt (comma-separated token IDs)", "1,2,3")
        
    with col2:
        max_new_tokens = st.slider("Tokens to Generate", 1, 20, 5)
        
    with col3:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0)
    
    if st.button("🎯 Generate Text", type="primary"):
        try:
            # Parse prompt
            prompt_tokens = [int(x.strip()) for x in prompt_text.split(",")]
            prompt_ids = [prompt_tokens]
            
            st.subheader("Generation Process")
            st.write(f"**Prompt:** {prompt_tokens}")
            
            # Generate with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated = model.generate(prompt_ids, max_new_tokens, temperature)
            
            progress_bar.progress(1.0)
            status_text.text("Generation complete!")
            
            # Display results
            st.subheader("Generated Sequence")
            original_length = len(prompt_tokens)
            generated_sequence = generated[0]
            
            # Highlight original vs generated
            original_part = generated_sequence[:original_length]
            new_part = generated_sequence[original_length:]
            
            st.write(f"**Complete:** {generated_sequence}")
            st.write(f"**Original:** {original_part} → **Generated:** {new_part}")
            
        except Exception as e:
            st.error(f"Generation failed: {e}")


def demo_training(config: TransformerConfig):
    """Demonstrate training process."""
    st.header("📊 Training Demonstration")
    st.markdown("Train the transformer on random data and watch the loss evolve.")
    
    # Initialize model
    if 'model' not in st.session_state or st.session_state.config != config:
        with st.spinner("Initializing transformer..."):
            st.session_state.model = SimpleTransformer(config, verbose=False)
            st.session_state.config = config
    
    model = st.session_state.model
    
    # Training settings
    st.subheader("Training Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_steps = st.slider("Training Steps", 1, 50, 10)
        
    with col2:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        
    with col3:
        seq_len = st.slider("Sequence Length", 2, min(12, config.seq_len), 6)
    
    if st.button("📚 Start Training", type="primary"):
        # Create training data
        batch_size = 2
        input_ids = [[random.randint(1, config.vocab_size-1) for _ in range(seq_len)] 
                    for _ in range(batch_size)]
        target_ids = [[random.randint(1, config.vocab_size-1) for _ in range(seq_len)]
                     for _ in range(batch_size)]
        
        st.subheader("Training Progress")
        
        # Training loop with progress tracking
        losses = []
        progress_bar = st.progress(0)
        loss_chart = st.empty()
        
        for step in range(num_steps):
            loss = model.train_step(input_ids, target_ids, learning_rate)
            losses.append(loss)
            
            # Update progress
            progress_bar.progress((step + 1) / num_steps)
            
            # Update loss chart
            if len(losses) > 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(losses, 'b-', linewidth=2)
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Evolution')
                ax.grid(True, alpha=0.3)
                loss_chart.pyplot(fig)
                plt.close(fig)
        
        st.success(f"Training complete! Final loss: {losses[-1]:.6f}")
        
        # Show training statistics
        st.subheader("Training Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Initial Loss", f"{losses[0]:.6f}")
            
        with col2:
            st.metric("Final Loss", f"{losses[-1]:.6f}")
            
        with col3:
            improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
            st.metric("Improvement", f"{improvement:.2f}%")


def show_architecture(config: TransformerConfig):
    """Show transformer architecture details."""
    st.header("🔍 Architecture Overview")
    st.markdown("Explore the transformer's internal structure and parameter distribution.")
    
    # Architecture diagram (text-based)
    st.subheader("Model Architecture")
    
    st.code(f"""
🏗️ Transformer Architecture ({config.total_params:,} parameters)

📥 Input Processing:
   Token Embeddings: {config.vocab_size:,} × {config.hidden_dim} = {config.vocab_size * config.hidden_dim:,}
   Positional Enc:   {config.seq_len:,} × {config.hidden_dim} = {config.seq_len * config.hidden_dim:,}

🔄 Transformer Layers (×{config.num_layers}):
   📍 Multi-Head Attention ({config.num_heads} heads):
      - Query Matrix:  {config.hidden_dim} × {config.hidden_dim} = {config.hidden_dim ** 2:,}
      - Key Matrix:    {config.hidden_dim} × {config.hidden_dim} = {config.hidden_dim ** 2:,}  
      - Value Matrix:  {config.hidden_dim} × {config.hidden_dim} = {config.hidden_dim ** 2:,}
      - Output Matrix: {config.hidden_dim} × {config.hidden_dim} = {config.hidden_dim ** 2:,}
   
   🔢 Feed-Forward Network:
      - Expand:   {config.hidden_dim} × {config.ff_dim} = {config.hidden_dim * config.ff_dim:,}
      - Contract: {config.ff_dim} × {config.hidden_dim} = {config.ff_dim * config.hidden_dim:,}
   
   ⚡ Layer Normalization: 2 × {config.hidden_dim} × 2 = {4 * config.hidden_dim}

📤 Output Processing:
   Final LayerNorm: {config.hidden_dim} × 2 = {config.hidden_dim * 2}
   Output Projection: {config.hidden_dim} × {config.vocab_size:,} = {config.hidden_dim * config.vocab_size:,}
    """, language="text")
    
    # Parameter breakdown
    st.subheader("Parameter Breakdown")
    
    # Calculate detailed parameter counts
    token_params = config.vocab_size * config.hidden_dim
    pos_params = config.seq_len * config.hidden_dim
    
    # Per layer
    attn_params = 4 * (config.hidden_dim ** 2)  # Q, K, V, O matrices
    ff_params = config.hidden_dim * config.ff_dim + config.ff_dim * config.hidden_dim
    norm_params = 4 * config.hidden_dim  # 2 norms × (weight + bias)
    
    layer_params = attn_params + ff_params + norm_params
    all_layer_params = config.num_layers * layer_params
    
    final_norm_params = 2 * config.hidden_dim
    output_params = config.hidden_dim * config.vocab_size
    
    total_params = token_params + pos_params + all_layer_params + final_norm_params + output_params
    
    # Create breakdown chart
    categories = ['Token Embeddings', 'Position Embeddings', 'Transformer Layers', 'Output Projection']
    param_counts = [token_params, pos_params, all_layer_params, output_params + final_norm_params]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    ax1.pie(param_counts, labels=categories, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Parameter Distribution')
    
    # Bar chart
    ax2.bar(categories, param_counts)
    ax2.set_ylabel('Parameter Count')
    ax2.set_title('Parameter Count by Component')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Attention head visualization
    st.subheader("Attention Head Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Heads", config.num_heads)
        
    with col2:
        st.metric("Dimension per Head", config.head_dim)
        
    with col3:
        st.metric("Total Attention Dim", config.hidden_dim)
    
    st.markdown(f"""
    **How Multi-Head Attention Works:**
    
    1. **Split**: The {config.hidden_dim}-dimensional input is split into {config.num_heads} heads of {config.head_dim} dimensions each
    2. **Process**: Each head computes attention independently over its {config.head_dim} dimensions  
    3. **Combine**: All heads are concatenated back to {config.hidden_dim} dimensions
    
    This allows the model to attend to different types of relationships simultaneously!
    """)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Simple Transformer Lab",
        page_icon="🎓",
        layout="wide"
    )
    
    main()