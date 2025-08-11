"""
Main Streamlit application for the Transformer Intuition Lab.
"""

import json
import os
import time
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit.components.v1 import html

from transformerlab.backends.factory import create_transformer, list_backends, get_backend_info
from transformerlab.core.tokenizer import load_corpus
from transformerlab.viz.plots import plot_loss_history
from transformerlab.benchmarks import PerformanceBenchmark


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

    # Add visualization tabs
    if st.session_state.model is not None:
        st.subheader("📈 Interactive Visualizations")
        
        viz_tabs = st.tabs(["Performance", "Architecture", "Attention", "Training"])
        
        with viz_tabs[0]:
            render_performance_comparison()
            
        with viz_tabs[1]:
            render_architecture_visualization()
            
        with viz_tabs[2]:
            render_attention_visualization()
            
        with viz_tabs[3]:
            render_training_metrics()
    
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

        # Get current backend
        backend_name = st.session_state.get('current_backend', 'numpy')
        
        # Create model with selected backend
        # Use verbose=False for python backend in web interface to keep console clean
        verbose = backend_name != "python"
        
        model = create_transformer(
            backend_name=backend_name,
            vocab_size=tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            norm_type=norm_type,
            activation_type=activation_type,
            residual_type=residual_type,
            pos_encoding_type=pos_encoding_type,
            verbose=verbose,
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

        # Initialize optimizer if not exists
        if not hasattr(st.session_state, 'optimizer') or st.session_state.optimizer is None:
            backend_name = st.session_state.get('current_backend', 'numpy')
            st.session_state.optimizer = create_backend_optimizer(backend_name, model)
        
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


def create_backend_optimizer(backend_name: str, model, optimizer_type: str = "adam", learning_rate: float = 0.001):
    """Create optimizer appropriate for the backend."""
    if backend_name == "numpy":
        from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
        return create_numpy_optimizer(optimizer_type, learning_rate=learning_rate)
    
    elif backend_name == "python":
        from transformerlab.backends.python_backend.optimizer import PythonAdamOptimizer, PythonSGDOptimizer
        if optimizer_type.lower() == "adam":
            return PythonAdamOptimizer(learning_rate=learning_rate)
        else:
            return PythonSGDOptimizer(learning_rate=learning_rate)
    
    elif backend_name == "torch":
        from transformerlab.backends.torch_backend.optimizer import create_torch_optimizer
        # Get PyTorch parameters from the model
        if hasattr(model, 'parameters'):
            return create_torch_optimizer(optimizer_type, model.parameters(), learning_rate=learning_rate)
        else:
            # Fallback to numpy optimizer if torch parameters not available
            from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
            return create_numpy_optimizer(optimizer_type, learning_rate=learning_rate)
    
    else:
        # Default fallback
        from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
        return create_numpy_optimizer(optimizer_type, learning_rate=learning_rate)


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


def render_performance_comparison():
    """Render interactive performance comparison charts."""
    if not st.session_state.get('experiment_history'):
        st.info("Train some models with different backends to see performance comparisons!")
        return
    
    # Load D3.js and Plotly
    st.markdown("""
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    """, unsafe_allow_html=True)
    
    # Load custom CSS
    css_path = os.path.join(os.path.dirname(__file__), 'static', 'css', 'visualizations.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Performance metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Speed Comparison")
        if st.button("Run Benchmark"):
            with st.spinner("Running performance benchmark..."):
                benchmark = PerformanceBenchmark()
                backends = ['python', 'numpy', 'torch']
                
                # Define benchmark configurations
                size_configs = [
                    {"vocab_size": 20, "hidden_dim": 32, "num_layers": 1, "num_heads": 2, "ff_dim": 64},
                    {"vocab_size": 30, "hidden_dim": 48, "num_layers": 2, "num_heads": 4, "ff_dim": 96}
                ]
                
                # Run benchmarks first
                benchmark_results = benchmark.benchmark_model_sizes(size_configs, backends, num_runs=1)
                
                # Then generate comparison analysis
                comparison = benchmark.compare_backends(benchmark_results)
                
                # Store both raw results and comparison
                st.session_state.benchmark_results = {
                    'raw_results': benchmark_results,
                    'comparison': comparison
                }
                st.session_state.raw_benchmark_data = benchmark_results
        
        if st.session_state.get('benchmark_results'):
            # Create interactive chart with JavaScript from raw benchmark data
            raw_results = st.session_state.benchmark_results.get('raw_results', [])
            
            # Convert benchmark results to chart-ready format
            chart_data = {}
            for result in raw_results:
                if result.backend not in chart_data:
                    chart_data[result.backend] = {
                        'forward_times': [],
                        'memory_usage': [],
                        'parameter_counts': []
                    }
                chart_data[result.backend]['forward_times'].append(result.forward_time_ms)
                chart_data[result.backend]['memory_usage'].append(result.memory_usage_mb)
                chart_data[result.backend]['parameter_counts'].append(result.parameter_count)
            
            chart_data_json = json.dumps(chart_data)
            # Get absolute path for JavaScript files
            js_path = os.path.join(os.path.dirname(__file__), 'static', 'js', 'performance_charts.js')
            if os.path.exists(js_path):
                with open(js_path, 'r') as f:
                    js_content = f.read()
            else:
                js_content = "console.log('JavaScript file not found');"
                
            html_content = f"""
            <div id="speed-chart" style="width:100%;height:400px;"></div>
            <script>{js_content}</script>
            <script>
                if (typeof PerformanceCharts !== 'undefined') {{
                    const charts = new PerformanceCharts();
                    charts.renderSpeedComparison('speed-chart', {chart_data_json});
                }} else {{
                    document.getElementById('speed-chart').innerHTML = '<p>Performance charts not available</p>';
                }}
            </script>
            """
            html(html_content, height=450)
    
    with col2:
        st.subheader("🧠 Memory Usage")
        if st.session_state.get('benchmark_results'):
            # Convert to memory data format for charts
            raw_results = st.session_state.benchmark_results.get('raw_results', [])
            memory_data = {}
            for result in raw_results:
                if result.backend not in memory_data:
                    memory_data[result.backend] = {
                        'parameters_mb': result.memory_usage_mb * 0.3,  # Estimate
                        'activations_mb': result.memory_usage_mb * 0.7,  # Estimate  
                        'gradients_mb': 0  # Not available from current benchmark
                    }
            memory_data_json = json.dumps(memory_data)
            html_content = f"""
            <div id="memory-chart" style="width:100%;height:400px;"></div>
            <script>
                if (typeof PerformanceCharts !== 'undefined') {{
                    const charts = new PerformanceCharts();
                    charts.renderMemoryComparison('memory-chart', {memory_data_json});
                }} else {{
                    document.getElementById('memory-chart').innerHTML = '<p>Memory charts not available</p>';
                }}
            </script>
            """
            html(html_content, height=450)


def render_architecture_visualization():
    """Render interactive network architecture visualization."""
    st.markdown("""
    <script src="https://d3js.org/d3.v7.min.js"></script>
    """, unsafe_allow_html=True)
    
    if st.session_state.model:
        model_config = {
            'vocab_size': st.session_state.model.vocab_size,
            'hidden_dim': st.session_state.model.hidden_dim,
            'num_layers': st.session_state.model.num_layers,
            'num_heads': st.session_state.model.num_heads,
            'ff_dim': st.session_state.model.ff_dim,
            'max_seq_len': st.session_state.model.max_seq_len,
            'norm_type': st.session_state.model.norm_type,
            'activation_type': st.session_state.model.activation_type,
            'residual_type': st.session_state.model.residual_type,
            'pos_encoding_type': st.session_state.model.pos_encoding_type,
        }
        
        config_json = json.dumps(model_config)
        
        # Get absolute path for JavaScript files
        js_path = os.path.join(os.path.dirname(__file__), 'static', 'js', 'network_architecture.js')
        if os.path.exists(js_path):
            with open(js_path, 'r') as f:
                js_content = f.read()
        else:
            js_content = "console.log('JavaScript file not found');"
        
        html_content = f"""
        <div id="architecture-viz" style="width:100%;height:600px;"></div>
        <script>{js_content}</script>
        <script>
            if (typeof NetworkArchitectureViz !== 'undefined') {{
                const archViz = new NetworkArchitectureViz('architecture-viz');
                archViz.renderArchitecture({config_json});
            }} else {{
                document.getElementById('architecture-viz').innerHTML = '<p>Architecture visualization not available</p>';
            }}
        </script>
        """
        html(html_content, height=650)
    else:
        st.info("Initialize a model to see the architecture visualization!")


def render_attention_visualization():
    """Render interactive attention visualization."""
    st.markdown("""
    <script src="https://d3js.org/d3.v7.min.js"></script>
    """, unsafe_allow_html=True)
    
    if not st.session_state.model:
        st.info("Initialize and train a model to see attention patterns!")
        return
    
    # Sample text for attention visualization
    sample_text = st.text_input("Enter text for attention analysis:", "The quick brown fox")
    
    if sample_text and st.button("Analyze Attention"):
        try:
            # Encode text
            tokens = st.session_state.tokenizer.encode(sample_text)
            if len(tokens) > 20:
                tokens = tokens[:20]  # Limit for visualization
                
            tokens_array = np.array([tokens])
            
            # Get attention weights
            logits, stats = st.session_state.model.forward(tokens_array)
            
            # Try to get attention weights from model
            attention_weights = None
            if hasattr(st.session_state.model, 'blocks') and len(st.session_state.model.blocks) > 0:
                first_block = st.session_state.model.blocks[0]
                if hasattr(first_block, 'attention'):
                    attention_weights = first_block.attention.get_attention_weights()
            
            if attention_weights is not None:
                # Decode tokens for display
                token_strings = [st.session_state.tokenizer.decode([token]) for token in tokens]
                
                # Convert to JavaScript-compatible format
                attention_data = {
                    'weights': attention_weights.tolist(),
                    'tokens': token_strings
                }
                
                viz_type = st.radio("Visualization Type", ["Heatmap", "Flow Diagram"])
                
                data_json = json.dumps(attention_data)
                
                if viz_type == "Heatmap":
                    # Get JavaScript content
                    js_path = os.path.join(os.path.dirname(__file__), 'static', 'js', 'attention_visualization.js')
                    if os.path.exists(js_path):
                        with open(js_path, 'r') as f:
                            js_content = f.read()
                    else:
                        js_content = "console.log('JavaScript file not found');"
                        
                    html_content = f"""
                    <div id="attention-viz" style="width:100%;height:600px;"></div>
                    <script>{js_content}</script>
                    <script>
                        if (typeof AttentionVisualizer !== 'undefined') {{
                            const attentionViz = new AttentionVisualizer('attention-viz');
                            const data = {data_json};
                            // Simplified attention rendering
                            attentionViz.renderHeatmap(data.weights, data.tokens);
                        }} else {{
                            document.getElementById('attention-viz').innerHTML = '<p>Attention visualization not available</p>';
                        }}
                    </script>
                    """
                else:
                    html_content = f"""
                    <div id="attention-flow" style="width:100%;height:600px;"></div>
                    <script>{js_content}</script>
                    <script>
                        if (typeof AttentionVisualizer !== 'undefined') {{
                            const attentionViz = new AttentionVisualizer('attention-flow');
                            const data = {data_json};
                            // Simplified flow diagram rendering
                            attentionViz.renderFlowDiagram(data.weights, data.tokens);
                        }} else {{
                            document.getElementById('attention-flow').innerHTML = '<p>Attention flow visualization not available</p>';
                        }}
                    </script>
                    """
                
                html(html_content, height=650)
            else:
                st.warning("Attention weights not available for this backend.")
                
        except Exception as e:
            st.error(f"Error analyzing attention: {str(e)}")


def render_training_metrics():
    """Render real-time training metrics."""
    if not st.session_state.model or not st.session_state.model.loss_history:
        st.info("Train the model to see training metrics!")
        return
    
    st.markdown("""
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    """, unsafe_allow_html=True)
    
    # Training loss over time
    loss_data = {
        'timestamps': list(range(len(st.session_state.model.loss_history))),
        'loss': st.session_state.model.loss_history,
        'learning_rate': [0.001] * len(st.session_state.model.loss_history)  # Placeholder
    }
    
    data_json = json.dumps(loss_data)
    
    # Get JavaScript content
    js_path = os.path.join(os.path.dirname(__file__), 'static', 'js', 'performance_charts.js')
    if os.path.exists(js_path):
        with open(js_path, 'r') as f:
            js_content = f.read()
    else:
        js_content = "console.log('JavaScript file not found');"
    
    html_content = f"""
    <div id="training-metrics" style="width:100%;height:400px;"></div>
    <script>{js_content}</script>
    <script>
        if (typeof PerformanceCharts !== 'undefined') {{
            const charts = new PerformanceCharts();
            charts.renderRealtimeMetrics('training-metrics', {data_json});
        }} else {{
            document.getElementById('training-metrics').innerHTML = '<p>Training metrics visualization not available</p>';
        }}
    </script>
    """
    html(html_content, height=450)
    
    # Model statistics
    if hasattr(st.session_state.model, 'get_model_stats'):
        try:
            stats = st.session_state.model.get_model_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Loss", f"{st.session_state.model.loss_history[-1]:.4f}")
            with col2:
                st.metric("Training Steps", len(st.session_state.model.loss_history))
            with col3:
                st.metric("Backend", st.session_state.get('current_backend', 'unknown').title())
                
        except Exception as e:
            st.warning(f"Could not load model statistics: {str(e)}")


if __name__ == "__main__":
    main()
