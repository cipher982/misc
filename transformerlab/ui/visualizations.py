"""
Interactive visualization components for the Transformer Intuition Lab.
"""

import json
import os

import numpy as np
import streamlit as st
from streamlit.components.v1 import html

from transformerlab.benchmarks import PerformanceBenchmark


def render_performance_comparison():
    """Render interactive performance comparison charts."""
    if not st.session_state.get("experiment_history"):
        st.info(
            "Train some models with different backends to see performance comparisons!"
        )
        return

    # Load D3.js and Plotly
    st.markdown(
        """
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    """,
        unsafe_allow_html=True,
    )

    # Load custom CSS
    css_path = os.path.join(
        os.path.dirname(__file__), "..", "static", "css", "visualizations.css"
    )
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Performance metrics comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Speed Comparison")
        if st.button("Run Benchmark"):
            with st.spinner("Running performance benchmark..."):
                benchmark = PerformanceBenchmark()
                backends = ["python", "numpy", "torch"]

                # Define benchmark configurations
                size_configs = [
                    {
                        "vocab_size": 20,
                        "hidden_dim": 32,
                        "num_layers": 1,
                        "num_heads": 2,
                        "ff_dim": 64,
                    },
                    {
                        "vocab_size": 30,
                        "hidden_dim": 48,
                        "num_layers": 2,
                        "num_heads": 4,
                        "ff_dim": 96,
                    },
                ]

                # Run benchmarks first
                benchmark_results = benchmark.benchmark_model_sizes(
                    size_configs, backends, num_runs=1
                )

                # Then generate comparison analysis
                comparison = benchmark.compare_backends(benchmark_results)

                # Store both raw results and comparison
                st.session_state.benchmark_results = {
                    "raw_results": benchmark_results,
                    "comparison": comparison,
                }
                st.session_state.raw_benchmark_data = benchmark_results

        if st.session_state.get("benchmark_results"):
            # Create interactive chart with JavaScript from raw benchmark data
            raw_results = st.session_state.benchmark_results.get("raw_results", [])

            # Convert benchmark results to chart-ready format
            chart_data = {}
            for result in raw_results:
                if result.backend not in chart_data:
                    chart_data[result.backend] = {
                        "forward_times": [],
                        "memory_usage": [],
                        "parameter_counts": [],
                    }
                chart_data[result.backend]["forward_times"].append(
                    result.forward_time_ms
                )
                chart_data[result.backend]["memory_usage"].append(
                    result.memory_usage_mb
                )
                chart_data[result.backend]["parameter_counts"].append(
                    result.parameter_count
                )

            chart_data_json = json.dumps(chart_data)
            # Get absolute path for JavaScript files
            js_path = os.path.join(
                os.path.dirname(__file__), "..", "static", "js", "performance_charts.js"
            )
            if os.path.exists(js_path):
                with open(js_path) as f:
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
        if st.session_state.get("benchmark_results"):
            # Convert to memory data format for charts
            raw_results = st.session_state.benchmark_results.get("raw_results", [])
            memory_data = {}
            for result in raw_results:
                if result.backend not in memory_data:
                    memory_data[result.backend] = {
                        "parameters_mb": result.memory_usage_mb * 0.3,  # Estimate
                        "activations_mb": result.memory_usage_mb * 0.7,  # Estimate
                        "gradients_mb": 0,  # Not available from current benchmark
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
    st.markdown(
        """
    <script src="https://d3js.org/d3.v7.min.js"></script>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.model:
        model_config = {
            "vocab_size": st.session_state.model.vocab_size,
            "hidden_dim": st.session_state.model.hidden_dim,
            "num_layers": st.session_state.model.num_layers,
            "num_heads": st.session_state.model.num_heads,
            "ff_dim": st.session_state.model.ff_dim,
            "max_seq_len": st.session_state.model.max_seq_len,
            "norm_type": st.session_state.model.norm_type,
            "activation_type": st.session_state.model.activation_type,
            "residual_type": st.session_state.model.residual_type,
            "pos_encoding_type": st.session_state.model.pos_encoding_type,
        }

        config_json = json.dumps(model_config)

        # Get absolute path for JavaScript files
        js_path = os.path.join(
            os.path.dirname(__file__), "..", "static", "js", "network_architecture.js"
        )
        if os.path.exists(js_path):
            with open(js_path) as f:
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
    st.markdown(
        """
    <script src="https://d3js.org/d3.v7.min.js"></script>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.model:
        st.info("Initialize and train a model to see attention patterns!")
        return

    # Sample text for attention visualization
    sample_text = st.text_input(
        "Enter text for attention analysis:", "The quick brown fox"
    )

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
            if (
                hasattr(st.session_state.model, "blocks")
                and len(st.session_state.model.blocks) > 0
            ):
                first_block = st.session_state.model.blocks[0]
                if hasattr(first_block, "attention"):
                    attention_weights = first_block.attention.get_attention_weights()

            if attention_weights is not None:
                # Decode tokens for display
                token_strings = [
                    st.session_state.tokenizer.decode([token]) for token in tokens
                ]

                # Convert to JavaScript-compatible format
                attention_data = {
                    "weights": attention_weights.tolist(),
                    "tokens": token_strings,
                }

                viz_type = st.radio("Visualization Type", ["Heatmap", "Flow Diagram"])

                data_json = json.dumps(attention_data)

                if viz_type == "Heatmap":
                    # Get JavaScript content
                    js_path = os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "static",
                        "js",
                        "attention_visualization.js",
                    )
                    if os.path.exists(js_path):
                        with open(js_path) as f:
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

    st.markdown(
        """
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    """,
        unsafe_allow_html=True,
    )

    # Training loss over time
    loss_data = {
        "timestamps": list(range(len(st.session_state.model.loss_history))),
        "loss": st.session_state.model.loss_history,
        "learning_rate": [0.001]
        * len(st.session_state.model.loss_history),  # Placeholder
    }

    data_json = json.dumps(loss_data)

    # Get JavaScript content
    js_path = os.path.join(
        os.path.dirname(__file__), "..", "static", "js", "performance_charts.js"
    )
    if os.path.exists(js_path):
        with open(js_path) as f:
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
    if hasattr(st.session_state.model, "get_model_stats"):
        try:
            stats = st.session_state.model.get_model_stats()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Current Loss", f"{st.session_state.model.loss_history[-1]:.4f}"
                )
            with col2:
                st.metric("Training Steps", len(st.session_state.model.loss_history))
            with col3:
                st.metric(
                    "Backend",
                    st.session_state.get("current_backend", "unknown").title(),
                )

        except Exception as e:
            st.warning(f"Could not load model statistics: {str(e)}")
