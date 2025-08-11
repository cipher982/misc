"""
Main Streamlit application for the Transformer Intuition Lab.
"""

import streamlit as st

from transformerlab.ui.config_sidebar import (
    render_config_sidebar,
    render_action_buttons,
    render_experiment_controls,
)
from transformerlab.ui.experiments import save_experiment, compare_experiments
from transformerlab.ui.training import (
    initialize_model,
    train_model,
    render_model_info,
    render_training_plots,
    render_model_statistics,
    render_text_generation,
    render_corpus_info,
    reset_session,
)
from transformerlab.ui.visualizations import (
    render_performance_comparison,
    render_architecture_visualization,
    render_attention_visualization,
    render_training_metrics,
)


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

    # Render sidebar configuration
    config = render_config_sidebar()
    
    # Handle action buttons
    action = render_action_buttons(config)
    if action:
        action_type, action_config = action
        if action_type == 'initialize':
            initialize_model(action_config)
        elif action_type == 'train':
            train_model(action_config)
        elif action_type == 'reset':
            reset_session()
    
    # Handle experiment controls
    experiment_action = render_experiment_controls()
    if experiment_action:
        exp_type, _ = experiment_action
        if exp_type == 'save_experiment':
            save_experiment()
        elif exp_type == 'compare_experiments':
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
        # Render model information and metrics
        render_model_info()
        
        # Render training plots
        render_training_plots()
        
        # Render model statistics
        render_model_statistics()
        
        # Render text generation interface
        render_text_generation()

    else:
        st.info(
            "👈 Use the sidebar to configure and initialize your transformer model!"
        )

        # Show corpus info
        render_corpus_info()


if __name__ == "__main__":
    main()
