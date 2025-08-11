"""
Experiment management UI components for the Transformer Intuition Lab.
"""

import matplotlib.pyplot as plt
import streamlit as st


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