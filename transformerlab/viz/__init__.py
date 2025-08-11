"""
Visualization components for the Transformer Intuition Lab.
"""

from .plots import (
    create_dashboard_plots,
    plot_activation_distribution,
    plot_attention_heatmap,
    plot_comparison_chart,
    plot_gradient_flow,
    plot_layer_statistics,
    plot_loss_history,
    save_plots_to_file,
)

__all__ = [
    "plot_loss_history",
    "plot_attention_heatmap",
    "plot_layer_statistics",
    "plot_activation_distribution",
    "plot_comparison_chart",
    "plot_gradient_flow",
    "create_dashboard_plots",
    "save_plots_to_file",
]
