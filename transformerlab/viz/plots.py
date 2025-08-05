"""
Visualization functions for the Transformer Intuition Lab.
Uses matplotlib for plotting.
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for Streamlit


def plot_loss_history(
    loss_history: list[float], title: str = "Training Loss"
) -> plt.Figure:
    """Plot training loss over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if loss_history:
        steps = range(1, len(loss_history) + 1)
        ax.plot(steps, loss_history, "b-", linewidth=2, alpha=0.8)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add moving average
        if len(loss_history) > 10:
            window = min(10, len(loss_history) // 4)
            moving_avg = np.convolve(
                loss_history, np.ones(window) / window, mode="valid"
            )
            ax.plot(
                range(window, len(loss_history) + 1),
                moving_avg,
                "r--",
                linewidth=2,
                alpha=0.7,
                label=f"{window}-step moving average",
            )
            ax.legend()

    plt.tight_layout()
    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray, title: str = "Attention Weights"
) -> plt.Figure:
    """Plot attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Take the first head for visualization
    if attention_weights.ndim == 4:
        attention_weights = attention_weights[0, 0]  # First batch, first head

    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    plt.tight_layout()
    return fig


def plot_layer_statistics(
    layer_stats: list[dict], stat_name: str, title: str = None
) -> plt.Figure:
    """Plot statistics across layers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = range(len(layer_stats))
    values = []

    for layer_stat in layer_stats:
        if "attention" in layer_stat and stat_name in layer_stat["attention"]:
            values.append(layer_stat["attention"][stat_name])
        elif "feed_forward" in layer_stat and stat_name in layer_stat["feed_forward"]:
            values.append(layer_stat["feed_forward"][stat_name])
        else:
            values.append(0)

    ax.plot(layers, values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(stat_name.replace("_", " ").title())
    ax.set_title(title or f"{stat_name.replace('_', ' ').title()} Across Layers")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_activation_distribution(
    activation_values: np.ndarray, activation_type: str
) -> plt.Figure:
    """Plot distribution of activation values."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Flatten the array
    flat_values = activation_values.flatten()

    # Create histogram
    ax.hist(flat_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{activation_type} Activation Distribution")
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_val = np.mean(flat_values)
    std_val = np.std(flat_values)
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
    ax.axvline(
        mean_val + std_val,
        color="orange",
        linestyle=":",
        label=f"+1σ: {mean_val + std_val:.3f}",
    )
    ax.axvline(
        mean_val - std_val,
        color="orange",
        linestyle=":",
        label=f"-1σ: {mean_val - std_val:.3f}",
    )
    ax.legend()

    plt.tight_layout()
    return fig


def plot_comparison_chart(
    configs: list[dict], metrics: list[str], title: str = "Model Comparison"
) -> plt.Figure:
    """Plot comparison chart for different model configurations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(configs))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        values = [config.get(metric, 0) for config in configs]
        ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels([f"Config {i+1}" for i in range(len(configs))])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_gradient_flow(gradients: list[float], layer_names: list[str]) -> plt.Figure:
    """Plot gradient flow through layers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(gradients))
    ax.plot(x, gradients, "o-", linewidth=2, markersize=8, color="red")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Flow Through Layers")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45)
    ax.grid(True, alpha=0.3)

    # Add threshold lines
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="Normal")
    ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="Vanishing")
    ax.axhline(y=10.0, color="red", linestyle="--", alpha=0.7, label="Exploding")
    ax.legend()

    plt.tight_layout()
    return fig


def create_dashboard_plots(model_stats: dict) -> dict[str, plt.Figure]:
    """Create all dashboard plots from model statistics."""
    plots = {}

    # Loss history
    if model_stats.get("loss_history"):
        plots["loss"] = plot_loss_history(model_stats["loss_history"])

    # Layer statistics
    if model_stats.get("layer_stats"):
        layer_stats = model_stats["layer_stats"]

        # Attention statistics
        plots["attention_weights_mean"] = plot_layer_statistics(
            layer_stats,
            "attention_weights_mean",
            "Mean Attention Weights Across Layers",
        )

        # Feed-forward statistics
        plots["output_mean"] = plot_layer_statistics(
            layer_stats, "output_mean", "Mean Output Across Layers"
        )

    # Attention heatmap (from first layer)
    if (
        model_stats.get("layer_stats")
        and model_stats["layer_stats"][0].get("attention", {}).get("attention_weights")
        is not None
    ):
        attention_weights = model_stats["layer_stats"][0]["attention"][
            "attention_weights"
        ]
        plots["attention_heatmap"] = plot_attention_heatmap(attention_weights)

    return plots


def save_plots_to_file(plots: dict[str, plt.Figure], filename: str):
    """Save multiple plots to a single file."""
    n_plots = len(plots)
    if n_plots == 0:
        return

    # Calculate grid dimensions
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    plot_items = list(plots.items())
    for i, (name, plot_fig) in enumerate(plot_items):
        row = i // cols
        col = i % cols

        # Copy the plot to the subplot
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.clear()

        # Get the plot data and recreate it
        for line in plot_fig.gca().lines:
            ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
            )

        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_plots, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
