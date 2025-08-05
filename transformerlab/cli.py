"""
Modern CLI for Transformer Intuition Lab using Typer and Rich.
"""

import sys
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

app = typer.Typer(
    name="transformerlab",
    help="🧠 Interactive playground for understanding transformer architectures",
    add_completion=False,
)
console = Console()


@app.command()
def web(
    port: int = typer.Option(
        8501, "--port", "-p", help="Port to run the web interface on"
    ),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    headless: bool = typer.Option(False, "--headless", help="Run in headless mode"),
):
    """Launch the interactive web interface."""
    try:
        import streamlit.web.cli as stcli

        console.print(
            Panel.fit(
                f"🚀 Launching Transformer Intuition Lab on http://{host}:{port}",
                title="[bold blue]Transformer Intuition Lab[/bold blue]",
            )
        )

        # Launch Streamlit with command line options
        sys.argv = [
            "streamlit",
            "run",
            "transformerlab/app.py",
            "--server.port",
            str(port),
            "--server.address",
            host,
        ]
        if headless:
            sys.argv.extend(["--server.headless", "true"])

        stcli.main()

    except ImportError:
        console.print("[red]Error: Streamlit not installed. Run 'uv sync'[/red]")
        raise typer.Exit(1)


@app.command()
def demo():
    """Run the interactive demo showcasing key features."""
    try:
        from demo import main as run_demo

        console.print(
            Panel.fit(
                "🎯 Running Transformer Intuition Lab Demo",
                title="[bold green]Demo Mode[/bold green]",
            )
        )

        run_demo()
    except ImportError:
        console.print("[red]Error: Demo module not found[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    corpus: str = typer.Option(
        "tiny_shakespeare.txt", "--corpus", "-c", help="Corpus file to use"
    ),
    hidden_dim: int = typer.Option(256, "--hidden-dim", help="Hidden dimension size"),
    num_layers: int = typer.Option(
        6, "--layers", "-l", help="Number of transformer layers"
    ),
    num_heads: int = typer.Option(8, "--heads", help="Number of attention heads"),
    norm_type: str = typer.Option("LayerNorm", "--norm", help="Normalization type"),
    activation: str = typer.Option("ReLU", "--activation", help="Activation function"),
    residual: str = typer.Option(
        "Pre-LN", "--residual", help="Residual connection type"
    ),
    pos_encoding: str = typer.Option(
        "Sinusoidal", "--pos-encoding", help="Positional encoding type"
    ),
    steps: int = typer.Option(50, "--steps", "-s", help="Number of training steps"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for loss plot"
    ),
):
    """Train a transformer model with specified configuration."""

    try:
        import matplotlib.pyplot as plt

        from transformerlab.core.tokenizer import load_corpus
        from transformerlab.core.transformer import Transformer
        from transformerlab.viz.plots import plot_loss_history
    except ImportError as e:
        console.print(f"[red]Error importing modules: {e}[/red]")
        raise typer.Exit(1)

    # Load corpus
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Loading corpus...", total=None)
        try:
            text, tokenizer = load_corpus(f"transformerlab/data/{corpus}")
            progress.update(
                task,
                description=f"Loaded corpus: {len(text)} characters, {tokenizer.vocab_size} vocab",
            )
        except FileNotFoundError:
            console.print(f"[red]Error: Corpus file '{corpus}' not found[/red]")
            raise typer.Exit(1)

    # Create model
    console.print("\n🏗️ Creating transformer model:")
    console.print(f"   Hidden dim: {hidden_dim}")
    console.print(f"   Layers: {num_layers}")
    console.print(f"   Heads: {num_heads}")
    console.print(f"   Norm: {norm_type}")
    console.print(f"   Activation: {activation}")
    console.print(f"   Residual: {residual}")
    console.print(f"   Pos encoding: {pos_encoding}")

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        norm_type=norm_type,
        activation_type=activation,
        residual_type=residual,
        pos_encoding_type=pos_encoding,
    )

    # Train model
    console.print(f"\n🎯 Training for {steps} steps...")

    batch_size, seq_len = 2, 128
    tokens = tokenizer.encode(text)

    # Create training data
    x = []
    targets = []
    for i in range(batch_size):
        start_idx = i * seq_len
        end_idx = min(start_idx + seq_len, len(tokens))
        x.append(tokens[start_idx:end_idx])
        if end_idx < len(tokens):
            targets.append(tokens[start_idx + 1 : end_idx + 1])
        else:
            targets.append(tokens[start_idx:end_idx])

    x = np.array(x)
    targets = np.array(targets)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Training...", total=steps)

        for step in range(steps):
            logits, stats = model.forward(x, targets)
            progress.update(task, advance=1)

            if step % 10 == 0:
                progress.update(
                    task, description=f"Training... Loss: {stats['loss']:.4f}"
                )

    # Display results
    console.print("\n✅ Training completed!")
    console.print(f"   Final loss: {model.loss_history[-1]:.4f}")

    # Create results table
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Final Loss", f"{model.loss_history[-1]:.4f}")
    table.add_row("Steps", str(steps))
    table.add_row("Model Size", f"{model.get_model_size():,} parameters")
    console.print(table)

    # Create loss plot
    if output or console.is_interactive:
        fig = plot_loss_history(model.loss_history, "Training Loss")
        if output:
            plt.savefig(output, dpi=150, bbox_inches="tight")
            console.print(f"📊 Loss plot saved to {output}")
        else:
            plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
            console.print("📊 Loss plot saved to training_loss.png")
        plt.close(fig)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt to generate from"),
    model_config: Path | None = typer.Option(
        None, "--config", "-c", help="Model configuration file"
    ),
    max_length: int = typer.Option(
        50, "--max-length", "-m", help="Maximum generation length"
    ),
    temperature: float = typer.Option(
        0.8, "--temperature", "-t", help="Generation temperature"
    ),
    corpus: str = typer.Option(
        "tiny_shakespeare.txt", "--corpus", help="Corpus file to use"
    ),
):
    """Generate text using a trained transformer model."""

    try:
        from transformerlab.core.tokenizer import load_corpus
        from transformerlab.core.transformer import Transformer
    except ImportError as e:
        console.print(f"[red]Error importing modules: {e}[/red]")
        raise typer.Exit(1)

    # Load corpus and create model
    text, tokenizer = load_corpus(f"transformerlab/data/{corpus}")

    # For now, create a simple model (in a real app, you'd load a trained model)
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        norm_type="LayerNorm",
        activation_type="ReLU",
        residual_type="Pre-LN",
        pos_encoding_type="Sinusoidal",
    )

    # Train briefly
    console.print("🎯 Training model briefly for generation...")
    batch_size, seq_len = 2, 64
    tokens = tokenizer.encode(text[: seq_len * batch_size * 2])

    x = np.zeros((batch_size, seq_len), dtype=np.int32)
    targets = np.zeros((batch_size, seq_len), dtype=np.int32)

    for i in range(batch_size):
        start_idx = i * seq_len
        end_idx = min(start_idx + seq_len, len(tokens))
        x[i, : end_idx - start_idx] = tokens[start_idx:end_idx]
        if end_idx < len(tokens):
            targets[i, : end_idx - start_idx] = tokens[start_idx + 1 : end_idx + 1]

    for step in range(20):
        model.forward(x, targets)

    # Generate text
    console.print(f"🎲 Generating text from prompt: '{prompt}'")

    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = np.array([prompt_tokens])

    generated = model.generate(
        prompt_array, max_length=max_length, temperature=temperature
    )
    generated_text = tokenizer.decode(generated[0].tolist())

    console.print(
        Panel(
            generated_text,
            title=f"[bold green]Generated Text (T={temperature})[/bold green]",
            border_style="green",
        )
    )


@app.command()
def info():
    """Show information about the Transformer Intuition Lab."""

    table = Table(title="Transformer Intuition Lab Information")
    table.add_column("Feature", style="cyan")
    table.add_column("Status", style="green")

    table.add_row(
        "Python Version",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    table.add_row("Package Manager", "uv")
    table.add_row("Core Implementation", "Pure NumPy")
    table.add_row("Web Interface", "Streamlit")
    table.add_row("CLI Framework", "Typer + Rich")
    table.add_row("Testing", "pytest")
    table.add_row("Code Quality", "Black + Ruff + MyPy")

    console.print(table)

    console.print("\n[bold]Available Components:[/bold]")
    components = [
        "Normalization: LayerNorm, RMSNorm, None",
        "Residual Connections: Pre-LN, Post-LN, Sandwich",
        "Activation Functions: ReLU, GeLU, Swish, SwiGLU",
        "Positional Encoding: Sinusoidal, RoPE, ALiBi",
        "Attention: Multi-head scaled dot-product",
        "Visualization: Real-time plots and statistics",
    ]

    for component in components:
        console.print(f"  • {component}")


if __name__ == "__main__":
    app()
