# 🧠 Transformer Intuition Lab

**Interactive playground for understanding transformer architectures**

A hands-on environment where you can poke every layer-norm, activation, and position-encoding trick without hidden abstractions. Built with pure NumPy for maximum transparency and educational value.

## 🎯 Purpose

Give Python-savvy, math-averse developers a hands-on environment where they can see how architectural tweaks change signal statistics, training stability, and generated text. The Lab feels like a physics sandbox, not a black-box model zoo.

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (modern Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/transformerlab/transformerlab.git
   cd transformerlab
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Launch the web interface:**
   ```bash
   uv run transformerlab web
   ```

4. **Or use the CLI:**
   ```bash
   # Show available commands
   uv run transformerlab --help
   
   # Train a model
   uv run transformerlab train --layers 4 --steps 100
   
   # Generate text
   uv run transformerlab generate "Hello, world!"
   
   # Run demo
   uv run transformerlab demo
   ```

## 🎮 Features

### Interactive Web Interface
- **Real-time training** with live loss curves
- **Architecture toggles** for normalization, activation, and positional encoding
- **Attention visualization** with heatmaps
- **Experiment comparison** with side-by-side metrics
- **Code inspection** - click any component to see its NumPy implementation

### Command Line Interface
- **Rich terminal output** with progress bars and tables
- **Configuration management** with YAML files
- **Batch training** for multiple experiments
- **Text generation** with temperature and sampling controls

### Core Components
- **Tokenizer**: Character-level tokenization with vocabulary management
- **Normalization**: LayerNorm, RMSNorm, or no normalization
- **Activations**: ReLU, GeLU, Swish, SwiGLU
- **Positional Encoding**: Sinusoidal, RoPE, ALiBi
- **Attention**: Multi-head scaled dot-product attention
- **Feed-Forward**: Configurable activation and residual connections

## 🏗️ Architecture

### Model Components
```
Input → Embeddings → Positional Encoding → Transformer Blocks → Output
                                    ↓
Transformer Block: Norm → Attention → Norm → Feed-Forward → Residual
```

### Available Configurations
- **Normalization**: LayerNorm, RMSNorm, None
- **Residual Layout**: Pre-LN, Post-LN, Sandwich
- **Activation**: ReLU, GeLU, Swish, SwiGLU
- **Positional Encoding**: Sinusoidal, RoPE, ALiBi
- **Attention**: Multi-head with configurable heads and dimensions

## 📊 Visualization Features

- **Loss curves** with real-time updates
- **Attention heatmaps** showing token relationships
- **Layer statistics** (variance, gradient norms)
- **Activation distributions** across layers
- **Model comparison** charts
- **Gradient flow** visualization

## 🧪 Experiment Management

### Save and Load Experiments
```bash
# Save current configuration
uv run transformerlab train --save-config experiment.yaml

# Load and run experiment
uv run transformerlab train --config experiment.yaml
```

### Compare Experiments
```bash
# Train multiple configurations
uv run transformerlab train --norm LayerNorm --save-results layer_norm.json
uv run transformerlab train --norm RMSNorm --save-results rms_norm.json

# Compare results
uv run transformerlab compare layer_norm.json rms_norm.json
```

## 🎓 Learning Objectives

### For ML Engineers
- Understand how architectural choices affect training stability
- See the impact of normalization and residual connections
- Visualize attention patterns and their evolution
- Compare different positional encoding strategies

### For Data Scientists
- Hands-on experience with transformer internals
- Real-time experimentation with model parameters
- Understanding of gradient flow and optimization
- Practical knowledge for model debugging

### For Educators
- Classroom-ready demonstrations
- Interactive visualizations for concepts
- Reproducible experiments with config files
- Code transparency for educational purposes

## 🛠️ Development

### Setup Development Environment
```bash
# Install with development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Run code quality checks
uv run black .
uv run ruff check .
uv run mypy .
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=transformerlab

# Run specific test categories
uv run pytest -m "not slow"
uv run pytest -m integration
```

### Code Quality
- **Black**: Code formatting
- **Ruff**: Fast linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Automated quality checks

## 📈 Performance

- **Pure NumPy implementation** for maximum transparency
- **Optimized for CPU** with vectorized operations
- **Memory efficient** with in-place operations where possible
- **Fast startup** with minimal dependencies

## 🎯 Target Audience

- **Backend ML developers** who want to understand transformer internals
- **Data scientists** comfortable with notebooks and light math
- **Instructors** looking for classroom demos
- **Researchers** prototyping architectural ideas

## 🔧 Technical Details

### Dependencies
- **Core**: NumPy 2.0+, Streamlit 1.32+
- **CLI**: Typer, Rich, Pydantic
- **Development**: Black, Ruff, MyPy, Pre-commit
- **Testing**: Pytest, Coverage

### Architecture
- **Modular design** with clear separation of concerns
- **Type hints** throughout for better IDE support
- **Configuration management** with Pydantic validation
- **Error handling** with informative messages

## 🚀 Future Enhancements

- **GPU acceleration** with CuPy or JAX
- **More corpora** and data loading options
- **Advanced optimizers** (Adam, AdamW, etc.)
- **Model checkpointing** and resuming
- **Distributed training** support
- **Export to PyTorch/TensorFlow** for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

### Development Guidelines
- Follow the existing code style (Black + Ruff)
- Add type hints to all functions
- Include tests for new features
- Update documentation as needed

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for transparent transformer implementations
- Built for educational purposes and research
- Thanks to the open-source community for tools and libraries

---

**Ready to dive deep into transformers? Start with `uv sync` and `uv run transformerlab web`!** 🚀