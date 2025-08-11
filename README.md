# 🧠 Transformer Intuition Lab

**Multi-backend transformer implementation for educational and research purposes**

Compare how transformers work across three different implementations:
- **🐍 Pure Python**: Maximum transparency with step-by-step execution logging  
- **⚡ NumPy**: Fast vectorized operations with educational clarity
- **🚀 PyTorch**: Production-ready with GPU support and automatic differentiation

## 🎯 Purpose

Give Python-savvy, math-averse developers a hands-on environment where they can see how architectural tweaks change signal statistics, training stability, and generated text. The Lab feels like a physics sandbox, not a black-box model zoo.

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (modern Python package manager)

### Installation & Demo

```bash
# Install dependencies and run demo
make install
make demo

# Comprehensive test suite
make test

# Quick backend compatibility test
make test-backends

# Performance testing
make test-performance

# Quick performance comparison
make compare

# Full benchmark suite
make benchmark
```

### Manual Usage

```python
from transformerlab.backends.factory import create_transformer

# Create models with different backends
model_numpy = create_transformer("numpy", vocab_size=50, hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128)
model_python = create_transformer("python", vocab_size=50, hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128)  
model_torch = create_transformer("torch", vocab_size=50, hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128)

# Test forward pass
sample_input = [[1, 2, 3, 4, 5]]
sample_targets = [[2, 3, 4, 5, 6]]

logits, stats = model_numpy.forward(sample_input, sample_targets)
print(f"NumPy Loss: {stats['loss']:.4f}")
```

## 📋 Available Commands

Run `make help` to see all available commands:

### 🚀 **Quick Start**
- `make install` - Install dependencies with uv
- `make demo` - Launch interactive Streamlit web app
- `make test` - Test all three backends  
- `make benchmark` - Run comprehensive performance comparison

### 📊 **Analysis**
- `make compare` - Quick backend comparison (fast)
- `make training` - Test training functionality
- `make status` - Show system status

### 🔧 **Development**  
- `make lint` - Run code linting with ruff
- `make format` - Format code with black
- `make clean` - Clean generated files

## 🎯 **What You'll See**

### Backend Performance Comparison
```
Backend Comparison:
==================
  numpy:   0.53ms,  2,884 params, loss=2.9910
 python:   0.86ms,  2,916 params, loss=3.0211  
  torch:  17.93ms,  2,916 params, loss=2.9984
```

### Educational Python Output (Verbose Mode)
```
[PythonTransformer] Forward pass:
  Step 1: Token embedding lookup...
  Step 2: Adding positional encoding...
[PythonAttention] Forward pass: batch_size=1, seq_len=5
  Step 1: Computing Q, K, V projections...
  Step 2: Reshaping for 2 heads...
  Step 3: Computing attention scores...
```

### Performance Benchmarking
```
📊 Performance Summary
Model    Backend  Forward     Memory     Speedup    
tiny     numpy    0.71ms      0.5MB      1.00x      
tiny     python   19.80ms     0.4MB      0.04x      
tiny     torch    1.76ms      41.4MB     0.40x  
```

## 🎮 Features

### Multi-Backend Architecture
- **Mathematical Equivalence**: All backends produce consistent results
- **Educational Transparency**: See exactly how transformers work
- **Performance Analysis**: Compare speed and memory usage
- **Interactive Demo**: Web-based exploration with Streamlit
- **Benchmarking Suite**: Comprehensive performance testing

### Interactive Web Interface
- **Real-time training** with live loss curves and metrics
- **Backend selection** - Switch between NumPy, Python, PyTorch
- **Architecture toggles** for normalization, activation, and positional encoding
- **Advanced JavaScript visualizations** with D3.js and Plotly
- **Interactive attention heatmaps** with multi-head selection
- **Performance comparison charts** with memory and speed analysis
- **Network architecture diagrams** with data flow animation
- **Experiment comparison** with side-by-side metrics

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

### Multi-Backend Structure
```
transformerlab/backends/
├── abstract.py           # Abstract base classes
├── factory.py           # Backend factory & registry
├── numpy_backend/       # Fast NumPy implementation  
├── python_backend/      # Educational Python version
└── torch_backend/       # Production PyTorch version
```

### Model Components
```
Input → Embeddings → Positional Encoding → Transformer Blocks → Output
                                    ↓
Transformer Block: Norm → Attention → Norm → Feed-Forward → Residual
```

### Available Configurations
- **Backends**: NumPy (fast), Python (educational), PyTorch (production)
- **Normalization**: LayerNorm, RMSNorm, None
- **Residual Layout**: Pre-LN, Post-LN, Sandwich
- **Activation**: ReLU, GeLU, Swish, SwiGLU
- **Positional Encoding**: Sinusoidal, RoPE, ALiBi
- **Attention**: Multi-head with configurable heads and dimensions

## 📊 Advanced Visualization Features

### Interactive JavaScript Components
- **Attention heatmaps** with D3.js - clickable token relationships
- **Network architecture diagrams** with animated data flow
- **Performance comparison charts** with Plotly interactivity
- **Real-time training metrics** with live updates
- **Memory usage profiling** with detailed breakdowns
- **Speed vs accuracy scatter plots** for backend comparison

### Educational Visualizations
- **Loss curves** with real-time updates
- **Layer statistics** (variance, gradient norms)
- **Activation distributions** across layers
- **Gradient flow** visualization
- **Attention flow diagrams** showing information pathways
- **Backend comparison dashboards** with interactive controls

## 🧪 Experiment Management

### Save and Load Experiments
```bash
# Save current configuration
uv run transformerlab train --save-config experiment.yaml

# Load and run experiment
uv run transformerlab train --config experiment.yaml
```

### Generated Files

After running benchmarks and tests:
- `benchmark_report.txt` - Human-readable performance analysis
- `benchmark_results.json` - Raw benchmark data for analysis  
- Streamlit runs at `http://localhost:8501`

## 🎓 Educational Purpose

This project demonstrates:
- How transformers work at different levels of abstraction
- Performance trade-offs between implementations  
- The value of vectorization (NumPy vs Pure Python)
- Production considerations (PyTorch's ecosystem)
- Memory usage patterns across approaches

Perfect for students, researchers, and practitioners who want to understand transformer internals!

### For ML Engineers
- Compare backend performance and memory usage
- Understand implementation trade-offs
- See the impact of normalization and residual connections
- Visualize attention patterns and their evolution

### For Data Scientists
- Hands-on experience with transformer internals
- Educational transparency with step-by-step Python logging
- Real-time experimentation with model parameters
- Understanding of gradient flow and optimization

### For Educators
- Three different abstraction levels for teaching
- Classroom-ready demonstrations with make commands
- Interactive visualizations for concepts
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
# Comprehensive test suite with pytest
make test

# Quick backend compatibility test
make test-backends  

# Training functionality tests
make test-training

# Performance and benchmarking tests
make test-performance

# Slow/comprehensive tests
make test-slow

# Integration tests
make test-integration

# Manual pytest usage
uv run pytest transformerlab/tests/ -v
uv run pytest --cov=transformerlab
uv run pytest -m "not slow"
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