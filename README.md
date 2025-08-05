# 🧠 Transformer Intuition Lab

**Interactive playground for understanding transformer architectures**

A hands-on environment where you can poke every layer-norm, activation, and position-encoding trick without hidden abstractions. Built with pure NumPy for maximum transparency and educational value.

## 🎯 Purpose

Give Python-savvy, math-averse developers a hands-on environment where they can see how architectural tweaks change signal statistics, training stability, and generated text. The Lab feels like a physics sandbox, not a black-box model zoo.

## ✨ Features

### 🏗️ **Modular Architecture**
- **Normalization**: LayerNorm, RMSNorm, or None
- **Residual Connections**: Pre-LN, Post-LN, Sandwich
- **Activation Functions**: ReLU, GeLU, Swish, SwiGLU
- **Positional Encoding**: Sinusoidal, RoPE, ALiBi

### 📊 **Live Visualization**
- Real-time loss curves
- Attention heatmaps
- Layer statistics across the network
- Activation distributions
- Gradient flow analysis

### 🎛️ **Interactive Controls**
- Adjust model depth (2-12 layers)
- Tune hidden dimensions (64-512)
- Configure attention heads (2-16)
- Set feed-forward dimensions
- Control training parameters

### 🔬 **Educational Focus**
- Pure NumPy implementation (no PyTorch/TensorFlow)
- Transparent code with inline documentation
- Experiment comparison and saving
- Text generation with temperature control

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd transformerlab

# Install dependencies
pip install -e .
```

### Run the Interactive App

```bash
# Launch the Streamlit interface
streamlit run transformerlab/app.py
```

### Basic Usage

```python
from transformerlab.core.transformer import Transformer
from transformerlab.core.tokenizer import load_corpus

# Load corpus and create tokenizer
text, tokenizer = load_corpus("transformerlab/data/tiny_shakespeare.txt")

# Create transformer model
model = Transformer(
    vocab_size=tokenizer.vocab_size,
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    norm_type="LayerNorm",
    activation_type="ReLU",
    residual_type="Pre-LN",
    pos_encoding_type="Sinusoidal"
)

# Train the model
batch_size, seq_len = 2, 128
tokens = tokenizer.encode(text)
# ... create batches and train

# Generate text
prompt = tokenizer.encode("First Citizen:")
generated = model.generate(prompt, max_length=50, temperature=0.8)
print(tokenizer.decode(generated[0]))
```

## 🎓 Learning Objectives

### 1. **Toggle & See**
- Switch between Pre-LN and Post-LN to observe training stability differences
- Compare LayerNorm vs RMSNorm effects on gradient flow
- Watch how different activations affect signal propagation

### 2. **Probe a Neuron**
- Click on specific layers to see attention patterns
- Examine Q-K dot product statistics
- Observe how RoPE maintains magnitude stability at long sequences

### 3. **Share Insights**
- Save experiment configurations
- Export model snapshots
- Compare multiple runs side-by-side

## 🏗️ Architecture Components

### **Tokenization**
- Character-level tokenizer for educational purposes
- Simple vocabulary building from corpus
- Batch encoding with padding support

### **Embeddings**
- Learnable token embeddings
- Configurable positional encoding
- Support for multiple positional encoding schemes

### **Attention Mechanism**
- Multi-head scaled dot-product attention
- Causal masking for autoregressive training
- Attention weight visualization

### **Feed-Forward Networks**
- Two-layer MLP with configurable activation
- Residual connections with different patterns
- Statistics tracking for analysis

### **Normalization**
- LayerNorm with learnable parameters
- RMSNorm (simplified LayerNorm)
- Optional normalization for ablation studies

## 📈 Visualization Features

### **Training Metrics**
- Real-time loss curves with moving averages
- Per-layer statistics tracking
- Gradient norm monitoring

### **Attention Analysis**
- Heatmaps showing attention patterns
- Head-specific attention visualization
- Position-wise attention analysis

### **Model Statistics**
- Activation distributions
- Weight statistics across layers
- Signal propagation analysis

## 🧪 Experiment Management

### **Configuration Saving**
- Save current model configuration
- Export experiment results
- Share configurations with others

### **Comparison Tools**
- Side-by-side experiment comparison
- Statistical analysis of differences
- Visualization of multiple runs

## 🎯 Target Audience

### **Backend ML Developers**
- Know NumPy/PyTorch but tune by copy-pasting configs
- Want to understand why Pre-LN trains faster
- Need to grasp RoPE vs sinusoidal differences

### **Data Scientists**
- Comfortable in notebooks, light math background
- Read papers but find proofs impenetrable
- Want visual confirmation of theoretical concepts

### **Instructors/Mentors**
- Need classroom demos that run on laptops
- Want single-command setup
- Require reproducible experiments

## 🔧 Technical Details

### **Dependencies**
- NumPy (core computations)
- Streamlit (web interface)
- Matplotlib (visualization)
- PyYAML (configuration)

### **Performance**
- CPU-only implementation
- Optimized for educational clarity
- Supports models up to ~100K parameters
- Fast enough for interactive exploration

### **Limitations**
- No GPU acceleration
- Limited to small models
- Character-level tokenization only
- Single corpus included

## 🚧 Future Enhancements

### **Planned Features**
- [ ] Support for more corpora
- [ ] Subword tokenization
- [ ] Gradient-based optimization
- [ ] More positional encoding schemes
- [ ] Export to PyTorch/HuggingFace

### **Research Directions**
- [ ] Attention pattern analysis tools
- [ ] Model interpretability features
- [ ] Performance profiling
- [ ] Memory usage optimization

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature proposals

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the original "Attention Is All You Need" paper
- Built for educational purposes in transformer architecture
- Thanks to the open-source community for inspiration and tools

---

**Happy exploring! 🧠✨**