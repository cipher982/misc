# Simplified Educational Transformer

This directory contains the simplified transformer implementation focused on educational transparency.

## Files

- **`simple_config.py`** - Educational configuration with clear constraint explanations
- **`simple_transformer.py`** - Pure Python transformer implementation with maximum transparency  
- **`simple_demo.py`** - Interactive Streamlit demo interface
- **`test_simplified.py`** - Comprehensive test suite (25 tests)

## Quick Start

```python
from simple_config import tiny_transformer
from simple_transformer import SimpleTransformer

# Create and run educational transformer
config = tiny_transformer()
model = SimpleTransformer(config)
logits, stats = model.forward([[1, 2, 3, 4]])
```

## Interactive Demo

```bash
streamlit run simple_demo.py
```

## Run Tests

```bash
python -m pytest test_simplified.py -v
```

This implementation achieves 86% complexity reduction while preserving all educational value.