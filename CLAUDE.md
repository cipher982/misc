# Transformer Intuition Lab

## What it is
An educational toolkit that exposes the math and computation behind transformers—no black boxes.

## Current focus
- Active: simplified single-implementation in the repo root
- Legacy: `transformerlab/` multi-backend (Python/NumPy/PyTorch) kept for reference; de-emphasized due to type/import complexity

## Quick start
```bash
uv run python simple_transformer.py        # Run demo
uv run python -m pytest test_simplified.py # Run tests
uv run streamlit run simple_demo.py        # Launch UI
```

## Key files
- `simple_transformer.py`: plain-Python transformer showing every operation
- `simple_config.py`: minimal config with constraint explanations
- `simple_demo.py`: Streamlit UI
- `test_simplified.py`: 25-test validation suite

## Design principles
- Single implementation for clarity (no factories or backends)
- Explicit loops over vectorization to reveal mechanics
- Clear, educational error messages

## Learning and visualization
- Real-time UI to tweak architecture and training parameters
- Attention maps, architecture diagrams, and training metrics

## Testing guidance
- Prefer `test_simplified.py`; the original `transformerlab/tests/` has import dependency issues
- Tests emphasize mathematical correctness

## Environment
- Python 3.13
- Managed with uv (formatting, linting, and dependencies)