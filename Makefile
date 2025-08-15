# Transformer Intuition Lab - Simplified
# Educational transformer implementation with maximum transparency
# REQUIRES: Python 3.13 ONLY

.PHONY: help install demo test benchmark clean check

# Default target
help:
	@echo "🧠 Transformer Intuition Lab - Simplified"
	@echo "========================================"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install - Install dependencies"
	@echo "  make demo    - Run educational transformer demo"
	@echo "  make test    - Run unit tests"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test           - Unit tests (25 tests)"
	@echo "  make test-all       - All tests (unit + e2e + integration)"
	@echo "  make test-e2e       - E2E UI tests with Playwright"
	@echo "  make test-performance - Performance benchmarks"
	@echo "  make test-integration - Integration tests"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make check   - Lint and format code"
	@echo "  make clean   - Clean generated files"
	@echo ""

# Installation
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync
	@echo "✅ Installation complete!"

# Educational demo
demo:
	@echo "🚀 Running Transformer Demo..."
	uv run python transformer.py

# Interactive UI demo
ui:
	@echo "🚀 Starting Interactive Demo UI..."
	@echo "📱 Open http://localhost:8501 in your browser"
	uv run streamlit run app.py

# Test suite
test:
	@echo "🧪 Running unit tests..."
	uv run python -m pytest tests/test_simplified.py -v

# Run all tests including E2E
test-all:
	@echo "🧪 Running comprehensive test suite..."
	uv run python -m pytest tests/ -v

# E2E tests for Streamlit UI
test-e2e:
	@echo "🌐 Running E2E tests with Playwright..."
	uv run playwright install chromium --with-deps
	uv run python -m pytest tests/test_e2e.py -v --headed

# Performance and benchmark tests
test-performance:
	@echo "⚡ Running performance tests..."
	uv run python -m pytest tests/test_e2e.py::TestPerformanceBenchmarks -v

# Integration tests
test-integration:
	@echo "🔗 Running integration tests..."
	uv run python -m pytest tests/test_e2e.py::TestFullPipeline -v


# Code quality
check:
	@echo "🔍 Linting and formatting..."
	uv run ruff check *.py tests/*.py --fix
	uv run ruff format *.py tests/*.py
	@echo "✅ Code quality checks completed!"

# Cleanup
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"