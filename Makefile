# Transformer Intuition Lab - Makefile
# Quick commands for running the multi-backend transformer system
# REQUIRES: Python 3.13 ONLY

.PHONY: help install test benchmark demo clean lint format

# Python version check - ensures Python 3.13 is being used
PYTHON_VERSION_CHECK := $(shell python --version 2>&1 | grep -c "Python 3.13")
ifeq ($(PYTHON_VERSION_CHECK),0)
    $(error ❌ Python 3.13 required! Current: $(shell python --version 2>&1). Use: python3.13 or update your PATH)
endif

# Default target
help:
	@echo "🧠 Transformer Intuition Lab - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev-install - Install with development dependencies"
	@echo "  make demo        - Run interactive Streamlit demo"
	@echo "  make test        - Test all backends (requires dev-install)"
	@echo "  make benchmark   - Run performance comparison"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test        - Run comprehensive test suite"
	@echo "  make test-backends - Quick backend compatibility test"
	@echo "  make test-training - Test training functionality"
	@echo "  make test-performance - Run performance tests (fast)"
	@echo "  make test-slow   - Run comprehensive slow tests"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make dev-install - Install with development dependencies (required for tests)"
	@echo "  make lint        - Run code linting with ruff"
	@echo "  make format      - Format code with black"  
	@echo "  make check       - Run lint + format (fast, pre-commit ready)"
	@echo "  make setup-precommit - Install pre-commit hooks"
	@echo "  make clean       - Clean generated files"
	@echo ""
	@echo "📊 Analysis:"
	@echo "  make compare     - Quick backend comparison"
	@echo ""

# Installation
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync
	@echo "✅ Installation complete!"

# Development installation with dev dependencies
dev-install:
	@echo "📦 Installing dependencies with dev tools..."
	uv sync --group dev
	@echo "✅ Development installation complete!"

# Main demo application
demo:
	@echo "🚀 Starting Transformer Intuition Lab demo..."
	@echo "📱 Open http://localhost:8501 in your browser"
	uv run streamlit run transformerlab/app.py

# Test all backends (comprehensive)
test:
	@echo "🧪 Running comprehensive test suite..."
	uv run pytest transformerlab/tests/ -v
	@echo "✅ All tests completed!"

# Quick backend test
test-backends:
	@echo "🧪 Testing transformer backends (quick)..."
	uv run python test_backends.py
	@echo "✅ Backend testing complete!"

# Test specific components
test-training:
	@echo "🏋️ Testing training functionality..."
	uv run pytest transformerlab/tests/test_training.py -v

test-performance:
	@echo "⚡ Running performance tests..."
	uv run pytest transformerlab/tests/test_performance.py -v -m "not slow"

test-slow:
	@echo "🐌 Running slow/comprehensive tests..."
	uv run pytest transformerlab/tests/ -v -m "slow"

test-integration:
	@echo "🔗 Running integration tests..."
	uv run pytest transformerlab/tests/ -v -m "integration"

# Performance benchmarking
benchmark:
	@echo "⚡ Running performance benchmarks..."
	uv run python benchmarks.py
	@echo "📊 Results saved to:"
	@echo "  - benchmark_report.txt"
	@echo "  - benchmark_results.json"

# Quick backend comparison
compare:
	@echo "🔍 Quick backend comparison..."
	@uv run python scripts/quick_compare.py

# Test training functionality
training:
	@echo "🏋️ Testing training functionality..."
	@uv run python scripts/quick_training.py

# Development tools
lint:
	@echo "🔍 Running code linting..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check transformerlab/ --fix; \
	else \
		echo "⚠️  ruff not installed, skipping linting"; \
	fi

format:
	@echo "✨ Formatting code with black..."
	@if command -v black >/dev/null 2>&1; then \
		black transformerlab/ *.py; \
	else \
		echo "⚠️  black not installed, skipping formatting"; \
	fi

# Combined quality checks (fast - suitable for pre-commit)
check: lint format
	@echo "✅ Code quality checks completed!"

# Install pre-commit hooks
setup-precommit:
	@echo "⚙️  Setting up pre-commit hooks..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "✅ Pre-commit hooks installed!"; \
		echo "   Hooks will run: make lint && make format"; \
	else \
		echo "⚠️  pre-commit not installed. Install with:"; \
		echo "   uv tool install pre-commit"; \
	fi

# Cleanup
clean:
	@echo "🧹 Cleaning generated files..."
	rm -f benchmark_report.txt
	rm -f benchmark_results.json
	rm -rf __pycache__
	rm -rf transformerlab/__pycache__
	rm -rf transformerlab/*/__pycache__
	rm -rf transformerlab/*/*/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

# Advanced targets
docs:
	@echo "📚 Generating documentation..."
	@echo "📝 API Documentation available in code docstrings"
	@echo "🌐 Run 'make demo' for interactive documentation"

status:
	@echo "📊 System Status:"
	@echo "==============="
	@echo "🐍 Python: $$(python3 --version)"
	@echo "📦 UV: $$(uv --version 2>/dev/null || echo 'Not installed')"
	@echo "🧠 Backends: $$(python3 -c 'from transformerlab.backends.factory import list_backends; print(\", \".join(list_backends()))' 2>/dev/null || echo 'Not available')"
	@echo "📁 Project: $$(pwd)"

# Full workflow for new users
quickstart: install test demo
	@echo ""
	@echo "🎉 Quickstart complete!"
	@echo "========================"
	@echo "✅ Dependencies installed"
	@echo "✅ All backends tested"
	@echo "🚀 Demo is running at http://localhost:8501"