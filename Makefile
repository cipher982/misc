# Transformer Intuition Lab - Makefile
# Quick commands for running the multi-backend transformer system

.PHONY: help install test benchmark demo clean lint format

# Default target
help:
	@echo "🧠 Transformer Intuition Lab - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install     - Install dependencies"
	@echo "  make demo        - Run interactive Streamlit demo"
	@echo "  make test        - Test all backends (numpy, python, torch)"
	@echo "  make benchmark   - Run performance comparison"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make lint        - Run code linting with ruff"
	@echo "  make format      - Format code with black"  
	@echo "  make check       - Run lint + format (fast, pre-commit ready)"
	@echo "  make setup-precommit - Install pre-commit hooks"
	@echo "  make clean       - Clean generated files"
	@echo ""
	@echo "📊 Analysis:"
	@echo "  make compare     - Quick backend comparison"
	@echo "  make training    - Test training functionality"
	@echo ""

# Installation
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync
	@echo "✅ Installation complete!"

# Main demo application
demo:
	@echo "🚀 Starting Transformer Intuition Lab demo..."
	@echo "📱 Open http://localhost:8501 in your browser"
	uv run streamlit run transformerlab/app.py

# Test all backends
test:
	@echo "🧪 Testing all transformer backends..."
	uv run python test_backends.py
	@echo "✅ Backend testing complete!"

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