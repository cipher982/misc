"""
End-to-end tests for Streamlit UI using Playwright.

Tests all user interactions with the educational transformer demo.
"""

import subprocess
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


class TestStreamlitApp:
    """E2E tests for the Streamlit transformer demo."""
    
    @pytest.fixture(scope="class")
    def streamlit_server(self):
        """Start Streamlit server for testing."""
        # Start Streamlit in background
        process = subprocess.Popen([
            "uv", "run", "streamlit", "run", "app.py",
            "--server.port", "8502",  # Different port for testing
            "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        yield "http://localhost:8502"
        
        # Cleanup
        process.terminate()
        process.wait()

    def test_app_loads(self, page: Page, streamlit_server):
        """Test that the Streamlit app loads successfully."""
        page.goto(streamlit_server)
        
        # Check for main title
        expect(page.locator("text=Transformer Intuition Lab")).to_be_visible()
        
    def test_config_sidebar(self, page: Page, streamlit_server):
        """Test configuration sidebar functionality."""
        page.goto(streamlit_server)
        
        # Test architecture configuration
        vocab_slider = page.locator("input[data-testid='stSlider'][aria-label*='vocab']")
        if vocab_slider.count() > 0:
            vocab_slider.fill("500")
            
        # Test that configuration updates are reflected
        expect(page.locator("text=Architecture")).to_be_visible()
        
    def test_training_demo(self, page: Page, streamlit_server):
        """Test training demonstration functionality."""
        page.goto(streamlit_server)
        
        # Look for training controls
        train_button = page.locator("button", has_text="Train")
        if train_button.count() > 0:
            train_button.click()
            
            # Wait for training to complete
            expect(page.locator("text=Loss")).to_be_visible(timeout=10000)
            
    def test_generation_demo(self, page: Page, streamlit_server):
        """Test text generation functionality."""
        page.goto(streamlit_server)
        
        # Look for generation controls
        generate_button = page.locator("button", has_text="Generate")
        if generate_button.count() > 0:
            generate_button.click()
            
            # Check that generation produces output
            expect(page.locator("text=Generated")).to_be_visible(timeout=5000)
            
    def test_visualization_components(self, page: Page, streamlit_server):
        """Test that visualization components render."""
        page.goto(streamlit_server)
        
        # Check for charts/plots
        chart_elements = page.locator("[data-testid='stPlotlyChart'], canvas, svg")
        
        # Should have at least one visualization
        expect(chart_elements.first).to_be_visible(timeout=5000)
        
    def test_responsive_design(self, page: Page, streamlit_server):
        """Test responsive design on different screen sizes."""
        page.goto(streamlit_server)
        
        # Test mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})
        expect(page.locator("text=Transformer Intuition Lab")).to_be_visible()
        
        # Test desktop viewport
        page.set_viewport_size({"width": 1200, "height": 800})
        expect(page.locator("text=Transformer Intuition Lab")).to_be_visible()


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance tests using the benchmarks.py system."""
    
    def test_benchmark_execution(self):
        """Test that our simplified transformer has good performance."""
        from transformer import SimpleTransformer
        from config import TransformerConfig
        import time
        
        config = TransformerConfig(
            vocab_size=100, hidden_dim=64, num_heads=4, num_layers=2
        )
        model = SimpleTransformer(config)
        
        # Benchmark forward pass
        start_time = time.time()
        for _ in range(10):  # Run 10 iterations
            logits, _ = model.forward([[1, 2, 3, 4, 5]])
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time:.4f}s"
        
    def test_transformer_performance(self):
        """Test transformer performance metrics."""
        # Import and run a quick performance test
        from transformer import SimpleTransformer
        from config import TransformerConfig
        import time
        
        config = TransformerConfig(
            vocab_size=100,
            hidden_dim=64,
            num_heads=4,
            num_layers=2
        )
        
        model = SimpleTransformer(config)
        
        # Test forward pass performance
        start_time = time.time()
        logits, _ = model.forward([[1, 2, 3, 4, 5]])
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 1.0, "Forward pass too slow"
        assert logits is not None, "Forward pass failed"


@pytest.mark.integration  
class TestFullPipeline:
    """Integration tests for complete workflows."""
    
    def test_config_to_training_pipeline(self):
        """Test complete pipeline from config to training."""
        from transformer import SimpleTransformer
        from config import TransformerConfig
        
        # Create config
        config = TransformerConfig(
            vocab_size=50,
            hidden_dim=32,
            num_heads=2,
            num_layers=1
        )
        
        # Initialize model
        model = SimpleTransformer(config)
        
        # Test training step
        sample_input = [[1, 2, 3, 4, 5]]
        sample_targets = [[2, 3, 4, 5, 6]]
        
        loss = model.train_step(sample_input, sample_targets)
        
        assert loss > 0, "Training loss should be positive"
        assert loss < 10, "Training loss should be reasonable"
        
    def test_generation_pipeline(self):
        """Test complete text generation pipeline."""
        from transformer import SimpleTransformer
        from config import TransformerConfig
        
        config = TransformerConfig(
            vocab_size=50,
            hidden_dim=32,
            num_heads=2,
            num_layers=1
        )
        
        model = SimpleTransformer(config)
        
        # Test generation
        prompt = [[1, 2, 3]]  # Batch format
        generated = model.generate(prompt, max_new_tokens=5)
        
        assert len(generated[0]) == len(prompt[0]) + 5, "Generation length incorrect"
        assert all(0 <= token < config.vocab_size for token in generated[0]), "Invalid tokens generated"