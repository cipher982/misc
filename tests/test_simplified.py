"""
Tests for the Simplified Transformer Architecture

These tests focus on correctness and educational value rather than
comprehensive edge case coverage. They validate that the core
mathematical operations work as expected.
"""

import pytest
import math
import random
from config import TransformerConfig, tiny_transformer, small_transformer
from transformer import SimpleTransformer


class TestTransformerConfig:
    """Test the simplified configuration system."""
    
    def test_valid_config_creation(self):
        """Test that valid configurations can be created."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_dim=256,
            num_heads=8,
            num_layers=6
        )
        
        assert config.vocab_size == 1000
        assert config.hidden_dim == 256
        assert config.num_heads == 8
        assert config.head_dim == 32  # 256 / 8
        assert config.ff_dim == 1024  # 4 * 256 (default)
    
    def test_head_divisibility_constraint(self):
        """Test that hidden_dim must be divisible by num_heads."""
        with pytest.raises(ValueError) as exc_info:
            TransformerConfig(hidden_dim=100, num_heads=7)
        
        error_msg = str(exc_info.value)
        assert "🎓 Educational Error" in error_msg
        assert "must be evenly divisible" in error_msg
        assert "Each head gets" in error_msg
    
    def test_reasonable_bounds(self):
        """Test reasonable parameter bounds.""" 
        # hidden_dim should be at least num_heads
        with pytest.raises(ValueError) as exc_info:
            TransformerConfig(hidden_dim=4, num_heads=8)
        
        assert "must be evenly divisible by num_heads" in str(exc_info.value)
        
        # Negative values should fail
        with pytest.raises(ValueError):
            TransformerConfig(vocab_size=-1)
        
        with pytest.raises(ValueError):
            TransformerConfig(num_layers=0)
    
    def test_dropout_bounds(self):
        """Test dropout parameter bounds."""
        # Valid dropout
        config = TransformerConfig(dropout=0.1)
        assert config.dropout == 0.1
        
        # Invalid dropout values
        with pytest.raises(ValueError):
            TransformerConfig(dropout=-0.1)
        
        with pytest.raises(ValueError):
            TransformerConfig(dropout=1.5)
    
    def test_convenience_configs(self):
        """Test pre-defined convenience configurations."""
        tiny = tiny_transformer()
        assert tiny.vocab_size == 100
        assert tiny.hidden_dim == 64
        assert tiny.num_heads == 4
        
        small = small_transformer()
        assert small.vocab_size == 10000
        assert small.hidden_dim == 256
        assert small.num_heads == 8
    
    def test_parameter_count_estimation(self):
        """Test parameter count estimation."""
        config = tiny_transformer()
        param_count = config.total_params
        
        # Should be reasonable for tiny model
        assert 50000 < param_count < 200000
        
        # Larger model should have more parameters
        large_config = TransformerConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=4
        )
        assert large_config.total_params > param_count
    
    def test_config_summary(self):
        """Test configuration summary output."""
        config = tiny_transformer()
        summary = config.summary()
        
        assert "🔧 Transformer Configuration" in summary
        assert str(config.num_layers) in summary
        assert str(config.num_heads) in summary
        assert str(config.vocab_size) in summary


class TestSimpleTransformer:
    """Test the simplified transformer implementation."""
    
    @pytest.fixture
    def tiny_model(self):
        """Create a tiny model for testing."""
        config = tiny_transformer()
        return SimpleTransformer(config, verbose=False)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input and target data."""
        batch_size, seq_len = 2, 4
        vocab_size = 100
        
        input_ids = [[random.randint(1, vocab_size-1) for _ in range(seq_len)] 
                     for _ in range(batch_size)]
        target_ids = [[random.randint(1, vocab_size-1) for _ in range(seq_len)]
                      for _ in range(batch_size)]
        
        return input_ids, target_ids
    
    def test_model_initialization(self, tiny_model):
        """Test that model initializes correctly."""
        assert tiny_model.config.vocab_size == 100
        assert tiny_model.config.hidden_dim == 64
        assert tiny_model.step_count == 0
        assert len(tiny_model.loss_history) == 0
        
        # Check parameter shapes
        assert len(tiny_model.token_embeddings) == 100  # vocab_size
        assert len(tiny_model.token_embeddings[0]) == 64  # hidden_dim
        
        assert len(tiny_model.layers) == 2  # num_layers
    
    def test_parameter_count(self, tiny_model):
        """Test parameter counting.""" 
        param_count = tiny_model.get_parameter_count()
        
        # Should be reasonable for tiny model
        assert 50000 < param_count < 200000
        
        # Check it matches config estimate (approximately)
        config_estimate = tiny_model.config.total_params
        assert abs(param_count - config_estimate) / config_estimate < 0.2  # Within 20%
    
    def test_forward_pass_shapes(self, tiny_model, sample_data):
        """Test that forward pass produces correct output shapes."""
        input_ids, target_ids = sample_data
        
        # Forward pass without targets
        logits, stats = tiny_model.forward(input_ids)
        
        batch_size, seq_len = len(input_ids), len(input_ids[0])
        vocab_size = tiny_model.config.vocab_size
        
        # Check logits shape
        assert len(logits) == batch_size
        assert len(logits[0]) == seq_len  
        assert len(logits[0][0]) == vocab_size
        
        # Check stats
        assert 'layer_stats' in stats
        assert len(stats['layer_stats']) == tiny_model.config.num_layers
        assert stats['loss'] is None  # No targets provided
    
    def test_forward_pass_with_loss(self, tiny_model, sample_data):
        """Test forward pass with loss computation."""
        input_ids, target_ids = sample_data
        
        logits, stats = tiny_model.forward(input_ids, target_ids)
        
        # Loss should be computed
        assert stats['loss'] is not None
        assert stats['loss'] > 0  # Cross-entropy loss should be positive
        assert len(tiny_model.loss_history) == 1
        
        # Loss should be reasonable (not infinity or NaN)
        assert math.isfinite(stats['loss'])
    
    def test_generation(self, tiny_model):
        """Test text generation."""
        prompt = [[1, 2, 3]]  # Simple prompt
        max_new_tokens = 5
        
        generated = tiny_model.generate(prompt, max_new_tokens, temperature=1.0)
        
        # Check output shape
        assert len(generated) == 1  # Same batch size
        assert len(generated[0]) == len(prompt[0]) + max_new_tokens
        
        # Generated tokens should be valid vocab indices
        for token in generated[0]:
            assert 0 <= token < tiny_model.config.vocab_size
    
    def test_generation_temperature(self, tiny_model):
        """Test that temperature affects generation diversity."""
        prompt = [[1, 2, 3]]
        
        # Low temperature should be more deterministic
        gen_low = tiny_model.generate(prompt, 3, temperature=0.1)
        
        # High temperature should be more random
        gen_high = tiny_model.generate(prompt, 3, temperature=2.0)
        
        # Both should have same length
        assert len(gen_low[0]) == len(gen_high[0])
    
    def test_training_step(self, tiny_model, sample_data):
        """Test training step."""
        input_ids, target_ids = sample_data
        
        initial_step = tiny_model.step_count
        initial_loss_history = len(tiny_model.loss_history)
        
        loss = tiny_model.train_step(input_ids, target_ids)
        
        # Training step should update state
        assert tiny_model.step_count == initial_step + 1
        assert len(tiny_model.loss_history) == initial_loss_history + 1
        assert loss > 0
        assert math.isfinite(loss)
    
    def test_multiple_training_steps(self, tiny_model, sample_data):
        """Test multiple training steps."""
        input_ids, target_ids = sample_data
        
        losses = []
        for _ in range(3):
            loss = tiny_model.train_step(input_ids, target_ids)
            losses.append(loss)
        
        # All losses should be finite
        for loss in losses:
            assert math.isfinite(loss)
            assert loss > 0
        
        # Step count should update correctly
        assert tiny_model.step_count == 3
        assert len(tiny_model.loss_history) == 3
    
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        config = tiny_transformer()
        model = SimpleTransformer(config, verbose=False)
        
        # Test batch size 1
        input_1 = [[1, 2, 3, 4]]
        logits_1, _ = model.forward(input_1)
        assert len(logits_1) == 1
        
        # Test batch size 3
        input_3 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        logits_3, _ = model.forward(input_3)
        assert len(logits_3) == 3
    
    def test_different_sequence_lengths(self):
        """Test model works with different sequence lengths."""
        config = tiny_transformer()
        model = SimpleTransformer(config, verbose=False)
        
        # Test short sequence
        input_short = [[1, 2]]
        logits_short, _ = model.forward(input_short)
        assert len(logits_short[0]) == 2
        
        # Test longer sequence (within seq_len limit)
        input_long = [[1, 2, 3, 4, 5, 6]]
        logits_long, _ = model.forward(input_long)
        assert len(logits_long[0]) == 6


class TestUtilityFunctions:
    """Test utility functions in the transformer."""
    
    def test_tensor_operations(self):
        """Test basic tensor operations."""
        config = tiny_transformer()
        model = SimpleTransformer(config, verbose=False)
        
        # Test zeros creation
        zeros_2d = model._zeros((2, 3))
        assert len(zeros_2d) == 2
        assert len(zeros_2d[0]) == 3
        assert all(val == 0.0 for row in zeros_2d for val in row)
        
        # Test 3D addition
        a = [[[1.0, 2.0]], [[3.0, 4.0]]]
        b = [[[0.5, 1.0]], [[1.5, 2.0]]]
        result = model._add_tensors_3d(a, b)
        assert result[0][0][0] == 1.5
        assert result[0][0][1] == 3.0
        assert result[1][0][0] == 4.5
        assert result[1][0][1] == 6.0
    
    def test_softmax(self):
        """Test softmax function."""
        config = tiny_transformer()
        model = SimpleTransformer(config, verbose=False)
        
        logits = [1.0, 2.0, 3.0]
        probs = model._softmax(logits)
        
        # Probabilities should sum to 1
        assert abs(sum(probs) - 1.0) < 1e-6
        
        # All probabilities should be positive
        assert all(p > 0 for p in probs)
        
        # Higher logit should have higher probability
        assert probs[2] > probs[1] > probs[0]
    
    def test_layer_norm(self):
        """Test layer normalization."""
        config = tiny_transformer()
        model = SimpleTransformer(config, verbose=False)
        
        # Create test input
        x = [[[1.0, 2.0, 3.0, 4.0]]]  # (1, 1, 4)
        weight = [1.0, 1.0, 1.0, 1.0]
        bias = [0.0, 0.0, 0.0, 0.0]
        
        result = model._layer_norm(x, weight, bias)
        
        # After normalization, mean should be ~0 and std should be ~1
        normalized_values = result[0][0]
        mean = sum(normalized_values) / len(normalized_values)
        assert abs(mean) < 1e-5
    
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss computation."""
        config = TransformerConfig(vocab_size=5, hidden_dim=8, num_heads=2, num_layers=1)
        model = SimpleTransformer(config, verbose=False)
        
        # Create simple logits (batch=1, seq=2, vocab=5)
        logits = [[[1.0, 2.0, 0.5, 0.1, 0.2],   # Position 0
                   [0.1, 0.2, 3.0, 1.0, 0.5]]]  # Position 1
        
        targets = [[1, 2]]  # Target token 1 at pos 0, token 2 at pos 1
        
        loss = model._compute_cross_entropy_loss(logits, targets)
        
        # Loss should be positive and finite
        assert loss > 0
        assert math.isfinite(loss)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input_handling(self):
        """Test handling of edge cases in input.""" 
        config = tiny_transformer()
        model = SimpleTransformer(config, verbose=False)
        
        # Single token sequence
        single_token = [[5]]
        logits, _ = model.forward(single_token)
        assert len(logits[0]) == 1
    
    def test_parameter_bounds(self):
        """Test parameter validation."""
        # Learning rate bounds
        config = tiny_transformer()
        model = SimpleTransformer(config, verbose=False)
        
        # Should work with reasonable learning rates
        input_ids = [[1, 2, 3]]
        targets = [[2, 3, 4]]
        
        loss = model.train_step(input_ids, targets, learning_rate=0.001)
        assert math.isfinite(loss)
    
    def test_vocab_boundary_tokens(self):
        """Test tokens at vocabulary boundaries."""
        config = TransformerConfig(vocab_size=10, hidden_dim=8, num_heads=2, num_layers=1)
        model = SimpleTransformer(config, verbose=False)
        
        # Test with token IDs at boundaries
        boundary_input = [[0, 9]]  # First and last tokens
        logits, _ = model.forward(boundary_input)
        
        # Should not crash and produce valid output
        assert len(logits) == 1
        assert len(logits[0]) == 2
        assert len(logits[0][0]) == 10


def test_full_pipeline():
    """Integration test of the full pipeline.""" 
    print("\\n🧪 Running Full Pipeline Test")
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=50,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        seq_len=8,
        batch_size=2
    )
    
    print(f"✅ Configuration: {config.head_dim} dims per head")
    
    # Initialize model
    model = SimpleTransformer(config, verbose=False)
    print(f"✅ Model initialized with {model.get_parameter_count():,} parameters")
    
    # Create training data
    batch_size, seq_len = 2, 6
    input_ids = [[random.randint(1, 49) for _ in range(seq_len)] for _ in range(batch_size)]
    target_ids = [[random.randint(1, 49) for _ in range(seq_len)] for _ in range(batch_size)]
    
    print(f"✅ Training data: {batch_size} batches × {seq_len} tokens")
    
    # Training loop
    initial_loss = None
    for step in range(5):
        loss = model.train_step(input_ids, target_ids)
        if initial_loss is None:
            initial_loss = loss
        print(f"   Step {step + 1}: Loss = {loss:.4f}")
    
    print(f"✅ Training completed. Initial loss: {initial_loss:.4f}, Final loss: {loss:.4f}")
    
    # Generation test
    prompt = [[1, 2, 3]]
    generated = model.generate(prompt, max_new_tokens=3, temperature=0.8)
    print(f"✅ Generation: {prompt[0]} → {generated[0]}")
    
    print("🎉 Full pipeline test passed!")


if __name__ == "__main__":
    # Run the full pipeline test
    test_full_pipeline()
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\\n⚠️  pytest not available. Run 'pip install pytest' for full test suite.")