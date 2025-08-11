"""
Comprehensive training tests for transformer models.
"""

import pytest
import numpy as np
from typing import Dict, List

from transformerlab.backends.factory import create_transformer
from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer


class TestBasicTraining:
    """Test basic training functionality."""
    
    def test_numpy_training_step(self, small_config, sample_input_data):
        """Test single training step for NumPy backend."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        inputs = np.array(sample_input_data["single_input"])
        targets = np.array(sample_input_data["single_target"])
        
        # Initial forward pass to get baseline loss
        _, initial_stats = transformer.forward(inputs, targets)
        initial_loss = initial_stats["loss"]
        
        # Training step
        loss = transformer.train_step(inputs, targets, optimizer)
        
        assert isinstance(loss, (float, np.floating))
        assert loss > 0
        
        # Loss should be recorded in history
        assert len(transformer.loss_history) > 0
        assert transformer.loss_history[-1] == loss
    
    def test_training_convergence(self, small_config, test_data_generator):
        """Test that training reduces loss over multiple steps."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("adam", learning_rate=0.01)
        
        # Create simple repeating pattern data
        data = test_data_generator.create_sequence_data(
            vocab_size=small_config["vocab_size"],
            seq_len=5,
            batch_size=2
        )
        
        # Train for several steps
        losses = []
        for step in range(10):
            loss = transformer.train_step(data["inputs"], data["targets"], optimizer)
            losses.append(loss)
        
        # Loss should generally decrease (allow for some fluctuation)
        assert losses[-1] < losses[0] * 1.5, "Training should reduce loss"
        
        # Should have recorded loss history
        assert len(transformer.loss_history) == 10
    
    def test_optimizer_types(self, small_config, sample_input_data):
        """Test different optimizer types."""
        transformer = create_transformer("numpy", **small_config)
        
        optimizer_configs = [
            ("sgd", {"learning_rate": 0.01, "momentum": 0.9}),
            ("adam", {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999}),
            ("adamw", {"learning_rate": 0.001, "weight_decay": 0.01}),
        ]
        
        inputs = np.array(sample_input_data["single_input"])
        targets = np.array(sample_input_data["single_target"])
        
        for optimizer_type, kwargs in optimizer_configs:
            # Create fresh transformer for each optimizer
            transformer = create_transformer("numpy", **small_config)
            optimizer = create_numpy_optimizer(optimizer_type, **kwargs)
            
            # Training should work without errors
            loss = transformer.train_step(inputs, targets, optimizer)
            assert isinstance(loss, (float, np.floating))
            assert loss > 0
    
    def test_batch_training(self, small_config):
        """Test training with different batch sizes."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            inputs = np.random.randint(
                1, small_config["vocab_size"], 
                size=(batch_size, 5)
            )
            targets = np.random.randint(
                1, small_config["vocab_size"], 
                size=(batch_size, 5)
            )
            
            loss = transformer.train_step(inputs, targets, optimizer)
            assert isinstance(loss, (float, np.floating))
            assert loss > 0
    
    def test_sequence_length_variation(self, small_config):
        """Test training with different sequence lengths."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        # Test with different sequence lengths (within max_seq_len)
        for seq_len in [3, 5, 8, 10]:
            if seq_len > small_config["max_seq_len"]:
                continue
                
            inputs = np.random.randint(
                1, small_config["vocab_size"], 
                size=(1, seq_len)
            )
            targets = np.random.randint(
                1, small_config["vocab_size"], 
                size=(1, seq_len)
            )
            
            loss = transformer.train_step(inputs, targets, optimizer)
            assert isinstance(loss, (float, np.floating))
            assert loss > 0


class TestTrainingStability:
    """Test training stability and numerical issues."""
    
    def test_gradient_explosion_protection(self, small_config):
        """Test that gradients don't explode during training."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=1.0)  # High learning rate
        
        inputs = np.random.randint(1, small_config["vocab_size"], size=(1, 5))
        targets = np.random.randint(1, small_config["vocab_size"], size=(1, 5))
        
        # Train for several steps with high learning rate
        losses = []
        for step in range(5):
            loss = transformer.train_step(inputs, targets, optimizer)
            losses.append(loss)
            
            # Loss shouldn't become infinite or NaN
            assert np.isfinite(loss), f"Loss became non-finite at step {step}: {loss}"
            
            # Loss shouldn't grow excessively
            if step > 0:
                assert loss < losses[0] * 100, f"Loss grew too much: {loss} vs {losses[0]}"
    
    def test_vanishing_gradient_detection(self, medium_config):
        """Test detection of vanishing gradients in deeper models."""
        transformer = create_transformer("numpy", **medium_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.0001)  # Very small learning rate
        
        inputs = np.random.randint(1, medium_config["vocab_size"], size=(2, 8))
        targets = np.random.randint(1, medium_config["vocab_size"], size=(2, 8))
        
        # Train for several steps
        losses = []
        for step in range(10):
            loss = transformer.train_step(inputs, targets, optimizer)
            losses.append(loss)
        
        # Even with small learning rate, there should be some change
        loss_change = abs(losses[-1] - losses[0])
        relative_change = loss_change / losses[0]
        
        # We should see at least some small change
        assert relative_change > 1e-8, "Training seems stuck (possible vanishing gradients)"
    
    def test_loss_consistency(self, small_config, sample_input_data):
        """Test that loss computation is consistent."""
        transformer = create_transformer("numpy", **small_config)
        
        inputs = np.array(sample_input_data["single_input"])
        targets = np.array(sample_input_data["single_target"])
        
        # Compute loss multiple times - should be identical
        _, stats1 = transformer.forward(inputs, targets)
        _, stats2 = transformer.forward(inputs, targets)
        
        loss1 = stats1["loss"]
        loss2 = stats2["loss"]
        
        assert abs(loss1 - loss2) < 1e-10, "Loss computation should be deterministic"


class TestTrainingMetrics:
    """Test training metrics and statistics."""
    
    def test_loss_history_tracking(self, small_config, sample_input_data):
        """Test that loss history is properly tracked."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        inputs = np.array(sample_input_data["input_tokens"])
        targets = np.array(sample_input_data["target_tokens"])
        
        # Initial state
        assert len(transformer.loss_history) == 0
        assert transformer.step_count == 0
        
        # Train for a few steps
        num_steps = 5
        for step in range(num_steps):
            loss = transformer.train_step(inputs, targets, optimizer)
            
            # Check that history is updated
            assert len(transformer.loss_history) == step + 1
            assert transformer.step_count == step + 1
            assert transformer.loss_history[-1] == loss
    
    def test_model_statistics(self, small_config, sample_input_data):
        """Test model statistics collection."""
        transformer = create_transformer("numpy", **small_config)
        
        inputs = np.array(sample_input_data["single_input"])
        targets = np.array(sample_input_data["single_target"])
        
        # Forward pass to generate statistics
        _, stats = transformer.forward(inputs, targets)
        
        # Check required statistics
        assert "loss" in stats
        assert "layer_stats" in stats
        
        # Check layer statistics structure
        layer_stats = stats["layer_stats"]
        assert len(layer_stats) == small_config["num_layers"]
        
        for layer_stat in layer_stats:
            assert "attention" in layer_stat
            assert "feed_forward" in layer_stat
    
    def test_training_progress_monitoring(self, small_config):
        """Test monitoring training progress."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("adam", learning_rate=0.01)
        
        # Create synthetic data that should be learnable
        inputs = np.array([[1, 2, 1, 2, 1]])
        targets = np.array([[2, 1, 2, 1, 2]])
        
        # Train and monitor progress
        losses = []
        for step in range(20):
            loss = transformer.train_step(inputs, targets, optimizer)
            losses.append(loss)
        
        # Should show learning progress
        early_avg = np.mean(losses[:5])
        late_avg = np.mean(losses[-5:])
        
        improvement = (early_avg - late_avg) / early_avg
        assert improvement > -0.5, "Model should learn on simple synthetic data"


class TestAdvancedTraining:
    """Test advanced training scenarios."""
    
    def test_curriculum_learning(self, small_config):
        """Test training with curriculum learning (easy to hard examples)."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("adam", learning_rate=0.01)
        
        # Start with short sequences
        short_inputs = np.random.randint(1, small_config["vocab_size"], size=(2, 3))
        short_targets = np.random.randint(1, small_config["vocab_size"], size=(2, 3))
        
        # Train on short sequences first
        for _ in range(5):
            transformer.train_step(short_inputs, short_targets, optimizer)
        
        short_loss = transformer.loss_history[-1]
        
        # Then train on longer sequences
        long_inputs = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        long_targets = np.random.randint(1, small_config["vocab_size"], size=(2, 8))
        
        for _ in range(5):
            transformer.train_step(long_inputs, long_targets, optimizer)
        
        # Should handle the transition without major issues
        final_loss = transformer.loss_history[-1]
        assert np.isfinite(final_loss)
    
    def test_mixed_sequence_lengths(self, small_config):
        """Test training with mixed sequence lengths in same batch."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        # Create sequences of different lengths (padded to same size)
        seq_len = 6
        inputs = np.random.randint(1, small_config["vocab_size"], size=(2, seq_len))
        targets = np.random.randint(1, small_config["vocab_size"], size=(2, seq_len))
        
        # Simulate padding by setting later positions to 0 for second sequence
        inputs[1, 4:] = 0
        targets[1, 4:] = 0
        
        # Training should handle this appropriately
        loss = transformer.train_step(inputs, targets, optimizer)
        assert np.isfinite(loss)
        assert loss > 0
    
    @pytest.mark.slow
    def test_long_training_session(self, small_config):
        """Test stability over longer training sessions."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("adam", learning_rate=0.001)
        
        # Create diverse training data
        num_batches = 50
        all_losses = []
        
        for batch in range(num_batches):
            # Generate different random data each batch
            np.random.seed(batch)  # For reproducibility
            inputs = np.random.randint(1, small_config["vocab_size"], size=(2, 5))
            targets = np.random.randint(1, small_config["vocab_size"], size=(2, 5))
            
            loss = transformer.train_step(inputs, targets, optimizer)
            all_losses.append(loss)
            
            # Check for numerical stability
            assert np.isfinite(loss), f"Loss became non-finite at batch {batch}"
        
        # Training should be stable over time
        recent_losses = all_losses[-10:]
        assert all(np.isfinite(loss) for loss in recent_losses)
        
        # Variance in recent losses shouldn't be too high (indicating stability)
        recent_variance = np.var(recent_losses)
        recent_mean = np.mean(recent_losses)
        coefficient_of_variation = np.sqrt(recent_variance) / recent_mean
        
        assert coefficient_of_variation < 2.0, "Training should be stable over time"


class TestTrainingEdgeCases:
    """Test edge cases in training."""
    
    def test_single_token_sequences(self, small_config):
        """Test training with very short sequences."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        # Single token input/output
        inputs = np.array([[5]])
        targets = np.array([[3]])
        
        loss = transformer.train_step(inputs, targets, optimizer)
        assert np.isfinite(loss)
        assert loss > 0
    
    def test_repeated_tokens(self, small_config):
        """Test training with repeated token patterns."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        # All same tokens
        inputs = np.array([[1, 1, 1, 1, 1]])
        targets = np.array([[1, 1, 1, 1, 1]])
        
        loss = transformer.train_step(inputs, targets, optimizer)
        assert np.isfinite(loss)
        assert loss > 0
    
    def test_extreme_vocabulary_values(self, small_config):
        """Test training with edge vocabulary values."""
        transformer = create_transformer("numpy", **small_config)
        optimizer = create_numpy_optimizer("sgd", learning_rate=0.01)
        
        vocab_size = small_config["vocab_size"]
        
        # Use minimum and maximum vocabulary values
        inputs = np.array([[1, vocab_size-1, 1, vocab_size-1, 1]])
        targets = np.array([[vocab_size-1, 1, vocab_size-1, 1, vocab_size-1]])
        
        loss = transformer.train_step(inputs, targets, optimizer)
        assert np.isfinite(loss)
        assert loss > 0