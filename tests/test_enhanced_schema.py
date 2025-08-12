"""
Comprehensive test suite for the enhanced schema system.

Tests cover:
- Configuration validation and error handling
- Mathematical correctness of estimators
- Auto-repair functionality
- Cross-field constraint validation
- Modern transformer architecture support
"""

import pytest
from transformerlab.schema.model_config import ModelConfig, ValidationError
from transformerlab.schema.auto_repair import (
    snap_config, create_repair_plans, get_valid_heads,
    config_to_url_params, config_from_url_params
)


class TestModelConfigValidation:
    """Test configuration validation and constraint checking."""
    
    def test_valid_basic_config(self):
        """Basic valid configuration should parse successfully."""
        config = ModelConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            seq_len=1024,
            vocab_size=32000
        )
        
        assert config.head_dim == 32
        assert config.effective_kv_heads == 8
        assert config.ff_dim == 1024  # 4.0 * 256, aligned to 64
        
    def test_gqa_config(self):
        """Grouped Query Attention configuration."""
        config = ModelConfig(
            hidden_dim=512,
            num_heads=16,
            num_kv_heads=4,  # 4:1 GQA ratio
            num_layers=8,
            seq_len=2048,
            vocab_size=32000
        )
        
        assert config.head_dim == 32
        assert config.effective_kv_heads == 4
        
    def test_head_divisibility_error(self):
        """Should reject configs where hidden_dim not divisible by num_heads."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                hidden_dim=100,  # Not divisible by 8
                num_heads=8,
                num_layers=6,
                seq_len=1024,
                vocab_size=32000
            )
        
        error_msg = str(exc_info.value)
        assert "divisible by num_heads" in error_msg
        assert "Valid options:" in error_msg
        
    def test_gqa_divisibility_error(self):
        """Should reject invalid GQA configurations."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                hidden_dim=256,
                num_heads=8,
                num_kv_heads=3,  # 8 not divisible by 3
                num_layers=6,
                seq_len=1024,
                vocab_size=32000
            )
        
        error_msg = str(exc_info.value)
        assert "divisible by num_kv_heads" in error_msg
        
    def test_flash_attention_constraint(self):
        """FlashAttention requires head_dim multiple of 16."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                hidden_dim=200,  # head_dim = 25, not multiple of 16
                num_heads=8,
                num_layers=6,
                attention_impl="flash",
                seq_len=1024,
                vocab_size=32000
            )
        
        error_msg = str(exc_info.value)
        assert "FlashAttention requires head_dim" in error_msg
        assert "multiple of 16" in error_msg
        
    def test_dtype_device_compatibility(self):
        """Should warn about float16 on CPU."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                hidden_dim=256,
                num_heads=8,
                num_layers=6,
                device="cpu",
                dtype="float16",
                seq_len=1024,
                vocab_size=32000
            )
        
        error_msg = str(exc_info.value)
        assert "float16 not well supported on CPU" in error_msg
        
    def test_rope_even_dimension_constraint(self):
        """RoPE requires even dimensions."""
        # This should work (rope_fraction * head_dim = even)
        config = ModelConfig(
            hidden_dim=256,
            num_heads=8,  # head_dim = 32
            num_layers=6,
            pos_encoding_type="RoPE",
            rope_fraction=0.5,  # 0.5 * 32 = 16 (even)
            seq_len=1024,
            vocab_size=32000
        )
        
        # This should fail (rope_fraction * head_dim = odd)
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                hidden_dim=192,
                num_heads=8,  # head_dim = 24
                num_layers=6,
                pos_encoding_type="RoPE", 
                rope_fraction=0.5,  # 0.5 * 24 = 12 (even, should work)
                seq_len=1024,
                vocab_size=32000
            )


class TestMathematicalCorrectness:
    """Test parameter/memory/FLOPs estimations for correctness."""
    
    def test_parameter_estimation_basic(self):
        """Test parameter counting on known configuration."""
        config = ModelConfig(
            hidden_dim=256,
            num_heads=8,
            ff_mult=4.0,
            num_layers=6,
            vocab_size=32000,
            tie_weights=True
        )
        
        # Manual calculation for verification
        d = 256
        vocab = 32064  # aligned vocab size
        ff = 1024  # 4.0 * 256, aligned to 64
        
        # Expected params:
        # - Embeddings: 32064 * 256 (tied weights, so no output projection)
        # - Per layer: 4*d² (attention) + 3*d*ff (SwiGLU) + 4*d (norms)
        embed_params = vocab * d
        layer_params = 4 * d * d + 3 * d * ff + 4 * d
        expected = embed_params + 6 * layer_params
        
        actual = config.estimate_parameters()
        
        # Should be very close (within 1% due to alignment)
        assert abs(actual - expected) / expected < 0.01
        
    def test_gqa_parameter_reduction(self):
        """Test that GQA reduces parameters correctly."""
        # Standard MHA
        config_mha = ModelConfig(
            hidden_dim=512,
            num_heads=16,
            num_layers=6,
            seq_len=1024,
            vocab_size=32000
        )
        
        # 4:1 GQA
        config_gqa = ModelConfig(
            hidden_dim=512,
            num_heads=16,
            num_kv_heads=4,
            num_layers=6,
            seq_len=1024,
            vocab_size=32000
        )
        
        params_mha = config_mha.estimate_parameters()
        params_gqa = config_gqa.estimate_parameters()
        
        # GQA should have fewer parameters due to smaller K,V projections
        assert params_gqa < params_mha
        
        # Calculate expected reduction
        d = 512
        # MHA: Q(d²) + K(d²) + V(d²) + O(d²) = 4d²
        # GQA: Q(d²) + K(d*d/4) + V(d*d/4) + O(d²) = 3.5d²
        reduction_per_layer = 6 * 0.5 * d * d  # 6 layers * 0.5d² reduction
        expected_diff = reduction_per_layer
        actual_diff = params_mha - params_gqa
        
        # Should be close to expected reduction
        assert abs(actual_diff - expected_diff) / expected_diff < 0.1
        
    def test_memory_scaling(self):
        """Test memory estimation scales correctly with batch size."""
        config = ModelConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            seq_len=1024,
            vocab_size=32000
        )
        
        mem_1 = config.estimate_memory_mb(batch_size=1)
        mem_2 = config.estimate_memory_mb(batch_size=2)
        
        # Parameters should be same
        assert mem_1["parameters_mb"] == mem_2["parameters_mb"]
        
        # KV cache should double
        assert mem_2["kv_cache_mb"] == 2 * mem_1["kv_cache_mb"]
        
        # Total should increase by KV cache amount
        expected_diff = mem_1["kv_cache_mb"]
        actual_diff = mem_2["total_mb"] - mem_1["total_mb"]
        assert abs(actual_diff - expected_diff) < 0.1
        
    def test_flops_prefill_vs_decode(self):
        """Test FLOPs difference between prefill and decode."""
        config = ModelConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            seq_len=2048,
            vocab_size=32000
        )
        
        prefill_flops = config.estimate_flops_per_token("prefill")
        decode_flops = config.estimate_flops_per_token("decode")
        
        # Prefill should have much higher attention FLOPs due to O(L²) scaling
        prefill_total = prefill_flops["total_flops_per_token_prefill"]
        decode_total = decode_flops["total_flops_per_token_decode"]
        
        assert prefill_total > decode_total
        
        # FFN FLOPs should be the same (doesn't depend on sequence length)
        assert prefill_flops["ffn_flops_prefill"] == decode_flops["ffn_flops_decode"]


class TestAutoRepair:
    """Test the auto-repair functionality."""
    
    def test_get_valid_heads(self):
        """Test valid head selection with hardware constraints."""
        # Test basic case
        valid_heads = get_valid_heads(256, "bfloat16", "eager")
        
        # Should include canonical options
        assert 8 in valid_heads  # 256/8 = 32 (canonical)
        assert 4 in valid_heads  # 256/4 = 64 (canonical)
        
        # Should prefer canonical head dims first
        head_32_index = valid_heads.index(8)  # 256/8 = 32
        head_64_index = valid_heads.index(4)  # 256/4 = 64
        
        # Both should be early in the list
        assert head_32_index < len(valid_heads) // 2
        assert head_64_index < len(valid_heads) // 2
        
    def test_snap_config_valid_unchanged(self):
        """Auto-repair should not change already valid configs."""
        config = {
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "seq_len": 1024,
            "vocab_size": 32000
        }
        
        fixed, changes = snap_config(config)
        
        assert changes == []  # No changes needed
        assert fixed == config  # Config unchanged
        
    def test_snap_config_fixes_heads(self):
        """Auto-repair should fix head divisibility issues."""
        config = {
            "hidden_dim": 100,  # Not divisible by 8
            "num_heads": 8,
            "num_layers": 6,
            "seq_len": 1024,
            "vocab_size": 32000
        }
        
        fixed, changes = snap_config(config)
        
        assert len(changes) > 0
        assert "num_heads" in changes[0]
        
        # Fixed config should be valid
        ModelConfig(**fixed)
        
    def test_repair_plans_multiple_options(self):
        """Should generate multiple repair strategies."""
        config = {
            "hidden_dim": 100,
            "num_heads": 8,
            "num_layers": 6,
            "seq_len": 1024,
            "vocab_size": 32000
        }
        
        plans = create_repair_plans(config)
        
        assert len(plans) > 0
        
        # Each plan should have required fields
        for plan in plans:
            assert "name" in plan
            assert "config" in plan
            assert "diff" in plan
            assert "rationale" in plan
            
            # Plan config should be valid
            ModelConfig(**plan["config"])
            
    def test_locked_fields_respected(self):
        """Locked fields should not be modified during repair."""
        config = {
            "hidden_dim": 100,
            "num_heads": 8,
            "num_layers": 6,
            "seq_len": 1024,
            "vocab_size": 32000
        }
        
        # Lock hidden_dim, should adjust num_heads instead
        fixed, changes = snap_config(config, locked_fields=["hidden_dim"])
        
        assert fixed["hidden_dim"] == 100  # Should remain unchanged
        assert fixed["num_heads"] != 8      # Should be adjusted
        
    def test_url_serialization_roundtrip(self):
        """URL serialization should work correctly."""
        config = {
            "hidden_dim": 256,
            "num_heads": 8,
            "ff_mult": 4.0,
            "num_layers": 6,
            "tie_weights": True,
            "dtype": "bfloat16"
        }
        
        # Serialize to URL params
        url_params = config_to_url_params(config)
        
        # Deserialize back
        recovered = config_from_url_params(url_params)
        
        # Should match original (with proper types)
        assert recovered["hidden_dim"] == 256
        assert recovered["num_heads"] == 8
        assert recovered["ff_mult"] == 4.0
        assert recovered["tie_weights"] is True
        assert recovered["dtype"] == "bfloat16"


class TestWarnings:
    """Test the warning system for performance hints."""
    
    def test_alignment_warnings(self):
        """Should warn about suboptimal alignments."""
        config = ModelConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            seq_len=1024,
            vocab_size=30000  # Not aligned to 64
        )
        
        warnings = config.get_warnings()
        
        # Should warn about vocab alignment
        assert any("vocab_size" in w and "aligned" in w for w in warnings)
        
    def test_performance_warnings(self):
        """Should warn about performance issues."""
        config = ModelConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            seq_len=4096,      # Long sequence
            attention_impl="eager",  # Not flash
            vocab_size=32000
        )
        
        warnings = config.get_warnings()
        
        # Should warn about eager attention with long sequences
        assert any("eager attention" in w.lower() and "flash" in w.lower() for w in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])