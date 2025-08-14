"""
Intelligent auto-repair system for transformer configurations.

This module provides:
- Smart constraint fixing with minimal changes
- Multiple repair strategies with rationales
- Preference for canonical architectures (head_dim 32/64/128)
- Hardware-aware optimizations
"""

import urllib.parse
from typing import Any

from .model_config import ModelConfig, ValidationError


def get_valid_heads(
    hidden_dim: int, dtype: str = "bfloat16", attention_impl: str = "eager"
) -> list[int]:
    """
    Get valid head counts with preference for canonical head dimensions.

    Returns heads ordered by:
    1. Canonical head dimensions {32, 64, 128} first
    2. Hardware alignment requirements
    3. Mathematical divisibility
    """
    # Start with mathematical requirement: hidden_dim must be divisible by heads
    candidates = [h for h in range(1, 33) if hidden_dim % h == 0]

    # Apply hardware alignment filters
    if dtype in ["float16", "bfloat16"]:
        # Modern GPUs prefer head_dim multiple of 8 for tensor cores
        candidates = [h for h in candidates if (hidden_dim // h) % 8 == 0]

    if attention_impl == "flash":
        # FlashAttention requires head_dim multiple of 16
        candidates = [h for h in candidates if (hidden_dim // h) % 16 == 0]

    # Separate canonical from non-canonical head dimensions
    canonical_heads = []
    other_heads = []

    for h in candidates:
        head_dim = hidden_dim // h
        if head_dim in {32, 64, 128}:
            canonical_heads.append(h)
        else:
            other_heads.append(h)

    # Return canonical architectures first (proven to work well)
    return canonical_heads + other_heads


def snap_config(
    config: dict[str, Any], locked_fields: list[str] = None
) -> tuple[dict[str, Any], list[str]]:
    """
    Auto-repair configuration with minimal changes and diff tracking.

    Args:
        config: Configuration dictionary to repair
        locked_fields: Fields that should not be modified during repair

    Returns:
        Tuple of (fixed_config, list_of_changes_made)
    """
    locked_fields = locked_fields or []
    fixed_config = config.copy()
    changes = []

    # Check if already valid
    try:
        ModelConfig(**fixed_config)
        return fixed_config, []  # Already valid, no changes needed
    except ValidationError as e:
        error_msg = str(e)

        # Fix head divisibility (most common issue)
        if "divisible by num_heads" in error_msg and "num_heads" not in locked_fields:
            old_heads = fixed_config["num_heads"]
            valid_heads = get_valid_heads(
                fixed_config["hidden_dim"],
                fixed_config.get("dtype", "bfloat16"),
                fixed_config.get("attention_impl", "eager"),
            )
            if valid_heads:
                # Choose closest valid head count
                new_heads = min(valid_heads, key=lambda x: abs(x - old_heads))
                fixed_config["num_heads"] = new_heads

                old_head_dim = fixed_config["hidden_dim"] // old_heads
                new_head_dim = fixed_config["hidden_dim"] // new_heads
                changes.append(
                    f"num_heads {old_heads} → {new_heads} "
                    f"(head_dim {old_head_dim} → {new_head_dim})"
                )

        # Fix GQA/MQA constraints if present
        if fixed_config.get("num_kv_heads") and "num_kv_heads" not in locked_fields:
            heads = fixed_config["num_heads"]
            kv_heads = fixed_config["num_kv_heads"]

            if heads % kv_heads != 0:
                # Find valid KV head counts
                valid_kv = [k for k in range(1, heads + 1) if heads % k == 0]
                new_kv = min(valid_kv, key=lambda x: abs(x - kv_heads))
                fixed_config["num_kv_heads"] = new_kv
                changes.append(f"num_kv_heads {kv_heads} → {new_kv}")

        # Adjust ff_mult to reasonable bounds
        if "ff_mult" not in locked_fields:
            ff_mult = fixed_config.get("ff_mult", 4.0)
            if ff_mult < 1.0:
                fixed_config["ff_mult"] = 2.0
                changes.append(f"ff_mult {ff_mult:.1f} → 2.0")
            elif ff_mult > 8.0:
                fixed_config["ff_mult"] = 4.0
                changes.append(f"ff_mult {ff_mult:.1f} → 4.0")

        # Fix device/dtype incompatibilities
        if (
            fixed_config.get("device") == "cpu"
            and fixed_config.get("dtype") == "float16"
        ):
            if "dtype" not in locked_fields:
                fixed_config["dtype"] = "bfloat16"
                changes.append("dtype float16 → bfloat16 (CPU compatibility)")

        return fixed_config, changes


def create_repair_plans(
    config: dict[str, Any], locked_fields: list[str] = None
) -> list[dict[str, Any]]:
    """
    Generate multiple repair options with different strategies.

    This gives users choice in how to resolve conflicts, with clear rationales
    for each approach.
    """
    plans = []
    locked_fields = locked_fields or []

    # Check if repair is needed
    try:
        ModelConfig(**config)
        return []  # Already valid
    except ValidationError as e:
        error_msg = str(e)

        # Strategy A: Adjust num_heads (keeps model size similar)
        if "divisible by num_heads" in error_msg and "num_heads" not in locked_fields:
            plan_a = config.copy()
            valid_heads = get_valid_heads(config["hidden_dim"])
            if valid_heads:
                old_heads = config["num_heads"]
                new_heads = min(valid_heads, key=lambda x: abs(x - old_heads))
                plan_a["num_heads"] = new_heads

                new_head_dim = config["hidden_dim"] // new_heads
                plans.append(
                    {
                        "name": "Adjust Heads",
                        "config": plan_a,
                        "diff": f"num_heads {old_heads} → {new_heads}",
                        "rationale": f"Keeps hidden_dim={config['hidden_dim']}, head_dim becomes {new_head_dim}",
                    }
                )

        # Strategy B: Adjust hidden_dim (keeps head count)
        if "divisible by num_heads" in error_msg and "hidden_dim" not in locked_fields:
            plan_b = config.copy()
            heads = config["num_heads"]
            old_dim = config["hidden_dim"]

            # Find nearest valid hidden_dim (prefer multiples of 32 for efficiency)
            candidates = []
            for d in range(max(64, old_dim - 128), old_dim + 129, 32):
                if d % heads == 0 and (d // heads) % 8 == 0:
                    candidates.append(d)

            if candidates:
                new_dim = min(candidates, key=lambda x: abs(x - old_dim))
                plan_b["hidden_dim"] = new_dim
                plans.append(
                    {
                        "name": "Adjust Hidden Dim",
                        "config": plan_b,
                        "diff": f"hidden_dim {old_dim} → {new_dim}",
                        "rationale": f"Keeps num_heads={heads}, head_dim becomes {new_dim // heads}",
                    }
                )

        # Strategy C: Enable GQA for efficiency (if not already using it)
        if not config.get("num_kv_heads") and "num_kv_heads" not in locked_fields:
            plan_c = config.copy()
            heads = config.get("num_heads", 8)

            # Suggest 4:1 or 2:1 GQA ratios (common in modern models)
            for ratio in [4, 2]:
                if heads % ratio == 0:
                    kv_heads = heads // ratio
                    plan_c["num_kv_heads"] = kv_heads
                    plans.append(
                        {
                            "name": f"Enable {ratio}:1 GQA",
                            "config": plan_c,
                            "diff": f"Add num_kv_heads={kv_heads}",
                            "rationale": f"Reduces KV cache by {ratio}x, speeds up inference",
                        }
                    )
                    break

    return plans


def config_to_url_params(config: dict[str, Any]) -> str:
    """
    Serialize configuration to URL query parameters for sharing.

    This enables easy sharing of configurations via URLs.
    """
    # Remove None values and convert to strings
    simplified = {k: v for k, v in config.items() if v is not None}
    return urllib.parse.urlencode(simplified)


def config_from_url_params(params: str) -> dict[str, Any]:
    """
    Deserialize configuration from URL query parameters.

    Handles proper type conversion for different field types.
    """
    parsed = urllib.parse.parse_qs(params)
    config = {}

    # Define field types for proper conversion
    int_fields = {
        "hidden_dim",
        "num_heads",
        "num_kv_heads",
        "num_layers",
        "seq_len",
        "vocab_size",
    }
    float_fields = {"ff_mult", "rope_theta", "rope_fraction"}
    bool_fields = {"tie_weights"}

    for k, v_list in parsed.items():
        v = v_list[0]  # Take first value if multiple

        if k in int_fields:
            config[k] = int(v)
        elif k in float_fields:
            config[k] = float(v)
        elif k in bool_fields:
            config[k] = v.lower() == "true"
        else:
            config[k] = v  # String fields

    return config


def explain_constraint(field_name: str, error_msg: str) -> str:
    """
    Provide educational explanations for why certain constraints exist.

    This helps users understand transformer architecture requirements.
    """
    explanations = {
        "head_divisibility": (
            "Attention heads must evenly divide hidden_dim because each head "
            "processes hidden_dim/num_heads dimensions. Non-integer head dimensions "
            "would break the attention computation."
        ),
        "gqa_divisibility": (
            "In Grouped Query Attention, query heads are grouped to share "
            "key/value heads. num_heads must be divisible by num_kv_heads "
            "to ensure equal group sizes."
        ),
        "head_alignment": (
            "Modern GPUs perform best when head dimensions are multiples of 8 "
            "(for float16/bfloat16) or 16 (for FlashAttention) due to tensor "
            "core architecture optimizations."
        ),
        "rope_even": (
            "RoPE (Rotary Position Embedding) applies complex rotations, "
            "requiring even dimensions to form real/imaginary pairs for "
            "each frequency component."
        ),
    }

    # Simple keyword matching to provide relevant explanations
    if "divisible by num_heads" in error_msg:
        return explanations["head_divisibility"]
    if "divisible by num_kv_heads" in error_msg:
        return explanations["gqa_divisibility"]
    if "multiple of" in error_msg and "head_dim" in error_msg:
        return explanations["head_alignment"]
    if "RoPE dimension" in error_msg and "even" in error_msg:
        return explanations["rope_even"]
    return "This constraint ensures mathematical consistency in the transformer architecture."
