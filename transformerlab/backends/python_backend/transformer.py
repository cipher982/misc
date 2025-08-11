"""
Pure Python implementation of complete transformer model.

This is the main transformer class that combines all components
and shows the complete training process step-by-step.
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Any

from ..abstract import AbstractTransformer, AbstractTransformerBlock, BackendConfig
from .attention import PythonAttention
from .feed_forward import PythonFeedForward
from .normalization import create_normalization
from .utils import (
    zeros, randn, matmul_3d, add_3d, get_shape, copy_tensor, 
    mean, std, softmax_2d
)


class PythonTransformerBlock(AbstractTransformerBlock):
    """Pure Python transformer block implementation."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        norm_type: str = "LayerNorm",
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim, num_heads, ff_dim, norm_type, activation_type, residual_type, dropout)
        
        # Create components
        self.attention = PythonAttention(hidden_dim, num_heads, dropout)
        self.ff = PythonFeedForward(hidden_dim, ff_dim, activation_type, residual_type)
        
        # Create normalization modules
        self.norm1 = create_normalization(norm_type, hidden_dim)
        self.norm2 = create_normalization(norm_type, hidden_dim)
        
        # Cache for backward pass
        self._cache = {}
    
    def forward(self, x: List[List[List[float]]], mask: Optional[Any] = None) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
        """Forward pass through transformer block with explicit steps."""
        print(f"\\n[PythonTransformerBlock] Forward pass: {self.residual_type}")
        print(f"  Input shape: {get_shape(x)}")
        
        # Self-attention with residual connection
        if self.residual_type == "Pre-LN":
            print("  Pre-LN: Normalize before attention")
            # Pre-LN: normalize before attention
            x_norm = self.norm1.forward(x) if self.norm1 else x
            attn_output, attn_stats = self.attention.forward(x_norm, mask)
            x = self._add_tensors(x, attn_output)  # Residual connection
        else:
            print("  Post-LN: Attention first, then normalize")
            # Post-LN: attention first, then normalize
            attn_output, attn_stats = self.attention.forward(x, mask)
            x = self._add_tensors(x, attn_output)  # Residual connection
            if self.norm1:
                x = self.norm1.forward(x)
        
        print(f"  After attention + residual: {get_shape(x)}")
        
        # Feed-forward with residual connection
        ff_output, ff_stats = self.ff.forward(x, self.norm2)
        
        print(f"  After feed-forward: {get_shape(ff_output)}")
        
        # Cache for backward pass
        self._cache = {
            'input': copy_tensor(x),
            'attn_output': copy_tensor(attn_output),
        }
        
        # Collect statistics
        block_stats = {
            "attention": attn_stats,
            "feed_forward": ff_stats,
            "norm_type": self.norm_type,
            "activation_type": self.activation_type,
            "residual_type": self.residual_type,
        }
        
        return ff_output, block_stats
    
    def _add_tensors(self, a: List[List[List[float]]], b: List[List[List[float]]]) -> List[List[List[float]]]:
        """Element-wise addition of two 3D tensors."""
        batch_size, seq_len, hidden_dim = get_shape(a)
        result = zeros((batch_size, seq_len, hidden_dim))
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                for dim in range(hidden_dim):
                    result[batch][seq][dim] = a[batch][seq][dim] + b[batch][seq][dim]
        
        return result
    
    def backward(self, grad_output: List[List[List[float]]]) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
        """Backward pass (simplified)."""
        if not self._cache:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        print("[PythonTransformerBlock] Backward pass (simplified)...")
        
        # Simplified gradients
        input_shape = get_shape(self._cache['input'])
        grad_input = zeros(input_shape)
        
        # Get gradients from components
        _, attn_gradients = self.attention.backward(grad_output)
        _, ff_gradients = self.ff.backward(grad_output)
        
        # Combine gradients
        all_gradients = {}
        all_gradients.update({f"attention_{k}": v for k, v in attn_gradients.items()})
        all_gradients.update({f"ff_{k}": v for k, v in ff_gradients.items()})
        
        return grad_input, all_gradients
    
    def get_parameters(self) -> List[Any]:
        """Get all trainable parameters."""
        params = []
        params.extend(self.attention.get_parameters())
        params.extend(self.ff.get_parameters())
        
        if self.norm1:
            params.extend(self.norm1.get_parameters())
        if self.norm2:
            params.extend(self.norm2.get_parameters())
        
        return params


class PythonTransformer(AbstractTransformer):
    """Pure Python complete transformer implementation."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        max_seq_len: int = 512,
        norm_type: str = "LayerNorm",
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
        pos_encoding_type: str = "Sinusoidal",
        dropout: float = 0.0,
        backend_config: Optional[BackendConfig] = None,
    ):
        super().__init__(
            vocab_size, hidden_dim, num_layers, num_heads, ff_dim,
            max_seq_len, norm_type, activation_type, residual_type,
            pos_encoding_type, dropout, backend_config
        )
        
        print(f"\\n[PythonTransformer] Initializing with:")
        print(f"  vocab_size={vocab_size}, hidden_dim={hidden_dim}")
        print(f"  num_layers={num_layers}, num_heads={num_heads}")
        print(f"  Backend: Pure Python (maximum transparency)")
        
        # Token embeddings (vocab_size, hidden_dim)
        self.token_embedding = randn((vocab_size, hidden_dim))
        
        # Positional encoding (simplified - just add to embeddings)
        self.pos_encoding = randn((max_seq_len, hidden_dim), scale=0.01)
        
        # Transformer blocks
        self.blocks = []
        for i in range(num_layers):
            print(f"  Creating transformer block {i+1}/{num_layers}")
            block = PythonTransformerBlock(
                hidden_dim, num_heads, ff_dim, norm_type, 
                activation_type, residual_type, dropout
            )
            self.blocks.append(block)
        
        # Final normalization
        self.final_norm = create_normalization(norm_type, hidden_dim)
        
        # Output projection (hidden_dim, vocab_size)
        self.output_projection = randn((hidden_dim, vocab_size))
        self.output_bias = [0.0] * vocab_size
        
        # Cache for backward pass
        self._forward_cache = {}
        
        print(f"  Total parameters: {self.get_parameter_count()}")
    
    def forward(self, x: List[List[int]], targets: Optional[List[List[int]]] = None) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
        """Forward pass through complete transformer."""
        batch_size, seq_len = len(x), len(x[0])
        
        print(f"\\n[PythonTransformer] Forward pass:")
        print(f"  Input shape: batch_size={batch_size}, seq_len={seq_len}")
        
        # Step 1: Token embeddings
        print("  Step 1: Token embedding lookup...")
        embeddings = zeros((batch_size, seq_len, self.hidden_dim))
        for batch in range(batch_size):
            for seq in range(seq_len):
                token_id = x[batch][seq]
                for dim in range(self.hidden_dim):
                    embeddings[batch][seq][dim] = self.token_embedding[token_id][dim]
        
        # Step 2: Add positional encoding
        print("  Step 2: Adding positional encoding...")
        for batch in range(batch_size):
            for seq in range(seq_len):
                for dim in range(self.hidden_dim):
                    embeddings[batch][seq][dim] += self.pos_encoding[seq][dim]
        
        # Step 3: Pass through transformer blocks
        h = embeddings
        layer_stats = []
        
        for i, block in enumerate(self.blocks):
            print(f"  Step 3.{i+1}: Transformer block {i+1}/{len(self.blocks)}")
            h, block_stats = block.forward(h, mask=None)  # No mask for simplicity
            layer_stats.append(block_stats)
        
        # Step 4: Final normalization
        if self.final_norm:
            print("  Step 4: Final layer normalization...")
            h = self.final_norm.forward(h)
        
        # Step 5: Output projection
        print("  Step 5: Output projection to vocabulary...")
        logits = add_3d(matmul_3d(h, self.output_projection), self.output_bias)
        
        print(f"  Final logits shape: {get_shape(logits)}")
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            print("  Computing cross-entropy loss...")
            loss = self._compute_loss(logits, targets)
            self.loss_history.append(loss)
            self.step_count += 1
            print(f"  Loss: {loss:.6f}")
        
        # Cache for backward pass
        self._forward_cache = {
            'embeddings': copy_tensor(embeddings),
            'final_hidden': copy_tensor(h),
        }
        
        # Collect model statistics
        model_stats = {
            "loss": loss,
            "layer_stats": layer_stats,
            "embeddings_mean": mean(embeddings),
            "embeddings_std": std(embeddings),
            "final_hidden_mean": mean(h),
            "final_hidden_std": std(h),
            "logits_mean": mean(logits),
            "logits_std": std(logits),
            "config": {
                "norm_type": self.norm_type,
                "activation_type": self.activation_type,
                "residual_type": self.residual_type,
                "pos_encoding_type": self.pos_encoding_type,
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
            },
        }
        
        return logits, model_stats
    
    def _compute_loss(self, logits: List[List[List[float]]], targets: List[List[int]]) -> float:
        """Compute cross-entropy loss with explicit steps."""
        batch_size, seq_len, vocab_size = get_shape(logits)
        
        print(f"    Computing loss for {batch_size} batches, {seq_len} tokens each")
        
        total_loss = 0.0
        num_tokens = 0
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                target_idx = targets[batch][seq]
                
                if target_idx > 0:  # Skip padding tokens
                    # Get logits for this position
                    position_logits = logits[batch][seq]
                    
                    # Apply softmax (with numerical stability)
                    max_logit = max(position_logits)
                    exp_logits = [math.exp(logit - max_logit) for logit in position_logits]
                    exp_sum = sum(exp_logits)
                    probs = [exp_logit / exp_sum for exp_logit in exp_logits]
                    
                    # Compute negative log likelihood
                    target_prob = probs[target_idx]
                    total_loss -= math.log(target_prob + 1e-8)
                    num_tokens += 1
        
        return total_loss / max(num_tokens, 1)
    
    def backward(self, logits: List[List[List[float]]], targets: List[List[int]]) -> Tuple[float, Dict[str, Any]]:
        """Backward pass (simplified for educational purposes)."""
        print("\\n[PythonTransformer] Backward pass (simplified)...")
        
        # Compute loss
        loss = self._compute_loss(logits, targets)
        
        # For educational purposes, return zero gradients
        gradients = {}
        
        # Add zero gradients for all parameters
        param_names = self.get_parameter_names()
        for name in param_names:
            if "embedding" in name:
                gradients[name] = zeros(get_shape(self.token_embedding))
            elif "projection" in name:
                gradients[name] = zeros(get_shape(self.output_projection))
            elif "bias" in name:
                gradients[name] = [0.0] * len(self.output_bias)
            # Block gradients would be more complex...
        
        return loss, gradients
    
    def train_step(self, x: List[List[int]], targets: List[List[int]], optimizer) -> float:
        """Single training step with detailed logging."""
        print(f"\\n[PythonTransformer] Training step {self.step_count + 1}")
        
        # Forward pass
        logits, stats = self.forward(x, targets)
        
        # Backward pass
        loss, gradients = self.backward(logits, targets)
        
        # Parameter update (simplified)
        print("  Updating parameters...")
        parameters = self.get_parameters()
        grad_list = [gradients.get(name, self._zero_like_param(param)) 
                     for name, param in zip(self.get_parameter_names(), parameters)]
        
        optimizer.update_parameters(parameters, grad_list)
        
        print(f"  Training step complete. Loss: {loss:.6f}")
        return loss
    
    def _zero_like_param(self, param: Any) -> Any:
        """Create zero gradient with same shape as parameter."""
        return zeros(get_shape(param))
    
    def generate(self, prompt: List[List[int]], max_length: int = 50, temperature: float = 1.0) -> List[List[int]]:
        """Generate text with detailed steps."""
        print(f"\\n[PythonTransformer] Generating {max_length} tokens at temperature={temperature}")
        
        batch_size = len(prompt)
        generated = copy_tensor(prompt)
        
        for step in range(max_length):
            print(f"  Generation step {step + 1}/{max_length}")
            
            # Forward pass
            logits, _ = self.forward(generated)
            
            # Get next token logits (last position)
            next_tokens = []
            for batch in range(batch_size):
                last_logits = logits[batch][-1]  # Last position logits
                
                # Apply temperature
                scaled_logits = [logit / temperature for logit in last_logits]
                
                # Apply softmax
                max_logit = max(scaled_logits)
                exp_logits = [math.exp(logit - max_logit) for logit in scaled_logits]
                exp_sum = sum(exp_logits)
                probs = [exp_logit / exp_sum for exp_logit in exp_logits]
                
                # Sample next token
                rand_val = random.random()
                cumsum = 0.0
                next_token = 0
                for token_id, prob in enumerate(probs):
                    cumsum += prob
                    if rand_val <= cumsum:
                        next_token = token_id
                        break
                
                next_tokens.append(next_token)
            
            # Append to generated sequence
            for batch in range(batch_size):
                generated[batch].append(next_tokens[batch])
        
        print(f"  Generation complete. Final length: {len(generated[0])}")
        return generated
    
    def get_parameters(self) -> List[Any]:
        """Get all trainable parameters."""
        params = [self.token_embedding]
        
        for block in self.blocks:
            params.extend(block.get_parameters())
        
        if self.final_norm:
            params.extend(self.final_norm.get_parameters())
        
        params.extend([self.output_projection, self.output_bias])
        
        return params
    
    def get_parameter_names(self) -> List[str]:
        """Get parameter names."""
        names = ['token_embedding']
        
        for i, block in enumerate(self.blocks):
            # This would need to be implemented properly
            names.extend([f'block_{i}_param_{j}' for j in range(len(block.get_parameters()))])
        
        if self.final_norm:
            names.extend(['final_norm_gamma', 'final_norm_beta'])
        
        names.extend(['output_projection', 'output_bias'])
        
        return names
    
    def get_parameter_count(self) -> int:
        """Count total parameters."""
        count = 0
        for param in self.get_parameters():
            shape = get_shape(param)
            param_count = 1
            for dim in shape:
                param_count *= dim
            count += param_count
        return count
    
    def save(self, path: str) -> None:
        """Save model (placeholder)."""
        raise NotImplementedError("Saving not implemented for Python backend")
    
    def load(self, path: str) -> None:
        """Load model (placeholder)."""
        raise NotImplementedError("Loading not implemented for Python backend")
    
    def _estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage."""
        # Rough estimate based on parameter count
        param_count = self.get_parameter_count()
        # Assume 4 bytes per float32 parameter
        param_memory_mb = (param_count * 4) / (1024 * 1024)
        
        return {
            "parameters_mb": param_memory_mb,
            "activations_mb": param_memory_mb * 0.5,  # Rough estimate
            "total_mb": param_memory_mb * 1.5,
        }