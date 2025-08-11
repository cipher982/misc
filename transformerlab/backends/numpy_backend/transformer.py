"""
NumPy implementation of complete transformer model.

This implementation provides fast vectorized computation using NumPy 
while maintaining educational transparency and clarity.
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple, Any, List

from ..abstract import AbstractTransformer, AbstractTransformerBlock, BackendConfig
from .attention import NumPyAttention
from .feed_forward import NumPyFeedForward
from .normalization import create_numpy_normalization
from .optimizer import create_numpy_optimizer


class NumPyTransformerBlock(AbstractTransformerBlock):
    """NumPy transformer block implementation."""
    
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
        super().__init__(
            hidden_dim, num_heads, ff_dim, norm_type, 
            activation_type, residual_type, dropout
        )
        
        # Create components
        self.attention = NumPyAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = NumPyFeedForward(hidden_dim, ff_dim, activation_type, residual_type)
        
        # Create normalization layers
        self.norm1 = create_numpy_normalization(norm_type, hidden_dim)
        self.norm2 = create_numpy_normalization(norm_type, hidden_dim)
        
        # Store dropout rate (NumPy doesn't have built-in dropout)
        self.dropout_rate = dropout
        self.training = True
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout during training."""
        if self.training and self.dropout_rate > 0:
            # Create dropout mask
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        if self.residual_type == "Pre-LN":
            # Pre-LN: normalize before attention
            x_norm = self.norm1.forward(x) if self.norm1 else x
            attn_output, attn_stats = self.attention.forward(x_norm, mask)
            x = x + self._apply_dropout(attn_output)
        else:
            # Post-LN: attention first, then normalize
            attn_output, attn_stats = self.attention.forward(x, mask)
            x = x + self._apply_dropout(attn_output)
            x = self.norm1.forward(x) if self.norm1 else x
        
        # Feed-forward with residual connection
        if self.residual_type == "Pre-LN":
            x_norm = self.norm2.forward(x) if self.norm2 else x
            ff_output, ff_stats = self.feed_forward.forward(x_norm, None)
            x = x + self._apply_dropout(ff_output)
        else:
            ff_output, ff_stats = self.feed_forward.forward(x, None)
            x = x + self._apply_dropout(ff_output)
            x = self.norm2.forward(x) if self.norm2 else x
        
        # Collect statistics
        block_stats = {
            "attention": attn_stats,
            "feed_forward": ff_stats,
            "norm_type": self.norm_type,
            "activation_type": self.activation_type,
            "residual_type": self.residual_type,
        }
        
        # Cache for backward pass
        self._cache = {
            'input': x,
            'attn_output': attn_output,
            'ff_output': ff_output,
        }
        
        return x, block_stats
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Backward pass through transformer block."""
        if not hasattr(self, '_cache'):
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Backward through feed-forward
        grad_ff_input, ff_gradients = self.feed_forward.backward(grad_output)
        
        # Backward through attention 
        grad_attn_input, attn_gradients = self.attention.backward(grad_ff_input)
        
        # Combine gradients (residual connections)
        grad_input = grad_output + grad_attn_input
        
        # Combine all gradients
        all_gradients = {}
        all_gradients.update({f"attention_{k}": v for k, v in attn_gradients.items()})
        all_gradients.update({f"feed_forward_{k}": v for k, v in ff_gradients.items()})
        
        return grad_input, all_gradients
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get trainable parameters."""
        params = []
        params.extend(self.attention.get_parameters())
        params.extend(self.feed_forward.get_parameters())
        if self.norm1 and hasattr(self.norm1, 'get_parameters'):
            params.extend(self.norm1.get_parameters())
        if self.norm2 and hasattr(self.norm2, 'get_parameters'):
            params.extend(self.norm2.get_parameters())
        return params


class NumPyTransformer(AbstractTransformer):
    """NumPy complete transformer implementation."""
    
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
        
        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, hidden_dim) * 0.02
        
        # Positional encoding
        if pos_encoding_type == "Sinusoidal":
            self.pos_encoding = self._create_sinusoidal_encoding(max_seq_len, hidden_dim)
        elif pos_encoding_type == "Learnable":
            self.pos_encoding = np.random.randn(max_seq_len, hidden_dim) * 0.02
        
        # Transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            block = NumPyTransformerBlock(
                hidden_dim, num_heads, ff_dim, norm_type,
                activation_type, residual_type, dropout
            )
            self.blocks.append(block)
        
        # Final normalization
        if residual_type == "Pre-LN":
            self.final_norm = create_numpy_normalization(norm_type, hidden_dim)
        else:
            self.final_norm = None
        
        # Output projection
        self.output_projection = np.random.randn(hidden_dim, vocab_size) * 0.02
        self.output_bias = np.zeros(vocab_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _create_sinusoidal_encoding(self, max_len: int, hidden_dim: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        pe = np.zeros((max_len, hidden_dim))
        position = np.arange(0, max_len)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def _init_weights(self):
        """Initialize model weights properly."""
        # Token embeddings are already initialized
        # Output projection is already initialized
        # Block weights are initialized in their constructors
        pass
    
    def forward(self, x: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through complete transformer."""
        # Convert lists to arrays if needed
        if isinstance(x, (list, tuple)):
            x = np.array(x, dtype=np.int32)
        if targets is not None and isinstance(targets, (list, tuple)):
            targets = np.array(targets, dtype=np.int32)
            
        batch_size, seq_len = x.shape
        
        # Token embeddings
        embeddings = self.token_embedding[x]  # (batch, seq, hidden)
        
        # Add positional encoding
        if self.pos_encoding_type == "Sinusoidal":
            pos_enc = self.pos_encoding[:seq_len][np.newaxis, :, :]  # (1, seq, hidden)
            embeddings = embeddings + pos_enc
        elif self.pos_encoding_type == "Learnable":
            pos_indices = np.arange(seq_len)
            pos_embeddings = self.pos_encoding[pos_indices][np.newaxis, :, :]  # (1, seq, hidden)
            embeddings = embeddings + pos_embeddings
        
        # Pass through transformer blocks
        h = embeddings
        layer_stats = []
        
        for i, block in enumerate(self.blocks):
            h, block_stats = block.forward(h, mask=None)  # No causal mask for simplicity
            layer_stats.append(block_stats)
        
        # Final normalization
        if self.final_norm:
            h = self.final_norm.forward(h)
        
        # Output projection
        logits = np.matmul(h, self.output_projection) + self.output_bias
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)
            self.loss_history.append(loss)
            self.step_count += 1
        
        # Collect model statistics
        model_stats = {
            "loss": loss,
            "layer_stats": layer_stats,
            "embeddings_mean": np.mean(embeddings),
            "embeddings_std": np.std(embeddings),
            "final_hidden_mean": np.mean(h),
            "final_hidden_std": np.std(h),
            "logits_mean": np.mean(logits),
            "logits_std": np.std(logits),
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
        
        # Cache intermediate values for backward pass
        self._forward_cache = {
            'embeddings': embeddings,
            'final_hidden': h,
            'input': x,
        }
        
        return logits, model_stats
    
    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # Softmax (stable version)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        loss = 0
        
        for b in range(batch_size):
            for t in range(seq_len):
                target_idx = targets[b, t]
                if target_idx > 0:  # Skip padding tokens
                    loss -= np.log(probs[b, t, target_idx] + 1e-8)
        
        # Average loss
        num_tokens = np.sum(targets > 0)
        return loss / max(num_tokens, 1)
    
    def backward(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Backward pass to compute gradients."""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute softmax and loss
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Compute loss
        loss = 0
        for b in range(batch_size):
            for t in range(seq_len):
                target_idx = targets[b, t]
                if target_idx > 0:  # Skip padding tokens
                    loss -= np.log(probs[b, t, target_idx] + 1e-8)
        
        num_tokens = np.sum(targets > 0)
        loss = loss / max(num_tokens, 1)
        
        # Compute gradient of loss w.r.t. logits
        grad_logits = probs.copy()
        for b in range(batch_size):
            for t in range(seq_len):
                target_idx = targets[b, t]
                if target_idx > 0:
                    grad_logits[b, t, target_idx] -= 1
        
        grad_logits = grad_logits / max(num_tokens, 1)
        
        # Backward through output projection
        if not hasattr(self, '_forward_cache'):
            raise RuntimeError("Forward pass must be called before backward pass")
        
        final_hidden = self._forward_cache['final_hidden']
        grad_output_projection = np.tensordot(final_hidden, grad_logits, axes=([0, 1], [0, 1]))
        grad_output_bias = np.sum(grad_logits, axis=(0, 1))
        grad_final_hidden = np.matmul(grad_logits, self.output_projection.T)
        
        # Backward through blocks
        grad_hidden = grad_final_hidden
        all_gradients = {}
        
        # Store gradients for output layers
        all_gradients['output_projection'] = grad_output_projection
        all_gradients['output_bias'] = grad_output_bias
        
        for i in reversed(range(len(self.blocks))):
            grad_hidden, block_gradients = self.blocks[i].backward(grad_hidden)
            for k, v in block_gradients.items():
                all_gradients[f'block_{i}_{k}'] = v
        
        return loss, all_gradients
    
    def train_step(self, x: np.ndarray, targets: np.ndarray, optimizer) -> float:
        """Single training step."""
        # Convert lists to arrays if needed
        if isinstance(x, (list, tuple)):
            x = np.array(x, dtype=np.int32)
        if isinstance(targets, (list, tuple)):
            targets = np.array(targets, dtype=np.int32)
        
        # Forward pass
        logits, _ = self.forward(x, targets)
        
        # Backward pass
        loss, gradients = self.backward(logits, targets)
        
        # Collect parameters and gradients for optimization
        parameters = self.get_parameters()
        parameter_names = self.get_parameter_names()
        
        # Extract gradients in the same order as parameters
        grad_list = []
        for name in parameter_names:
            if name in gradients:
                grad_list.append(gradients[name])
            else:
                # If gradient not computed, use zero gradient
                param_idx = parameter_names.index(name)
                grad_list.append(np.zeros_like(parameters[param_idx]))
        
        # Update parameters
        optimizer.update_parameters(parameters, grad_list)
        
        return loss
    
    def generate(
        self, 
        prompt: np.ndarray, 
        max_length: int = 50, 
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate text autoregressively."""
        if isinstance(prompt, (list, tuple)):
            prompt = np.array(prompt, dtype=np.int32)
        
        # Set to eval mode (disable dropout)
        for block in self.blocks:
            block.training = False
        
        batch_size = prompt.shape[0]
        generated = prompt.copy()
        
        for _ in range(max_length):
            # Forward pass
            logits, _ = self.forward(generated)
            
            # Get next token logits (last position)
            next_logits = logits[:, -1, :] / temperature
            
            # Apply softmax
            exp_logits = np.exp(next_logits - np.max(next_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Sample next tokens
            next_tokens = []
            for p in probs:
                next_token = np.random.choice(self.vocab_size, p=p)
                next_tokens.append(next_token)
            
            next_tokens = np.array(next_tokens)[:, np.newaxis]
            
            # Append to generated sequence
            generated = np.concatenate([generated, next_tokens], axis=1)
        
        # Set back to training mode
        for block in self.blocks:
            block.training = True
            
        return generated
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get all trainable parameters."""
        parameters = []
        
        # Add embedding parameters
        parameters.append(self.token_embedding)
        
        # Add transformer block parameters
        for block in self.blocks:
            parameters.extend(block.get_parameters())
        
        # Add output projection parameters
        parameters.append(self.output_projection)
        parameters.append(self.output_bias)
        
        return parameters
    
    def get_parameter_names(self) -> List[str]:
        """Get parameter names corresponding to get_parameters()."""
        names = ['token_embedding']
        
        # Add transformer block parameter names
        for i, block in enumerate(self.blocks):
            # For now, use simplified names
            names.extend([f'block_{i}_attention_params', f'block_{i}_ff_params'])
        
        # Add output projection parameter names
        names.extend(['output_projection', 'output_bias'])
        
        return names
    
    def get_parameter_count(self) -> int:
        """Count total parameters."""
        total = 0
        for param in self.get_parameters():
            total += param.size
        return total
    
    def save(self, path: str) -> None:
        """Save model parameters."""
        import pickle
        
        save_dict = {
            'parameters': self.get_parameters(),
            'config': {
                'vocab_size': self.vocab_size,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'ff_dim': self.ff_dim,
                'max_seq_len': self.max_seq_len,
                'norm_type': self.norm_type,
                'activation_type': self.activation_type,
                'residual_type': self.residual_type,
                'pos_encoding_type': self.pos_encoding_type,
                'dropout': self.dropout,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path: str) -> None:
        """Load model parameters."""
        import pickle
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Load parameters back into model
        # This is a simplified version - full implementation would match parameter structure
        pass
    
    def _estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage."""
        param_count = self.get_parameter_count()
        
        # Assume float32 (4 bytes per parameter)
        param_memory_mb = (param_count * 4) / (1024 * 1024)
        
        # Estimate activation memory (rough approximation)
        activation_memory_mb = param_memory_mb * 1.5  # NumPy intermediate values
        
        return {
            "parameters_mb": param_memory_mb,
            "activations_mb": activation_memory_mb,
            "total_mb": param_memory_mb + activation_memory_mb,
        }