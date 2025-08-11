"""
PyTorch implementation of complete transformer model.

This shows production-ready transformer implementation with GPU support,
automatic differentiation, and modern ML best practices.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..abstract import AbstractTransformer, AbstractTransformerBlock, BackendConfig
from .attention import TorchAttention
from .feed_forward import TorchFeedForward
from .normalization import create_torch_normalization


class TorchTransformerBlock(AbstractTransformerBlock, nn.Module):
    """PyTorch transformer block implementation."""

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
        # Initialize both parent classes
        AbstractTransformerBlock.__init__(
            self, hidden_dim, num_heads, ff_dim, norm_type,
            activation_type, residual_type, dropout
        )
        nn.Module.__init__(self)

        # Create components
        self.attention = TorchAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = TorchFeedForward(hidden_dim, ff_dim, activation_type, residual_type)

        # Create normalization layers
        self.norm1 = create_torch_normalization(norm_type, hidden_dim)
        self.norm2 = create_torch_normalization(norm_type, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        if self.residual_type == "Pre-LN":
            # Pre-LN: normalize before attention
            x_norm = self.norm1(x)
            attn_output, attn_stats = self.attention.forward(x_norm, mask)
            x = x + attn_output
        else:
            # Post-LN: attention first, then normalize
            attn_output, attn_stats = self.attention.forward(x, mask)
            x = x + attn_output
            x = self.norm1(x)

        # Apply dropout if specified
        if self.dropout is not None and self.training:
            x = self.dropout(x)

        # Feed-forward with residual connection
        if self.residual_type == "Pre-LN":
            x_norm = self.norm2(x)
            ff_output, ff_stats = self.feed_forward.forward(x_norm, None)
            x = x + ff_output
        else:
            ff_output, ff_stats = self.feed_forward.forward(x, None)
            x = x + ff_output
            x = self.norm2(x)

        # Apply dropout if specified
        if self.dropout is not None and self.training:
            x = self.dropout(x)

        # Collect statistics
        block_stats = {
            "attention": attn_stats,
            "feed_forward": ff_stats,
            "norm_type": self.norm_type,
            "activation_type": self.activation_type,
            "residual_type": self.residual_type,
        }

        return x, block_stats

    def backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Backward pass (handled automatically by PyTorch)."""
        # This is handled automatically by PyTorch's autograd
        # We maintain the interface for compatibility
        grad_input = torch.zeros_like(grad_output)

        # Collect gradients from components
        _, attn_gradients = self.attention.backward(grad_output)
        _, ff_gradients = self.feed_forward.backward(grad_output)

        # Combine gradients
        all_gradients = {}
        all_gradients.update({f"attention_{k}": v for k, v in attn_gradients.items()})
        all_gradients.update({f"feed_forward_{k}": v for k, v in ff_gradients.items()})

        return grad_input.detach().cpu().numpy(), all_gradients

    def get_parameters(self) -> list[Any]:
        """Get trainable parameters."""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy())
        return params

    def get_pytorch_parameters(self) -> list[torch.nn.Parameter]:
        """Get PyTorch parameters for optimizer."""
        return list(self.parameters())


class TorchTransformer(AbstractTransformer, nn.Module):
    """PyTorch complete transformer implementation."""

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
        backend_config: BackendConfig | None = None,
    ):
        # Initialize both parent classes
        AbstractTransformer.__init__(
            self, vocab_size, hidden_dim, num_layers, num_heads, ff_dim,
            max_seq_len, norm_type, activation_type, residual_type,
            pos_encoding_type, dropout, backend_config
        )
        nn.Module.__init__(self)

        # Determine device
        self.device = torch.device(
            backend_config.device if backend_config and backend_config.device != "cpu"
            and torch.cuda.is_available() else "cpu"
        )

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        if pos_encoding_type == "Sinusoidal":
            self.pos_encoding = self._create_sinusoidal_encoding(max_seq_len, hidden_dim)
            self.register_buffer('positional_encoding', self.pos_encoding)
        elif pos_encoding_type == "Learnable":
            self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TorchTransformerBlock(
                hidden_dim, num_heads, ff_dim, norm_type,
                activation_type, residual_type, dropout
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        if residual_type == "Pre-LN":
            self.final_norm = create_torch_normalization(norm_type, hidden_dim)
        else:
            self.final_norm = nn.Identity()

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=True)

        # Initialize weights
        self._init_weights()

        # Move to device
        self.to(self.device)

    def _create_sinusoidal_encoding(self, max_len: int, hidden_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                            -(math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _init_weights(self):
        """Initialize model weights properly."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

        if hasattr(self, 'pos_encoding') and isinstance(self.pos_encoding, nn.Embedding):
            nn.init.normal_(self.pos_encoding.weight, mean=0, std=0.02)

        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

        # Initialize transformer blocks (done in their own __init__)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass through complete transformer."""
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x, dtype=torch.long, device=self.device)
        if targets is not None and isinstance(targets, (list, tuple)):
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)

        batch_size, seq_len = x.size()

        # Token embeddings
        embeddings = self.token_embedding(x)  # (batch, seq, hidden)

        # Add positional encoding
        if self.pos_encoding_type == "Sinusoidal":
            pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)  # (1, seq, hidden)
            embeddings = embeddings + pos_enc
        elif hasattr(self.pos_encoding, 'weight'):
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = self.pos_encoding(positions)
            embeddings = embeddings + pos_embeddings

        # Pass through transformer blocks
        h = embeddings
        layer_stats = []

        for i, block in enumerate(self.blocks):
            h, block_stats = block.forward(h, mask=None)  # No causal mask for simplicity
            layer_stats.append(block_stats)

        # Final normalization
        h = self.final_norm(h)

        # Output projection
        logits = self.output_projection(h)  # (batch, seq, vocab)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross entropy: (batch*seq, vocab) and (batch*seq,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0  # Ignore padding tokens
            )
            self.loss_history.append(loss.item())
            self.step_count += 1

        # Collect model statistics
        with torch.no_grad():
            model_stats = {
                "loss": loss.item() if loss is not None else None,
                "layer_stats": layer_stats,
                "embeddings_mean": embeddings.mean().item(),
                "embeddings_std": embeddings.std().item(),
                "final_hidden_mean": h.mean().item(),
                "final_hidden_std": h.std().item(),
                "logits_mean": logits.mean().item(),
                "logits_std": logits.std().item(),
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

    def backward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, dict[str, Any]]:
        """Backward pass (handled automatically by PyTorch)."""
        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0
        )

        # Backward pass is handled automatically by PyTorch
        # when we call loss.backward()

        # For compatibility, return loss and dummy gradients
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy()
            else:
                gradients[name] = torch.zeros_like(param).cpu().numpy()

        return loss.item(), gradients

    def train_step(self, x: torch.Tensor, targets: torch.Tensor, optimizer) -> float:
        """Single training step using PyTorch's autograd."""
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x, dtype=torch.long, device=self.device)
        if isinstance(targets, (list, tuple)):
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits, _ = self.forward(x, targets)

        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0
        )

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        return loss.item()

    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None
    ) -> torch.Tensor:
        """Generate text with advanced sampling strategies."""
        if isinstance(prompt, (list, tuple)):
            prompt = torch.tensor(prompt, dtype=torch.long, device=self.device)

        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            batch_size = prompt.size(0)
            generated = prompt.clone()

            for _ in range(max_length):
                # Forward pass
                logits, _ = self.forward(generated)

                # Get next token logits (last position)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < top_k_logits[:, -1:]] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=1)

        self.train()  # Set back to training mode
        return generated

    def get_parameters(self) -> list[Any]:
        """Get trainable parameters."""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy())
        return params

    def get_parameter_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str) -> None:
        """Save model parameters."""
        torch.save({
            'model_state_dict': self.state_dict(),
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
        }, path)

    def load(self, path: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def _estimate_memory_usage(self) -> dict[str, float]:
        """Estimate memory usage."""
        param_count = self.get_parameter_count()

        # Estimate based on data type
        if self.backend_config and self.backend_config.dtype == "float16":
            bytes_per_param = 2
        else:
            bytes_per_param = 4

        param_memory_mb = (param_count * bytes_per_param) / (1024 * 1024)

        # Estimate activation memory (rough approximation)
        activation_memory_mb = param_memory_mb * 2  # Forward + backward pass

        return {
            "parameters_mb": param_memory_mb,
            "activations_mb": activation_memory_mb,
            "total_mb": param_memory_mb + activation_memory_mb,
        }

    def get_pytorch_parameters(self) -> list[torch.nn.Parameter]:
        """Get PyTorch parameters for optimizer."""
        return list(self.parameters())
