"""
PyTorch implementation of feed-forward network.

Shows modern ML practices with proper initialization, activation functions,
and efficient computation using PyTorch operations.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..abstract import AbstractFeedForward


class TorchFeedForward(AbstractFeedForward, nn.Module):
    """PyTorch feed-forward network implementation."""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
    ):
        # Initialize both parent classes
        AbstractFeedForward.__init__(
            self, hidden_dim, ff_dim, activation_type, residual_type
        )
        nn.Module.__init__(self)

        # Linear layers
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)

        # Activation function
        self.activation = self._get_activation_fn(activation_type)

        # Dropout (optional)
        self.dropout = (
            nn.Dropout(0.1) if residual_type in ["Pre-LN", "Post-LN"] else None
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Xavier uniform initialization for linear layers
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

        # Zero bias initialization
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def _get_activation_fn(self, activation_type: str):
        """Get PyTorch activation function."""
        if activation_type == "ReLU":
            return nn.ReLU()
        if activation_type == "GeLU":
            return nn.GELU()
        if activation_type == "Swish" or activation_type == "SiLU":
            return nn.SiLU()  # Swish is also called SiLU in PyTorch
        if activation_type == "SwiGLU":
            return self._swiglu_activation
        return nn.Identity()  # Linear activation

    def _swiglu_activation(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation function implementation."""
        # Split the input in half along the last dimension
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

    def forward(
        self, x: torch.Tensor, norm_module: Any | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass through feed-forward network."""
        batch_size, seq_len, hidden_dim = x.size()

        # Store input for residual connection
        residual = x

        # First linear transformation
        hidden = self.linear1(x)  # (batch, seq, ff_dim)

        # Apply activation
        if self.activation_type == "SwiGLU":
            # For SwiGLU, we need to adjust the first linear layer output size
            # This is a simplified implementation
            hidden = self.activation(hidden)
        else:
            hidden = self.activation(hidden)

        # Apply dropout if in training mode
        if self.dropout is not None and self.training:
            hidden = self.dropout(hidden)

        # Second linear transformation
        output = self.linear2(hidden)  # (batch, seq, hidden_dim)

        # Apply residual connection based on type
        if self.residual_type == "Pre-LN":
            # Pre-LN: Add residual connection
            final_output = residual + output
        elif self.residual_type == "Post-LN":
            # Post-LN: Add residual connection, then normalize if provided
            final_output = residual + output
            if norm_module is not None:
                final_output = norm_module(final_output)
        else:  # Sandwich or other
            final_output = residual + output

        # Collect statistics
        stats = self._compute_stats(x, hidden, output, final_output)

        return final_output, stats

    def _compute_stats(
        self,
        input_tensor: torch.Tensor,
        hidden: torch.Tensor,
        output: torch.Tensor,
        final_output: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute feed-forward statistics."""
        with torch.no_grad():
            stats = {
                "input_mean": input_tensor.mean().item(),
                "input_std": input_tensor.std().item(),
                "hidden_mean": hidden.mean().item(),
                "hidden_std": hidden.std().item(),
                "activated_mean": hidden.mean().item(),  # Same as hidden after activation
                "activated_std": hidden.std().item(),
                "output_mean": output.mean().item(),
                "output_std": output.std().item(),
                "final_output_mean": final_output.mean().item(),
                "final_output_std": final_output.std().item(),
                "activation_type": self.activation_type,
                "residual_type": self.residual_type,
            }

        return stats

    def backward(
        self, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Backward pass (handled automatically by PyTorch)."""
        # This is handled automatically by PyTorch's autograd
        # We maintain the interface for compatibility

        batch_size, seq_len, hidden_dim = grad_output.size()
        grad_input = torch.zeros_like(grad_output)

        gradients = {
            "linear1.weight": self.linear1.weight.grad
            if self.linear1.weight.grad is not None
            else torch.zeros_like(self.linear1.weight),
            "linear1.bias": self.linear1.bias.grad
            if self.linear1.bias.grad is not None
            else torch.zeros_like(self.linear1.bias),
            "linear2.weight": self.linear2.weight.grad
            if self.linear2.weight.grad is not None
            else torch.zeros_like(self.linear2.weight),
            "linear2.bias": self.linear2.bias.grad
            if self.linear2.bias.grad is not None
            else torch.zeros_like(self.linear2.bias),
        }

        # Convert to numpy for compatibility
        gradients = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in gradients.items()
        }

        return grad_input.detach().cpu().numpy(), gradients

    def get_parameters(self) -> list[Any]:
        """Get trainable parameters."""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy())
        return params

    def get_pytorch_parameters(self) -> list[torch.nn.Parameter]:
        """Get PyTorch parameters for optimizer."""
        return list(self.parameters())
