"""
PyTorch implementation of normalization layers.

Shows production-ready implementations with proper initialization
and efficient computation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List

from ..abstract import AbstractNormalization


class TorchLayerNorm(AbstractNormalization, nn.Module):
    """PyTorch Layer Normalization implementation."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        # Initialize both parent classes
        AbstractNormalization.__init__(self, hidden_dim, eps)
        nn.Module.__init__(self)
        
        # Use PyTorch's built-in LayerNorm for efficiency
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer normalization."""
        return self.layer_norm(x)
    
    def backward(self, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Backward pass (handled automatically by PyTorch)."""
        # This is handled automatically by PyTorch
        grad_input = torch.zeros_like(grad_output)
        
        gradients = {
            'weight': self.layer_norm.weight.grad if self.layer_norm.weight.grad is not None else torch.zeros_like(self.layer_norm.weight),
            'bias': self.layer_norm.bias.grad if self.layer_norm.bias.grad is not None else torch.zeros_like(self.layer_norm.bias),
        }
        
        # Convert to numpy for compatibility
        gradients = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                     for k, v in gradients.items()}
        
        return grad_input.detach().cpu().numpy(), gradients
    
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy())
        return params
    
    def get_pytorch_parameters(self) -> List[torch.nn.Parameter]:
        """Get PyTorch parameters for optimizer."""
        return list(self.parameters())


class TorchRMSNorm(AbstractNormalization, nn.Module):
    """PyTorch RMS Normalization implementation."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        # Initialize both parent classes
        AbstractNormalization.__init__(self, hidden_dim, eps)
        nn.Module.__init__(self)
        
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        
        # Register eps as buffer so it's saved with the model
        self.register_buffer('eps', torch.tensor(eps))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RMS normalization."""
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x / rms * self.weight
    
    def backward(self, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Backward pass (handled automatically by PyTorch)."""
        # This is handled automatically by PyTorch
        grad_input = torch.zeros_like(grad_output)
        
        gradients = {
            'weight': self.weight.grad if self.weight.grad is not None else torch.zeros_like(self.weight),
        }
        
        # Convert to numpy for compatibility
        gradients = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                     for k, v in gradients.items()}
        
        return grad_input.detach().cpu().numpy(), gradients
    
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy())
        return params
    
    def get_pytorch_parameters(self) -> List[torch.nn.Parameter]:
        """Get PyTorch parameters for optimizer."""
        return list(self.parameters())


def create_torch_normalization(norm_type: str, hidden_dim: int) -> torch.nn.Module:
    """Factory function for creating PyTorch normalization layers."""
    if norm_type == "LayerNorm":
        return TorchLayerNorm(hidden_dim)
    elif norm_type == "RMSNorm":
        return TorchRMSNorm(hidden_dim)
    elif norm_type == "None":
        return nn.Identity()  # No normalization
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")