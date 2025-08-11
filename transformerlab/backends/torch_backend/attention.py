"""
PyTorch implementation of multi-head attention.

This implementation uses PyTorch's native operations and shows modern
ML practices like proper initialization, efficient computation, and
automatic differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, Any, List

from ..abstract import AbstractAttention


class TorchAttention(AbstractAttention, nn.Module):
    """PyTorch multi-head attention implementation."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        # Initialize both parent classes
        AbstractAttention.__init__(self, hidden_dim, num_heads, dropout)
        nn.Module.__init__(self)
        
        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Dropout for attention weights
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Cache for attention weights visualization
        self._attention_weights = None
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through multi-head attention."""
        batch_size, seq_len, hidden_dim = x.size()
        
        # Linear transformations for Q, K, V
        q = self.q_proj(x)  # (batch, seq, hidden)
        k = self.k_proj(x)  # (batch, seq, hidden)
        v = self.v_proj(x)  # (batch, seq, hidden)
        
        # Reshape for multi-head attention: (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self._scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        # Final output projection
        output = self.out_proj(attention_output)
        
        # Cache attention weights for visualization
        self._attention_weights = attention_weights.detach().cpu()
        
        # Collect statistics
        stats = self._compute_stats(q, k, v, attention_weights, output)
        
        return output, stats
    
    def _scaled_dot_product_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention efficiently."""
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Broadcast mask to (batch, num_heads, seq, seq)
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout if specified
        if self.dropout_layer is not None and self.training:
            attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def _compute_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_weights: torch.Tensor,
        output: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute attention statistics."""
        with torch.no_grad():
            stats = {
                "q_mean": q.mean().item(),
                "q_std": q.std().item(),
                "k_mean": k.mean().item(),
                "k_std": k.std().item(),
                "v_mean": v.mean().item(),
                "v_std": v.std().item(),
                "attention_weights_mean": attention_weights.mean().item(),
                "attention_weights_std": attention_weights.std().item(),
                "attention_weights_max": attention_weights.max().item(),
                "attention_weights_min": attention_weights.min().item(),
                "output_mean": output.mean().item(),
                "output_std": output.std().item(),
                "attention_weights": attention_weights[0].cpu().numpy(),  # First batch for viz
            }
        
        return stats
    
    def backward(self, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Backward pass (handled automatically by PyTorch)."""
        # In PyTorch, backward pass is automatic, but we need to maintain
        # the interface for compatibility
        
        # This would typically not be called directly in PyTorch
        # Instead, gradients are computed automatically via autograd
        
        # For compatibility, we return dummy values
        batch_size, seq_len, hidden_dim = grad_output.size()
        grad_input = torch.zeros_like(grad_output)
        
        gradients = {
            'q_proj.weight': self.q_proj.weight.grad if self.q_proj.weight.grad is not None else torch.zeros_like(self.q_proj.weight),
            'k_proj.weight': self.k_proj.weight.grad if self.k_proj.weight.grad is not None else torch.zeros_like(self.k_proj.weight),
            'v_proj.weight': self.v_proj.weight.grad if self.v_proj.weight.grad is not None else torch.zeros_like(self.v_proj.weight),
            'out_proj.weight': self.out_proj.weight.grad if self.out_proj.weight.grad is not None else torch.zeros_like(self.out_proj.weight),
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
    
    def get_attention_weights(self) -> Any:
        """Get attention weights for visualization."""
        if self._attention_weights is not None:
            return self._attention_weights.numpy()
        return None
    
    def get_pytorch_parameters(self) -> List[torch.nn.Parameter]:
        """Get PyTorch parameters for optimizer."""
        return list(self.parameters())