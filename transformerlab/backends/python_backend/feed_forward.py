"""
Pure Python implementation of feed-forward network.

Shows explicit matrix operations and activation functions
using only built-in Python operations.
"""

from typing import List, Tuple, Dict, Optional, Any

from ..abstract import AbstractFeedForward
from .utils import (
    zeros, randn, matmul_3d, add_3d, add_tensors, relu, gelu, swish,
    copy_tensor, get_shape
)


class PythonFeedForward(AbstractFeedForward):
    """Pure Python feed-forward network implementation."""
    
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
    ):
        super().__init__(hidden_dim, ff_dim, activation_type, residual_type)
        
        # Initialize weight matrices
        self.w1 = randn((hidden_dim, ff_dim))  # First linear layer
        self.w2 = randn((ff_dim, hidden_dim))  # Second linear layer
        
        # Initialize biases
        self.b1 = [0.0] * ff_dim
        self.b2 = [0.0] * hidden_dim
        
        # Cache for backward pass
        self._cache = {}
    
    def forward(
        self, 
        x: List[List[List[float]]], 
        norm_module: Optional[Any] = None
    ) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
        """Forward pass through feed-forward network with explicit steps."""
        batch_size, seq_len, hidden_dim = get_shape(x)
        
        print(f"[PythonFeedForward] Forward pass: shape={get_shape(x)}")
        
        # Step 1: First linear transformation: x @ w1 + b1
        print("  Step 1: First linear transformation (hidden -> ff_dim)...")
        h = add_3d(matmul_3d(x, self.w1), self.b1)  # (batch, seq, ff_dim)
        print(f"    Transformed from {hidden_dim} to {self.ff_dim} dimensions")
        
        # Step 2: Apply activation function
        print(f"  Step 2: Applying {self.activation_type} activation...")
        if self.activation_type == "ReLU":
            h_activated = relu(h)
        elif self.activation_type == "GeLU":
            h_activated = gelu(h)
        elif self.activation_type == "Swish":
            h_activated = swish(h)
        else:
            h_activated = h  # Linear activation
        
        # Step 3: Second linear transformation: h_activated @ w2 + b2
        print("  Step 3: Second linear transformation (ff_dim -> hidden)...")
        output = add_3d(matmul_3d(h_activated, self.w2), self.b2)  # (batch, seq, hidden)
        print(f"    Transformed back from {self.ff_dim} to {hidden_dim} dimensions")
        
        # Step 4: Apply residual connection
        print(f"  Step 4: Applying {self.residual_type} residual connection...")
        if self.residual_type == "Pre-LN":
            # Pre-LN: residual connection is applied after the FFN
            final_output = add_tensors(x, output)
        elif self.residual_type == "Post-LN":
            # Post-LN: residual connection is applied before normalization
            final_output = add_tensors(x, output)
            if norm_module is not None:
                print("    Applying normalization after residual...")
                final_output = norm_module.forward(final_output)
        else:  # Sandwich
            # Sandwich: residual connection is applied in the middle
            final_output = add_tensors(x, output)
        
        print(f"    Final output shape: {get_shape(final_output)}")
        
        # Cache intermediate values for backward pass
        self._cache = {
            'input': copy_tensor(x),
            'h': copy_tensor(h),
            'h_activated': copy_tensor(h_activated),
            'output': copy_tensor(output),
            'norm_module': norm_module
        }
        
        # Collect statistics
        stats = self._compute_stats(x, h, h_activated, output, final_output)
        
        return final_output, stats
    
    def _compute_stats(
        self,
        x: List[List[List[float]]],
        h: List[List[List[float]]],
        h_activated: List[List[List[float]]],
        output: List[List[List[float]]],
        final_output: List[List[List[float]]],
    ) -> Dict[str, Any]:
        """Compute feed-forward statistics."""
        from .utils import mean, std
        
        return {
            "input_mean": mean(x),
            "input_std": std(x),
            "hidden_mean": mean(h),
            "hidden_std": std(h),
            "activated_mean": mean(h_activated),
            "activated_std": std(h_activated),
            "output_mean": mean(output),
            "output_std": std(output),
            "final_output_mean": mean(final_output),
            "final_output_std": std(final_output),
            "activation_type": self.activation_type,
            "residual_type": self.residual_type,
        }
    
    def backward(self, grad_output: List[List[List[float]]]) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
        """Backward pass (simplified for educational purposes)."""
        if not self._cache:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        print("[PythonFeedForward] Backward pass (simplified)...")
        
        # For educational purposes, we'll implement simplified gradients
        # In a full implementation, this would compute actual gradients
        
        input_shape = get_shape(self._cache['input'])
        grad_input = zeros(input_shape)
        
        gradients = {
            'w1': zeros(get_shape(self.w1)),
            'w2': zeros(get_shape(self.w2)),
            'b1': [0.0] * len(self.b1),
            'b2': [0.0] * len(self.b2),
        }
        
        return grad_input, gradients
    
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        return [self.w1, self.w2, self.b1, self.b2]