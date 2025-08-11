"""
Pure Python implementation of normalization layers.

Shows explicit computation of mean, variance, and normalization
using only built-in Python operations.
"""

import math
from typing import List, Tuple, Dict, Any

from ..abstract import AbstractNormalization
from .utils import zeros, get_shape, copy_tensor


class PythonLayerNorm(AbstractNormalization):
    """Pure Python Layer Normalization implementation."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__(hidden_dim, eps)
        
        # Initialize scale and shift parameters
        self.gamma = [1.0] * hidden_dim  # Scale parameter
        self.beta = [0.0] * hidden_dim   # Shift parameter
        
        # Cache for backward pass
        self._cache = {}
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """Forward pass through layer normalization with explicit steps."""
        batch_size, seq_len, hidden_dim = get_shape(x)
        
        print(f"[PythonLayerNorm] Forward pass: shape={get_shape(x)}")
        
        result = zeros((batch_size, seq_len, hidden_dim))
        means = []
        variances = []
        
        for batch in range(batch_size):
            batch_means = []
            batch_variances = []
            
            for seq in range(seq_len):
                print(f"  Normalizing batch={batch}, seq={seq}")
                
                # Step 1: Compute mean
                mean_val = sum(x[batch][seq]) / hidden_dim
                batch_means.append(mean_val)
                print(f"    Step 1: Mean = {mean_val:.6f}")
                
                # Step 2: Compute variance
                variance = sum((val - mean_val)**2 for val in x[batch][seq]) / hidden_dim
                batch_variances.append(variance)
                print(f"    Step 2: Variance = {variance:.6f}")
                
                # Step 3: Compute standard deviation (with epsilon for stability)
                std = math.sqrt(variance + self.eps)
                print(f"    Step 3: Std (with eps={self.eps}) = {std:.6f}")
                
                # Step 4: Normalize, scale, and shift
                for dim in range(hidden_dim):
                    # Normalize
                    normalized = (x[batch][seq][dim] - mean_val) / std
                    
                    # Apply learnable parameters
                    result[batch][seq][dim] = self.gamma[dim] * normalized + self.beta[dim]
                
                print(f"    Step 4: Applied gamma (scale) and beta (shift)")
            
            means.append(batch_means)
            variances.append(batch_variances)
        
        # Cache intermediate values for backward pass
        self._cache = {
            'input': copy_tensor(x),
            'means': means,
            'variances': variances,
            'normalized': copy_tensor(result)  # This would be before gamma/beta in full impl
        }
        
        print(f"  Final normalized output shape: {get_shape(result)}")
        return result
    
    def backward(self, grad_output: List[List[List[float]]]) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
        """Backward pass (simplified for educational purposes)."""
        if not self._cache:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        print("[PythonLayerNorm] Backward pass (simplified)...")
        
        input_shape = get_shape(self._cache['input'])
        grad_input = zeros(input_shape)
        
        gradients = {
            'gamma': [0.0] * len(self.gamma),
            'beta': [0.0] * len(self.beta),
        }
        
        return grad_input, gradients
    
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        return [self.gamma, self.beta]


class PythonRMSNorm(AbstractNormalization):
    """Pure Python RMS Normalization implementation."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__(hidden_dim, eps)
        
        # Initialize scale parameter (no bias in RMSNorm)
        self.gamma = [1.0] * hidden_dim
        
        # Cache for backward pass
        self._cache = {}
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """Forward pass through RMS normalization."""
        batch_size, seq_len, hidden_dim = get_shape(x)
        
        print(f"[PythonRMSNorm] Forward pass: shape={get_shape(x)}")
        
        result = zeros((batch_size, seq_len, hidden_dim))
        rms_values = []
        
        for batch in range(batch_size):
            batch_rms = []
            
            for seq in range(seq_len):
                print(f"  RMS normalizing batch={batch}, seq={seq}")
                
                # Step 1: Compute mean of squares
                mean_square = sum(val**2 for val in x[batch][seq]) / hidden_dim
                batch_rms.append(mean_square)
                print(f"    Step 1: Mean of squares = {mean_square:.6f}")
                
                # Step 2: Compute RMS (root mean square)
                rms = math.sqrt(mean_square + self.eps)
                print(f"    Step 2: RMS (with eps={self.eps}) = {rms:.6f}")
                
                # Step 3: Normalize and scale
                for dim in range(hidden_dim):
                    normalized = x[batch][seq][dim] / rms
                    result[batch][seq][dim] = self.gamma[dim] * normalized
                
                print(f"    Step 3: Applied RMS normalization and gamma scaling")
            
            rms_values.append(batch_rms)
        
        # Cache for backward pass
        self._cache = {
            'input': copy_tensor(x),
            'rms_values': rms_values,
        }
        
        print(f"  Final RMS normalized output shape: {get_shape(result)}")
        return result
    
    def backward(self, grad_output: List[List[List[float]]]) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
        """Backward pass (simplified for educational purposes)."""
        if not self._cache:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        print("[PythonRMSNorm] Backward pass (simplified)...")
        
        input_shape = get_shape(self._cache['input'])
        grad_input = zeros(input_shape)
        
        gradients = {
            'gamma': [0.0] * len(self.gamma),
        }
        
        return grad_input, gradients
    
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        return [self.gamma]


def create_normalization(norm_type: str, hidden_dim: int) -> AbstractNormalization:
    """Factory function for creating normalization layers."""
    if norm_type == "LayerNorm":
        return PythonLayerNorm(hidden_dim)
    elif norm_type == "RMSNorm":
        return PythonRMSNorm(hidden_dim)
    elif norm_type == "None":
        return None
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")