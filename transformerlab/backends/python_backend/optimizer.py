"""
Pure Python implementation of optimizers.

Shows explicit parameter updates using only built-in Python operations.
"""

from typing import List, Any

from ..abstract import AbstractOptimizer
from .utils import get_shape, zeros


class PythonSGDOptimizer(AbstractOptimizer):
    """Pure Python SGD optimizer implementation."""
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
    
    def update_parameters(self, parameters: List[Any], gradients: List[Any]) -> None:
        """Update parameters using SGD with explicit operations."""
        print(f"[PythonSGDOptimizer] Updating {len(parameters)} parameter tensors")
        print(f"  Learning rate: {self.learning_rate}")
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            print(f"  Updating parameter {i} with shape {get_shape(param)}")
            
            # Update parameters: param = param - learning_rate * grad
            self._update_tensor(param, grad, self.learning_rate)
    
    def _update_tensor(self, param: Any, grad: Any, lr: float) -> None:
        """Update a single tensor parameter."""
        param_shape = get_shape(param)
        
        if len(param_shape) == 1:
            # 1D tensor (bias)
            for i in range(len(param)):
                param[i] = param[i] - lr * grad[i]
        elif len(param_shape) == 2:
            # 2D tensor (weight matrix)
            for i in range(len(param)):
                for j in range(len(param[i])):
                    param[i][j] = param[i][j] - lr * grad[i][j]
        elif len(param_shape) == 3:
            # 3D tensor (embedding matrix or similar)
            for i in range(len(param)):
                for j in range(len(param[i])):
                    for k in range(len(param[i][j])):
                        param[i][j][k] = param[i][j][k] - lr * grad[i][j][k]
        else:
            raise NotImplementedError(f"Parameter updates for {len(param_shape)}D tensors not implemented")
    
    def zero_grad(self) -> None:
        """Zero out gradients (no-op for Python backend)."""
        pass


class PythonAdamOptimizer(AbstractOptimizer):
    """Pure Python Adam optimizer implementation."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Time step
        
        # Moment estimates (will be initialized on first use)
        self.m = {}  # First moment
        self.v = {}  # Second moment
    
    def update_parameters(self, parameters: List[Any], gradients: List[Any]) -> None:
        """Update parameters using Adam algorithm."""
        self.t += 1
        
        print(f"[PythonAdamOptimizer] Adam update step {self.t}")
        print(f"  Learning rate: {self.learning_rate}, beta1: {self.beta1}, beta2: {self.beta2}")
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            print(f"  Updating parameter {i} with shape {get_shape(param)}")
            
            # Initialize moments if first time
            if i not in self.m:
                self.m[i] = self._create_zero_like(param)
                self.v[i] = self._create_zero_like(param)
            
            # Update moments and parameters
            self._adam_update_tensor(param, grad, self.m[i], self.v[i], i)
    
    def _create_zero_like(self, tensor: Any) -> Any:
        """Create zero tensor with same shape as input."""
        shape = get_shape(tensor)
        return zeros(shape)
    
    def _adam_update_tensor(self, param: Any, grad: Any, m: Any, v: Any, param_idx: int) -> None:
        """Apply Adam update to a single tensor."""
        param_shape = get_shape(param)
        
        if len(param_shape) == 1:
            # 1D tensor
            for i in range(len(param)):
                # Update biased first moment estimate
                m[i] = self.beta1 * m[i] + (1 - self.beta1) * grad[i]
                
                # Update biased second moment estimate
                v[i] = self.beta2 * v[i] + (1 - self.beta2) * (grad[i] ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = m[i] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second moment estimate
                v_hat = v[i] / (1 - self.beta2 ** self.t)
                
                # Update parameter
                param[i] = param[i] - self.learning_rate * m_hat / (v_hat ** 0.5 + self.eps)
        
        elif len(param_shape) == 2:
            # 2D tensor
            for i in range(len(param)):
                for j in range(len(param[i])):
                    # Update biased first moment estimate
                    m[i][j] = self.beta1 * m[i][j] + (1 - self.beta1) * grad[i][j]
                    
                    # Update biased second moment estimate
                    v[i][j] = self.beta2 * v[i][j] + (1 - self.beta2) * (grad[i][j] ** 2)
                    
                    # Compute bias-corrected first moment estimate
                    m_hat = m[i][j] / (1 - self.beta1 ** self.t)
                    
                    # Compute bias-corrected second moment estimate
                    v_hat = v[i][j] / (1 - self.beta2 ** self.t)
                    
                    # Update parameter
                    param[i][j] = param[i][j] - self.learning_rate * m_hat / (v_hat ** 0.5 + self.eps)
        
        else:
            raise NotImplementedError(f"Adam updates for {len(param_shape)}D tensors not implemented")
    
    def zero_grad(self) -> None:
        """Zero out gradients (no-op for Python backend)."""
        pass