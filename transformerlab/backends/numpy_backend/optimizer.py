"""
NumPy implementation of optimizers.

Provides SGD, Adam, and AdamW optimizers using pure NumPy operations.
"""

import numpy as np
from typing import List, Any

from ..abstract import AbstractOptimizer


class NumPySGDOptimizer(AbstractOptimizer):
    """NumPy SGD optimizer implementation."""
    
    def __init__(
        self, 
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Momentum buffers (initialized lazily)
        self.velocity_buffers = None
        self.step_count = 0
    
    def update_parameters(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """Update parameters using SGD with optional momentum."""
        if self.velocity_buffers is None:
            # Initialize momentum buffers
            self.velocity_buffers = [np.zeros_like(p) for p in parameters]
        
        self.step_count += 1
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # Apply weight decay if specified
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Update velocity (momentum)
            if self.momentum > 0:
                self.velocity_buffers[i] = self.momentum * self.velocity_buffers[i] + grad
                update = self.velocity_buffers[i]
            else:
                update = grad
            
            # Update parameters
            param -= self.learning_rate * update
    
    def zero_grad(self) -> None:
        """Zero out gradients (no-op for NumPy implementation)."""
        # In NumPy, gradients are computed fresh each time, so no need to zero them
        pass


class NumPyAdamOptimizer(AbstractOptimizer):
    """NumPy Adam optimizer implementation."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Moment buffers (initialized lazily)
        self.m_buffers = None  # First moment
        self.v_buffers = None  # Second moment
        self.step_count = 0
    
    def update_parameters(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """Update parameters using Adam optimizer."""
        if self.m_buffers is None:
            # Initialize moment buffers
            self.m_buffers = [np.zeros_like(p) for p in parameters]
            self.v_buffers = [np.zeros_like(p) for p in parameters]
        
        self.step_count += 1
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # Apply weight decay if specified
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Update biased first moment estimate
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m_buffers[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v_buffers[i] / (1 - self.beta2 ** self.step_count)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self) -> None:
        """Zero out gradients (no-op for NumPy implementation)."""
        # In NumPy, gradients are computed fresh each time, so no need to zero them
        pass


class NumPyAdamWOptimizer(AbstractOptimizer):
    """NumPy AdamW optimizer implementation."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Moment buffers (initialized lazily)
        self.m_buffers = None  # First moment
        self.v_buffers = None  # Second moment
        self.step_count = 0
    
    def update_parameters(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """Update parameters using AdamW optimizer."""
        if self.m_buffers is None:
            # Initialize moment buffers
            self.m_buffers = [np.zeros_like(p) for p in parameters]
            self.v_buffers = [np.zeros_like(p) for p in parameters]
        
        self.step_count += 1
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # Update biased first moment estimate (without weight decay)
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate (without weight decay)
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m_buffers[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v_buffers[i] / (1 - self.beta2 ** self.step_count)
            
            # Update parameters with decoupled weight decay
            param -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param)
    
    def zero_grad(self) -> None:
        """Zero out gradients (no-op for NumPy implementation)."""
        # In NumPy, gradients are computed fresh each time, so no need to zero them
        pass


def create_numpy_optimizer(
    optimizer_type: str, 
    learning_rate: float = 0.01,
    **kwargs
) -> AbstractOptimizer:
    """Factory function for creating NumPy optimizers."""
    if optimizer_type.lower() == "sgd":
        return NumPySGDOptimizer(learning_rate, **kwargs)
    elif optimizer_type.lower() == "adam":
        return NumPyAdamOptimizer(learning_rate, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return NumPyAdamWOptimizer(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")