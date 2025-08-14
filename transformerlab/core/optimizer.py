"""
Simple optimizers for the Transformer Intuition Lab.
Pure NumPy implementation for educational purposes.
"""

import numpy as np


class SGDOptimizer:
    """Simple Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate

    def update_parameters(
        self, parameters: list[np.ndarray], gradients: list[np.ndarray]
    ) -> None:
        """
        Update parameters using gradients.

        Args:
            parameters: List of parameter arrays to update
            gradients: List of gradient arrays corresponding to parameters
        """
        for param, grad in zip(parameters, gradients, strict=False):
            param -= self.learning_rate * grad


class AdamOptimizer:
    """Adam optimizer with momentum and bias correction."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Learning rate for parameter updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant to prevent division by zero
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # time step
        self.m = {}  # first moment estimates
        self.v = {}  # second moment estimates

    def update_parameters(
        self, parameters: list[np.ndarray], gradients: list[np.ndarray]
    ) -> None:
        """
        Update parameters using Adam algorithm.

        Args:
            parameters: List of parameter arrays to update
            gradients: List of gradient arrays corresponding to parameters
        """
        self.t += 1

        for i, (param, grad) in enumerate(zip(parameters, gradients, strict=False)):
            # Initialize moments if first time
            if i not in self.m:
                self.m[i] = np.zeros_like(grad)
                self.v[i] = np.zeros_like(grad)

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def get_optimizer(optimizer_type: str, **kwargs) -> object:
    """
    Get optimizer instance by type.

    Args:
        optimizer_type: Type of optimizer ('SGD' or 'Adam')
        **kwargs: Additional arguments for optimizer initialization

    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "sgd":
        return SGDOptimizer(**kwargs)
    if optimizer_type.lower() == "adam":
        return AdamOptimizer(**kwargs)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
