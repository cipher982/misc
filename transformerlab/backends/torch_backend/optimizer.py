"""
PyTorch implementation of optimizers.

Wraps PyTorch's native optimizers to provide the abstract interface
while leveraging production-ready implementations.
"""

from typing import Any

import torch
import torch.optim as optim

from ..abstract import AbstractOptimizer


class TorchSGDOptimizer(AbstractOptimizer):
    """PyTorch SGD optimizer wrapper."""

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)

        # Create PyTorch SGD optimizer
        self.optimizer = optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.momentum = momentum
        self.weight_decay = weight_decay

    def update_parameters(self, parameters: list[Any], gradients: list[Any]) -> None:
        """Update parameters using PyTorch optimizer."""
        # In PyTorch, we don't manually update parameters
        # Instead, we call optimizer.step() after loss.backward()
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Perform optimization step."""
        self.optimizer.step()

    def get_state_dict(self) -> dict:
        """Get optimizer state dictionary."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state dictionary."""
        self.optimizer.load_state_dict(state_dict)


class TorchAdamOptimizer(AbstractOptimizer):
    """PyTorch Adam optimizer wrapper."""

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)

        # Create PyTorch Adam optimizer
        self.optimizer = optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def update_parameters(self, parameters: list[Any], gradients: list[Any]) -> None:
        """Update parameters using PyTorch optimizer."""
        # In PyTorch, we don't manually update parameters
        # Instead, we call optimizer.step() after loss.backward()
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Perform optimization step."""
        self.optimizer.step()

    def get_state_dict(self) -> dict:
        """Get optimizer state dictionary."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state dictionary."""
        self.optimizer.load_state_dict(state_dict)


class TorchAdamWOptimizer(AbstractOptimizer):
    """PyTorch AdamW optimizer wrapper."""

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(learning_rate)

        # Create PyTorch AdamW optimizer
        self.optimizer = optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def update_parameters(self, parameters: list[Any], gradients: list[Any]) -> None:
        """Update parameters using PyTorch optimizer."""
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Perform optimization step."""
        self.optimizer.step()

    def get_state_dict(self) -> dict:
        """Get optimizer state dictionary."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state dictionary."""
        self.optimizer.load_state_dict(state_dict)


def create_torch_optimizer(
    optimizer_type: str,
    parameters: list[torch.nn.Parameter],
    **kwargs
) -> AbstractOptimizer:
    """Factory function for creating PyTorch optimizers."""
    if optimizer_type.lower() == "sgd":
        return TorchSGDOptimizer(parameters, **kwargs)
    elif optimizer_type.lower() == "adam":
        return TorchAdamOptimizer(parameters, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return TorchAdamWOptimizer(parameters, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
