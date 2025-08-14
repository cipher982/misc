"""
Activation functions for the Transformer Intuition Lab.
Pure NumPy implementations for educational purposes.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation (SiLU): x * sigmoid(x)."""
    return x * (1 / (1 + np.exp(-x)))


def swiglu(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """SwiGLU activation: Swish(x) * gate."""
    return swish(x) * gate


class ActivationModule:
    """Activation function module with statistics tracking."""

    def __init__(self, activation_type: str = "ReLU"):
        self.activation_type = activation_type
        self.activation_fn = self._get_activation_fn(activation_type)

    def _get_activation_fn(self, activation_type: str):
        """Get activation function by name."""
        if activation_type == "ReLU":
            return relu
        if activation_type == "GeLU":
            return gelu
        if activation_type == "Swish":
            return swish
        if activation_type == "SwiGLU":
            return swiglu
        raise ValueError(f"Unknown activation type: {activation_type}")

    def forward(self, x: np.ndarray, gate: np.ndarray = None) -> np.ndarray:
        """Forward pass through activation function."""
        if self.activation_type == "SwiGLU":
            if gate is None:
                raise ValueError("SwiGLU requires a gate input")
            return self.activation_fn(x, gate)
        return self.activation_fn(x)

    def get_stats(self, x: np.ndarray, gate: np.ndarray = None) -> dict:
        """Get activation statistics for visualization."""
        output = self.forward(x, gate)

        stats = {
            "input_mean": np.mean(x),
            "input_std": np.std(x),
            "input_min": np.min(x),
            "input_max": np.max(x),
            "output_mean": np.mean(output),
            "output_std": np.std(output),
            "output_min": np.min(output),
            "output_max": np.max(output),
            "dead_neurons": (
                np.sum(output == 0) / output.size
                if self.activation_type == "ReLU"
                else 0
            ),
        }

        if gate is not None:
            stats.update({"gate_mean": np.mean(gate), "gate_std": np.std(gate)})

        return stats


def get_activation_module(activation_type: str) -> ActivationModule:
    """Factory function to create activation modules."""
    return ActivationModule(activation_type)
