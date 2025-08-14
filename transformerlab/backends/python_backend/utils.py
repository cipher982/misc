"""
Pure Python utility functions for mathematical operations.

These functions replace NumPy operations with explicit Python loops
for maximum educational transparency.
"""

import math
from typing import Any


def zeros(shape: tuple[int, ...]) -> list[Any]:
    """Create nested lists filled with zeros."""
    if len(shape) == 1:
        return [0.0] * shape[0]
    if len(shape) == 2:
        return [[0.0] * shape[1] for _ in range(shape[0])]
    if len(shape) == 3:
        return [[[0.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    raise NotImplementedError("Only support up to 3D tensors")


def randn(shape: tuple[int, ...], scale: float = 0.02) -> list[Any]:
    """Create nested lists with random normal values."""
    import random

    if len(shape) == 1:
        return [random.gauss(0, scale) for _ in range(shape[0])]
    if len(shape) == 2:
        return [
            [random.gauss(0, scale) for _ in range(shape[1])] for _ in range(shape[0])
        ]
    if len(shape) == 3:
        return [
            [[random.gauss(0, scale) for _ in range(shape[2])] for _ in range(shape[1])]
            for _ in range(shape[0])
        ]
    raise NotImplementedError("Only support up to 3D tensors")


def get_shape(tensor: list[Any]) -> tuple[int, ...]:
    """Get shape of nested list tensor."""
    if not isinstance(tensor, list):
        return ()

    shape = [len(tensor)]
    current = tensor

    while isinstance(current[0], list) and len(current) > 0:
        shape.append(len(current[0]))
        current = current[0]

    return tuple(shape)


def matmul_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Matrix multiplication for 2D lists."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Cannot multiply matrices of shape ({rows_a}, {cols_a}) and ({rows_b}, {cols_b})"
        )

    result = zeros((rows_a, cols_b))

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]

    return result


def matmul_3d(
    a: list[list[list[float]]], b: list[list[float]]
) -> list[list[list[float]]]:
    """Batch matrix multiplication: (batch, seq, dim1) @ (dim1, dim2) -> (batch, seq, dim2)."""
    batch_size = len(a)
    seq_len = len(a[0])
    dim2 = len(b[0])

    result = zeros((batch_size, seq_len, dim2))

    for batch in range(batch_size):
        for seq in range(seq_len):
            for out_dim in range(dim2):
                for in_dim in range(len(b)):
                    result[batch][seq][out_dim] += (
                        a[batch][seq][in_dim] * b[in_dim][out_dim]
                    )

    return result


def add_3d(a: list[list[list[float]]], b: list[float]) -> list[list[list[float]]]:
    """Add bias vector to 3D tensor: (batch, seq, dim) + (dim,) -> (batch, seq, dim)."""
    batch_size = len(a)
    seq_len = len(a[0])
    dim = len(a[0][0])

    result = zeros((batch_size, seq_len, dim))

    for batch in range(batch_size):
        for seq in range(seq_len):
            for d in range(dim):
                result[batch][seq][d] = a[batch][seq][d] + b[d]

    return result


def add_tensors(a: list[Any], b: list[Any]) -> list[Any]:
    """Element-wise addition of two tensors of the same shape."""
    if not isinstance(a[0], list):
        # 1D case
        return [a[i] + b[i] for i in range(len(a))]
    if not isinstance(a[0][0], list):
        # 2D case
        return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]
    # 3D case
    return [
        [
            [a[i][j][k] + b[i][j][k] for k in range(len(a[i][j]))]
            for j in range(len(a[i]))
        ]
        for i in range(len(a))
    ]


def softmax_2d(logits: list[list[float]]) -> list[list[float]]:
    """Softmax activation for 2D tensor."""
    result = []

    for row in logits:
        # Find max for numerical stability
        max_val = max(row)

        # Compute exponentials
        exp_vals = [math.exp(x - max_val) for x in row]

        # Compute sum
        exp_sum = sum(exp_vals)

        # Normalize
        softmax_row = [exp_val / exp_sum for exp_val in exp_vals]
        result.append(softmax_row)

    return result


def relu(x: list[Any]) -> list[Any]:
    """ReLU activation function."""
    if not isinstance(x[0], list):
        # 1D case
        return [max(0.0, val) for val in x]
    if not isinstance(x[0][0], list):
        # 2D case
        return [[max(0.0, x[i][j]) for j in range(len(x[i]))] for i in range(len(x))]
    # 3D case
    return [
        [[max(0.0, x[i][j][k]) for k in range(len(x[i][j]))] for j in range(len(x[i]))]
        for i in range(len(x))
    ]


def gelu(x: list[Any]) -> list[Any]:
    """GELU activation function (approximate)."""

    def gelu_single(val):
        return (
            0.5
            * val
            * (1 + math.tanh(math.sqrt(2 / math.pi) * (val + 0.044715 * val**3)))
        )

    if not isinstance(x[0], list):
        # 1D case
        return [gelu_single(val) for val in x]
    if not isinstance(x[0][0], list):
        # 2D case
        return [[gelu_single(x[i][j]) for j in range(len(x[i]))] for i in range(len(x))]
    # 3D case
    return [
        [
            [gelu_single(x[i][j][k]) for k in range(len(x[i][j]))]
            for j in range(len(x[i]))
        ]
        for i in range(len(x))
    ]


def swish(x: list[Any]) -> list[Any]:
    """Swish activation function."""

    def swish_single(val):
        return val / (1 + math.exp(-val))

    if not isinstance(x[0], list):
        # 1D case
        return [swish_single(val) for val in x]
    if not isinstance(x[0][0], list):
        # 2D case
        return [
            [swish_single(x[i][j]) for j in range(len(x[i]))] for i in range(len(x))
        ]
    # 3D case
    return [
        [
            [swish_single(x[i][j][k]) for k in range(len(x[i][j]))]
            for j in range(len(x[i]))
        ]
        for i in range(len(x))
    ]


def layer_norm(
    x: list[list[list[float]]], eps: float = 1e-6
) -> list[list[list[float]]]:
    """Layer normalization for 3D tensor."""
    batch_size = len(x)
    seq_len = len(x[0])
    hidden_dim = len(x[0][0])

    result = zeros((batch_size, seq_len, hidden_dim))

    for batch in range(batch_size):
        for seq in range(seq_len):
            # Compute mean
            mean = sum(x[batch][seq]) / hidden_dim

            # Compute variance
            variance = sum((val - mean) ** 2 for val in x[batch][seq]) / hidden_dim

            # Normalize
            std = math.sqrt(variance + eps)
            for dim in range(hidden_dim):
                result[batch][seq][dim] = (x[batch][seq][dim] - mean) / std

    return result


def transpose_2d(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose a 2D matrix."""
    rows = len(matrix)
    cols = len(matrix[0])

    result = zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]

    return result


def mean(tensor: list[Any]) -> float:
    """Calculate mean of all elements in tensor."""

    def flatten(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result

    flat = flatten(tensor)
    return sum(flat) / len(flat)


def std(tensor: list[Any]) -> float:
    """Calculate standard deviation of all elements in tensor."""

    def flatten(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result

    flat = flatten(tensor)
    mean_val = sum(flat) / len(flat)
    variance = sum((x - mean_val) ** 2 for x in flat) / len(flat)
    return math.sqrt(variance)


def copy_tensor(tensor: list[Any]) -> list[Any]:
    """Deep copy a tensor."""
    if not isinstance(tensor, list):
        return tensor
    return [copy_tensor(item) for item in tensor]
