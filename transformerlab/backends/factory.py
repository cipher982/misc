"""
Factory functions for creating backend implementations.

Provides a unified interface for instantiating transformers with different backends.
"""

from typing import Dict, Any, List, Type, Optional
import importlib

from .abstract import (
    AbstractTransformer,
    BackendType,
    BackendConfig,
)


# Registry of available backends
_BACKEND_REGISTRY: Dict[BackendType, Dict[str, Any]] = {
    BackendType.NUMPY: {
        "module": "transformerlab.backends.numpy_backend",
        "class": "NumPyTransformer",
        "description": "NumPy-based implementation with vectorized operations",
        "features": ["Fast vectorized computation", "Educational transparency", "No external ML frameworks"],
        "pros": ["Fast for CPU", "Easy to understand", "Minimal dependencies"],
        "cons": ["CPU only", "Manual gradient computation", "Limited scalability"],
        "best_for": "Learning transformer internals and fast CPU inference",
    },
    BackendType.PYTHON: {
        "module": "transformerlab.backends.python_backend", 
        "class": "PythonTransformer",
        "description": "Pure Python implementation with explicit loops and operations",
        "features": ["Maximum transparency", "Step-by-step execution", "No vectorization"],
        "pros": ["Complete algorithmic clarity", "Easy debugging", "No library dependencies"],
        "cons": ["Very slow", "High memory usage", "Not practical for training"],
        "best_for": "Understanding algorithms step-by-step and debugging",
    },
    BackendType.TORCH: {
        "module": "transformerlab.backends.torch_backend",
        "class": "TorchTransformer", 
        "description": "PyTorch-based implementation with automatic differentiation",
        "features": ["GPU acceleration", "Automatic gradients", "Production-ready"],
        "pros": ["Very fast", "GPU support", "Automatic differentiation", "Scalable"],
        "cons": ["Less transparent", "Large dependency", "Framework complexity"],
        "best_for": "Production training and GPU-accelerated inference",
    },
}


def list_backends() -> List[str]:
    """List all available backend names."""
    return [backend.value for backend in _BACKEND_REGISTRY.keys()]


def get_backend_info(backend_name: str) -> Dict[str, Any]:
    """Get detailed information about a backend."""
    try:
        backend_type = BackendType(backend_name)
        return _BACKEND_REGISTRY[backend_type].copy()
    except ValueError:
        available = list_backends()
        raise ValueError(f"Unknown backend '{backend_name}'. Available: {available}")


def create_transformer(
    backend_name: str,
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    max_seq_len: int = 512,
    norm_type: str = "LayerNorm",
    activation_type: str = "ReLU",
    residual_type: str = "Pre-LN",
    pos_encoding_type: str = "Sinusoidal",
    dropout: float = 0.0,
    device: str = "cpu",
    dtype: str = "float32",
    **kwargs
) -> AbstractTransformer:
    """Create a transformer model with the specified backend.
    
    Args:
        backend_name: Name of backend to use ('numpy', 'python', 'torch')
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        max_seq_len: Maximum sequence length
        norm_type: Normalization type ('LayerNorm', 'RMSNorm', 'None')
        activation_type: Activation function ('ReLU', 'GeLU', 'Swish')
        residual_type: Residual connection type ('Pre-LN', 'Post-LN', 'Sandwich')
        pos_encoding_type: Positional encoding type ('Sinusoidal', 'RoPE', 'ALiBi')
        dropout: Dropout probability
        device: Device to use ('cpu', 'cuda')
        dtype: Data type ('float32', 'float16')
        **kwargs: Additional backend-specific arguments
        
    Returns:
        AbstractTransformer instance
        
    Raises:
        ValueError: If backend is not available
        ImportError: If backend dependencies are missing
    """
    try:
        backend_type = BackendType(backend_name)
    except ValueError:
        available = list_backends()
        raise ValueError(f"Unknown backend '{backend_name}'. Available: {available}")
    
    backend_info = _BACKEND_REGISTRY[backend_type]
    
    # Import the backend module
    try:
        module = importlib.import_module(backend_info["module"])
        transformer_class = getattr(module, backend_info["class"])
    except ImportError as e:
        raise ImportError(
            f"Failed to import {backend_name} backend. "
            f"Make sure all dependencies are installed. Error: {e}"
        )
    except AttributeError as e:
        raise ImportError(
            f"Backend {backend_name} is malformed: {e}"
        )
    
    # Create backend configuration
    backend_config = BackendConfig(
        backend_type=backend_type,
        device=device,
        dtype=dtype,
        **kwargs
    )
    
    # Create and return transformer instance
    try:
        transformer = transformer_class(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            norm_type=norm_type,
            activation_type=activation_type,
            residual_type=residual_type,
            pos_encoding_type=pos_encoding_type,
            dropout=dropout,
            backend_config=backend_config,
            **kwargs
        )
        return transformer
    except Exception as e:
        raise RuntimeError(f"Failed to create {backend_name} transformer: {e}")


def compare_backends(
    backends: List[str],
    model_config: Dict[str, Any],
    include_details: bool = True
) -> Dict[str, Any]:
    """Compare multiple backends with the same configuration.
    
    Args:
        backends: List of backend names to compare
        model_config: Configuration dict for model creation
        include_details: Whether to include detailed backend information
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for backend_name in backends:
        try:
            # Get backend info
            backend_info = get_backend_info(backend_name)
            
            # Try to create transformer
            transformer = create_transformer(backend_name, **model_config)
            
            results[backend_name] = {
                "available": True,
                "parameter_count": transformer.get_parameter_count(),
                "backend_info": transformer.get_backend_info(),
            }
            
            if include_details:
                results[backend_name].update({
                    "description": backend_info["description"],
                    "features": backend_info["features"],
                    "pros": backend_info["pros"],
                    "cons": backend_info["cons"],
                    "best_for": backend_info["best_for"],
                })
                
        except Exception as e:
            results[backend_name] = {
                "available": False,
                "error": str(e),
            }
            
            if include_details:
                try:
                    backend_info = get_backend_info(backend_name)
                    results[backend_name].update({
                        "description": backend_info["description"],
                        "features": backend_info["features"],
                    })
                except:
                    pass
    
    return results


def get_recommended_backend(
    task: str = "learning",
    device: str = "cpu",
    model_size: str = "small"
) -> str:
    """Get recommended backend for specific use case.
    
    Args:
        task: Task type ('learning', 'training', 'inference', 'debugging')
        device: Target device ('cpu', 'cuda')
        model_size: Model size category ('small', 'medium', 'large')
        
    Returns:
        Recommended backend name
    """
    if task == "debugging" or task == "step-by-step":
        return "python"
    elif task == "learning" or (task == "training" and model_size == "small"):
        return "numpy"
    elif device == "cuda" or model_size in ["medium", "large"]:
        return "torch"
    else:
        return "numpy"