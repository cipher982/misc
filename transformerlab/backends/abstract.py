"""
Abstract base classes for transformer components.

Defines the common interface that all backend implementations must follow.
This ensures mathematical equivalence while allowing different implementation approaches.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Available backend types."""
    NUMPY = "numpy"
    PYTHON = "python"
    TORCH = "torch"


@dataclass
class BackendConfig:
    """Configuration for backend implementations."""
    backend_type: BackendType
    device: str = "cpu"
    dtype: str = "float32"
    optimize_memory: bool = False
    enable_profiling: bool = False


class AbstractTensor(ABC):
    """Abstract tensor interface for backend-agnostic operations."""
    
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        pass
    
    @abstractmethod
    def to_numpy(self) -> Any:
        """Convert to numpy array for visualization/comparison."""
        pass
    
    @abstractmethod
    def __add__(self, other):
        """Element-wise addition."""
        pass
    
    @abstractmethod
    def __matmul__(self, other):
        """Matrix multiplication."""
        pass


class AbstractOptimizer(ABC):
    """Abstract optimizer interface."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    @abstractmethod
    def update_parameters(self, parameters: List[Any], gradients: List[Any]) -> None:
        """Update parameters using gradients."""
        pass
    
    @abstractmethod
    def zero_grad(self) -> None:
        """Zero out gradients (if applicable to backend)."""
        pass


class AbstractNormalization(ABC):
    """Abstract normalization layer."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.eps = eps
    
    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through normalization."""
        pass
    
    @abstractmethod
    def backward(self, grad_output: Any) -> Tuple[Any, Dict[str, Any]]:
        """Backward pass through normalization."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        pass


class AbstractAttention(ABC):
    """Abstract multi-head attention layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
    
    @abstractmethod
    def forward(self, x: Any, mask: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Forward pass through attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_stats)
        """
        pass
    
    @abstractmethod
    def backward(self, grad_output: Any) -> Tuple[Any, Dict[str, Any]]:
        """Backward pass through attention."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> Any:
        """Get attention weights for visualization."""
        pass


class AbstractFeedForward(ABC):
    """Abstract feed-forward network."""
    
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
    ):
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.activation_type = activation_type
        self.residual_type = residual_type
    
    @abstractmethod
    def forward(self, x: Any, norm_module: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            norm_module: Optional normalization module
            
        Returns:
            Tuple of (output, ff_stats)
        """
        pass
    
    @abstractmethod
    def backward(self, grad_output: Any) -> Tuple[Any, Dict[str, Any]]:
        """Backward pass through feed-forward network."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        pass


class AbstractTransformerBlock(ABC):
    """Abstract transformer block."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        norm_type: str = "LayerNorm",
        activation_type: str = "ReLU",
        residual_type: str = "Pre-LN",
        dropout: float = 0.0,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.norm_type = norm_type
        self.activation_type = activation_type
        self.residual_type = residual_type
        self.dropout = dropout
    
    @abstractmethod
    def forward(self, x: Any, mask: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, block_stats)
        """
        pass
    
    @abstractmethod
    def backward(self, grad_output: Any) -> Tuple[Any, Dict[str, Any]]:
        """Backward pass through transformer block."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[Any]:
        """Get trainable parameters."""
        pass


class AbstractTransformer(ABC):
    """Abstract transformer model."""
    
    def __init__(
        self,
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
        backend_config: Optional[BackendConfig] = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.norm_type = norm_type
        self.activation_type = activation_type
        self.residual_type = residual_type
        self.pos_encoding_type = pos_encoding_type
        self.dropout = dropout
        self.backend_config = backend_config or BackendConfig(BackendType.NUMPY)
        
        # Training state
        self.training = True
        self.loss_history: List[float] = []
        self.step_count = 0
    
    @abstractmethod
    def forward(self, x: Any, targets: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Forward pass through transformer.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices for loss calculation
            
        Returns:
            Tuple of (logits, model_stats)
        """
        pass
    
    @abstractmethod
    def backward(self, logits: Any, targets: Any) -> Tuple[float, Dict[str, Any]]:
        """Backward pass to compute gradients.
        
        Args:
            logits: Model output logits
            targets: Target token indices
            
        Returns:
            Tuple of (loss, gradients_dict)
        """
        pass
    
    @abstractmethod
    def train_step(self, x: Any, targets: Any, optimizer: AbstractOptimizer) -> float:
        """Single training step.
        
        Args:
            x: Input tokens
            targets: Target tokens
            optimizer: Optimizer instance
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        prompt: Any, 
        max_length: int = 50, 
        temperature: float = 1.0
    ) -> Any:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token sequence
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[Any]:
        """Get all trainable parameters."""
        pass
    
    @abstractmethod
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model parameters from disk."""
        pass
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend-specific information."""
        return {
            "backend_type": self.backend_config.backend_type.value,
            "device": self.backend_config.device,
            "dtype": self.backend_config.dtype,
            "parameter_count": self.get_parameter_count(),
            "memory_usage": self._estimate_memory_usage(),
        }
    
    @abstractmethod
    def _estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in MB."""
        pass
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        return {
            "loss_history": self.loss_history,
            "step_count": self.step_count,
            "backend_info": self.get_backend_info(),
            "config": {
                "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "max_seq_len": self.max_seq_len,
                "norm_type": self.norm_type,
                "activation_type": self.activation_type,
                "residual_type": self.residual_type,
                "pos_encoding_type": self.pos_encoding_type,
                "dropout": self.dropout,
            },
        }