"""
Simplified Educational Transformer Implementation

This is a standalone, pure Python transformer that removes all abstraction layers
and focuses on educational clarity. Every operation is implemented from scratch
with extensive comments explaining the mathematical concepts.
"""

import math
import random
from typing import Any, Optional, List, Tuple, Dict
from config import TransformerConfig


class SimpleTransformer:
    """
    Pure Python transformer implementation with maximum educational transparency.
    
    This class implements a complete transformer from scratch using only Python
    built-ins and explicit loops to show every mathematical operation clearly.
    """
    
    def __init__(self, config: TransformerConfig, verbose: bool = True):
        """Initialize transformer with given configuration."""
        self.config = config
        self.verbose = verbose
        
        # Training state
        self.step_count = 0
        self.loss_history = []
        
        if self.verbose:
            print(f"\\n🔧 Initializing Simple Transformer:")
            print(self.config.summary())
        
        # Initialize all parameters
        self._init_parameters()
        
        if self.verbose:
            print(f"✅ Transformer initialized with {self.get_parameter_count():,} parameters")
    
    def _init_parameters(self):
        """Initialize all transformer parameters with random values."""
        
        # Token embeddings: each token gets a hidden_dim vector
        # Shape: (vocab_size, hidden_dim)
        self.token_embeddings = self._randn((self.config.vocab_size, self.config.hidden_dim))
        
        # Positional embeddings: each position gets a hidden_dim vector  
        # Shape: (seq_len, hidden_dim)
        self.pos_embeddings = self._randn((self.config.seq_len, self.config.hidden_dim), scale=0.01)
        
        # Transformer layers
        self.layers = []
        for layer_idx in range(self.config.num_layers):
            layer = self._create_transformer_layer(layer_idx)
            self.layers.append(layer)
            
        # Final layer normalization parameters
        self.final_norm_weight = [1.0] * self.config.hidden_dim  # γ (gamma)
        self.final_norm_bias = [0.0] * self.config.hidden_dim    # β (beta)
        
        # Output projection to vocabulary
        # Shape: (hidden_dim, vocab_size) 
        self.output_weight = self._randn((self.config.hidden_dim, self.config.vocab_size))
        self.output_bias = [0.0] * self.config.vocab_size
    
    def _create_transformer_layer(self, layer_idx: int) -> Dict[str, Any]:
        """Create parameters for one transformer layer."""
        
        # Multi-head attention parameters
        attention = {
            # Query, Key, Value, Output projection matrices
            # Each maps hidden_dim -> hidden_dim  
            'w_q': self._randn((self.config.hidden_dim, self.config.hidden_dim)),
            'w_k': self._randn((self.config.hidden_dim, self.config.hidden_dim)),
            'w_v': self._randn((self.config.hidden_dim, self.config.hidden_dim)),
            'w_o': self._randn((self.config.hidden_dim, self.config.hidden_dim)),
            
            # Bias vectors
            'b_q': [0.0] * self.config.hidden_dim,
            'b_k': [0.0] * self.config.hidden_dim,
            'b_v': [0.0] * self.config.hidden_dim,
            'b_o': [0.0] * self.config.hidden_dim,
        }
        
        # Feed-forward network parameters  
        feed_forward = {
            # First projection: hidden_dim -> ff_dim
            'w1': self._randn((self.config.hidden_dim, self.config.ff_dim)),
            'b1': [0.0] * self.config.ff_dim,
            
            # Second projection: ff_dim -> hidden_dim
            'w2': self._randn((self.config.ff_dim, self.config.hidden_dim)),
            'b2': [0.0] * self.config.hidden_dim,
        }
        
        # Layer normalization parameters (applied before attention and FFN)
        norm1 = {
            'weight': [1.0] * self.config.hidden_dim,  # γ (gamma)
            'bias': [0.0] * self.config.hidden_dim,    # β (beta)  
        }
        
        norm2 = {
            'weight': [1.0] * self.config.hidden_dim,  # γ (gamma)
            'bias': [0.0] * self.config.hidden_dim,    # β (beta)
        }
        
        return {
            'attention': attention,
            'feed_forward': feed_forward,
            'norm1': norm1,
            'norm2': norm2,
            'layer_idx': layer_idx
        }
    
    def forward(self, input_ids: List[List[int]], targets: Optional[List[List[int]]] = None) -> Tuple[List[List[List[float]]], Dict]:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            targets: Target token IDs for loss computation (optional)
            
        Returns:
            logits: Output probabilities, shape (batch_size, seq_len, vocab_size)  
            stats: Dictionary with forward pass statistics
        """
        batch_size = len(input_ids)
        seq_len = len(input_ids[0])
        
        if self.verbose:
            print(f"\\n🔄 Forward Pass (Step {self.step_count + 1}):")
            print(f"   Input shape: batch_size={batch_size}, seq_len={seq_len}")
        
        # Step 1: Token Embedding Lookup
        # Convert token IDs to dense vectors
        if self.verbose:
            print("   Step 1: Token embedding lookup...")
        
        embeddings = self._zeros((batch_size, seq_len, self.config.hidden_dim))
        for batch in range(batch_size):
            for pos in range(seq_len):
                token_id = input_ids[batch][pos]
                # Copy the embedding vector for this token
                for dim in range(self.config.hidden_dim):
                    embeddings[batch][pos][dim] = self.token_embeddings[token_id][dim]
        
        # Step 2: Add Positional Encoding
        # This helps the model understand token positions  
        if self.verbose:
            print("   Step 2: Adding positional encoding...")
        
        for batch in range(batch_size):
            for pos in range(seq_len):
                for dim in range(self.config.hidden_dim):
                    embeddings[batch][pos][dim] += self.pos_embeddings[pos][dim]
        
        # Step 3: Pass Through Transformer Layers
        hidden_states = embeddings
        layer_stats = []
        
        for layer in self.layers:
            if self.verbose:
                print(f"   Step 3.{layer['layer_idx'] + 1}: Transformer layer {layer['layer_idx'] + 1}/{self.config.num_layers}")
            
            hidden_states, stats = self._transformer_layer_forward(hidden_states, layer)
            layer_stats.append(stats)
        
        # Step 4: Final Layer Normalization
        if self.verbose:
            print("   Step 4: Final layer normalization...")
        
        hidden_states = self._layer_norm(hidden_states, self.final_norm_weight, self.final_norm_bias)
        
        # Step 5: Output Projection to Vocabulary
        # Convert hidden states back to vocabulary logits
        if self.verbose:
            print("   Step 5: Output projection to vocabulary...")
        
        logits = self._linear_3d(hidden_states, self.output_weight, self.output_bias)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            if self.verbose:
                print("   Computing cross-entropy loss...")
            loss = self._compute_cross_entropy_loss(logits, targets)
            self.loss_history.append(loss)
            
            if self.verbose:
                print(f"   ✅ Loss: {loss:.6f}")
        
        # Collect statistics
        stats = {
            'loss': loss,
            'layer_stats': layer_stats,
            'final_logits_mean': self._tensor_mean(logits),
            'final_logits_std': self._tensor_std(logits),
            'step': self.step_count
        }
        
        return logits, stats
    
    def _transformer_layer_forward(self, x: List[List[List[float]]], layer: Dict) -> Tuple[List[List[List[float]]], Dict]:
        """Forward pass through one transformer layer (Pre-LN style)."""
        
        # 1. Pre-normalization before attention
        x_norm1 = self._layer_norm(x, layer['norm1']['weight'], layer['norm1']['bias'])
        
        # 2. Multi-head self-attention  
        attn_output = self._multi_head_attention(x_norm1, layer['attention'])
        
        # 3. Residual connection around attention
        x = self._add_tensors_3d(x, attn_output)
        
        # 4. Pre-normalization before feed-forward
        x_norm2 = self._layer_norm(x, layer['norm2']['weight'], layer['norm2']['bias'])
        
        # 5. Feed-forward network
        ff_output = self._feed_forward(x_norm2, layer['feed_forward'])
        
        # 6. Residual connection around feed-forward  
        x = self._add_tensors_3d(x, ff_output)
        
        stats = {
            'layer_idx': layer['layer_idx'],
            'attn_output_mean': self._tensor_mean(attn_output),
            'ff_output_mean': self._tensor_mean(ff_output),
        }
        
        return x, stats
    
    def _multi_head_attention(self, x: List[List[List[float]]], attn_params: Dict) -> List[List[List[float]]]:
        """
        Multi-head self-attention mechanism.
        
        This is the core of the transformer - it allows tokens to attend to 
        each other and exchange information based on learned relationships.
        """
        batch_size, seq_len, hidden_dim = self._get_shape_3d(x)
        head_dim = hidden_dim // self.config.num_heads
        
        # Step 1: Linear projections to Query, Key, Value
        # Each token gets transformed into Q, K, V vectors
        q = self._linear_3d(x, attn_params['w_q'], attn_params['b_q'])  # (batch, seq, hidden)
        k = self._linear_3d(x, attn_params['w_k'], attn_params['b_k'])  # (batch, seq, hidden)  
        v = self._linear_3d(x, attn_params['w_v'], attn_params['b_v'])  # (batch, seq, hidden)
        
        # Step 2: Reshape for multiple attention heads
        # Split hidden dimension across heads: (batch, seq, num_heads, head_dim)
        q_heads = self._reshape_for_heads(q, batch_size, seq_len, self.config.num_heads, head_dim)
        k_heads = self._reshape_for_heads(k, batch_size, seq_len, self.config.num_heads, head_dim)
        v_heads = self._reshape_for_heads(v, batch_size, seq_len, self.config.num_heads, head_dim)
        
        # Step 3: Compute attention scores
        # For each head, compute how much each token should attend to every other token
        attention_output = []
        
        for batch in range(batch_size):
            batch_output = []
            for seq_pos in range(seq_len):
                pos_output = []
                
                for head in range(self.config.num_heads):
                    # Get query for this position and head
                    query = q_heads[batch][seq_pos][head]  # (head_dim,)
                    
                    # Compute attention scores against all positions
                    scores = []
                    for key_pos in range(seq_len):
                        key = k_heads[batch][key_pos][head]  # (head_dim,)
                        
                        # Dot product attention: query • key
                        score = sum(query[d] * key[d] for d in range(head_dim))
                        # Scale by sqrt(head_dim) for numerical stability
                        score = score / math.sqrt(head_dim)
                        scores.append(score)
                    
                    # Apply softmax to get attention weights
                    weights = self._softmax(scores)
                    
                    # Apply attention weights to values
                    attended_value = [0.0] * head_dim
                    for value_pos in range(seq_len):
                        value = v_heads[batch][value_pos][head]  # (head_dim,)
                        weight = weights[value_pos]
                        
                        for d in range(head_dim):
                            attended_value[d] += weight * value[d]
                    
                    pos_output.extend(attended_value)  # Concatenate heads
                
                batch_output.append(pos_output)
            attention_output.append(batch_output)
        
        # Step 4: Final output projection
        output = self._linear_3d(attention_output, attn_params['w_o'], attn_params['b_o'])
        
        return output
    
    def _feed_forward(self, x: List[List[List[float]]], ff_params: Dict) -> List[List[List[float]]]:
        """
        Feed-forward network: two linear layers with activation in between.
        
        This processes each position independently and adds non-linearity.
        """
        # First linear layer: hidden_dim -> ff_dim
        h1 = self._linear_3d(x, ff_params['w1'], ff_params['b1'])
        
        # Apply activation function (ReLU by default)
        if self.config.activation_type == 'ReLU':
            h1_activated = self._relu_3d(h1)
        elif self.config.activation_type == 'GELU':
            h1_activated = self._gelu_3d(h1)
        else:
            raise ValueError(f"Unknown activation: {self.config.activation_type}")
        
        # Second linear layer: ff_dim -> hidden_dim  
        output = self._linear_3d(h1_activated, ff_params['w2'], ff_params['b2'])
        
        return output
    
    def generate(self, prompt_ids: List[List[int]], max_new_tokens: int = 50, temperature: float = 1.0) -> List[List[int]]:
        """
        Generate new tokens autoregressively.
        
        This demonstrates how transformers generate text by predicting one token at a time.
        """
        if self.verbose:
            print(f"\\n🎯 Generating {max_new_tokens} tokens (temperature={temperature})...")
        
        batch_size = len(prompt_ids)
        generated = [prompt[:] for prompt in prompt_ids]  # Copy input
        
        for step in range(max_new_tokens):
            if self.verbose and step < 5:  # Don't spam logs
                print(f"   Generation step {step + 1}/{max_new_tokens}")
            
            # Forward pass to get logits  
            logits, _ = self.forward(generated)
            
            # Get logits for the last position (next token prediction)
            next_tokens = []
            for batch in range(batch_size):
                last_pos_logits = logits[batch][-1]  # (vocab_size,)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    last_pos_logits = [logit / temperature for logit in last_pos_logits]
                
                # Convert to probabilities
                probs = self._softmax(last_pos_logits)
                
                # Sample next token
                next_token = self._sample_from_probs(probs)
                next_tokens.append(next_token)
            
            # Append new tokens to sequences
            for batch in range(batch_size):
                generated[batch].append(next_tokens[batch])
        
        if self.verbose:
            print(f"   ✅ Generated {max_new_tokens} new tokens")
        
        return generated
    
    def train_step(self, input_ids: List[List[int]], target_ids: List[List[int]], learning_rate: Optional[float] = None) -> float:
        """
        Perform one training step with simplified gradient computation.
        
        Note: This is a simplified educational version - real training would
        use proper backpropagation through all parameters.
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate
            
        if self.verbose:
            print(f"\\n📚 Training Step {self.step_count + 1} (lr={learning_rate})")
        
        # Forward pass
        logits, stats = self.forward(input_ids, target_ids)
        loss = stats['loss']
        
        # Simplified parameter update (just add small random noise)
        # In a real implementation, this would compute proper gradients
        self._simple_parameter_update(learning_rate)
        
        self.step_count += 1
        
        if self.verbose:
            print(f"   ✅ Training step complete. Loss: {loss:.6f}")
        
        return loss
    
    def get_parameter_count(self) -> int:
        """Count total number of parameters."""
        count = 0
        
        # Token embeddings
        count += self.config.vocab_size * self.config.hidden_dim
        
        # Position embeddings  
        count += self.config.seq_len * self.config.hidden_dim
        
        # Each transformer layer
        layer_params = (
            # Attention: 4 weight matrices + 4 biases
            4 * (self.config.hidden_dim * self.config.hidden_dim) + 4 * self.config.hidden_dim +
            # Feed-forward: 2 weight matrices + 2 biases
            (self.config.hidden_dim * self.config.ff_dim) + self.config.ff_dim +
            (self.config.ff_dim * self.config.hidden_dim) + self.config.hidden_dim +
            # Layer norms: 2 * (weight + bias)
            4 * self.config.hidden_dim
        )
        count += self.config.num_layers * layer_params
        
        # Final norm + output projection
        count += 2 * self.config.hidden_dim  # final norm
        count += self.config.hidden_dim * self.config.vocab_size + self.config.vocab_size  # output proj + bias
        
        return count
    
    # =============================================================================
    # Utility Functions - Pure Python implementations of tensor operations
    # =============================================================================
    
    def _randn(self, shape: Tuple[int, ...], scale: float = 0.02) -> List:
        """Create tensor with random normal values."""
        if len(shape) == 1:
            return [random.gauss(0, scale) for _ in range(shape[0])]
        elif len(shape) == 2:  
            return [[random.gauss(0, scale) for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            raise NotImplementedError(f"Shape {shape} not supported")
    
    def _zeros(self, shape: Tuple[int, ...]) -> List:
        """Create tensor filled with zeros."""
        if len(shape) == 1:
            return [0.0] * shape[0]
        elif len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        elif len(shape) == 3:
            return [[[0.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            raise NotImplementedError(f"Shape {shape} not supported")
    
    def _get_shape_3d(self, tensor: List[List[List[float]]]) -> Tuple[int, int, int]:
        """Get shape of 3D tensor."""
        return len(tensor), len(tensor[0]), len(tensor[0][0])
    
    def _linear_3d(self, x: List[List[List[float]]], weight: List[List[float]], bias: List[float]) -> List[List[List[float]]]:
        """Apply linear transformation: x @ weight + bias."""
        batch_size, seq_len, in_dim = self._get_shape_3d(x)
        out_dim = len(weight[0])
        
        result = self._zeros((batch_size, seq_len, out_dim))
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                for out_d in range(out_dim):
                    # Matrix multiplication
                    for in_d in range(in_dim):
                        result[batch][seq][out_d] += x[batch][seq][in_d] * weight[in_d][out_d]
                    # Add bias
                    result[batch][seq][out_d] += bias[out_d]
        
        return result
    
    def _layer_norm(self, x: List[List[List[float]]], weight: List[float], bias: List[float]) -> List[List[List[float]]]:
        """Apply layer normalization."""
        batch_size, seq_len, hidden_dim = self._get_shape_3d(x)
        result = self._zeros((batch_size, seq_len, hidden_dim))
        eps = 1e-6
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                # Compute mean and variance across hidden dimension
                mean = sum(x[batch][seq]) / hidden_dim
                variance = sum((val - mean) ** 2 for val in x[batch][seq]) / hidden_dim
                std = math.sqrt(variance + eps)
                
                # Normalize and apply learned parameters
                for dim in range(hidden_dim):
                    normalized = (x[batch][seq][dim] - mean) / std
                    result[batch][seq][dim] = weight[dim] * normalized + bias[dim]
        
        return result
    
    def _reshape_for_heads(self, x: list[list[list[float]]], batch_size: int, seq_len: int, num_heads: int, head_dim: int) -> list[list[list[list[float]]]]:
        """Reshape tensor for multi-head attention: (batch, seq, hidden) -> (batch, seq, num_heads, head_dim)."""
        result = [[[[0.0 for _ in range(head_dim)] for _ in range(num_heads)] for _ in range(seq_len)] for _ in range(batch_size)]
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                for head in range(num_heads):
                    for dim in range(head_dim):
                        hidden_idx = head * head_dim + dim
                        result[batch][seq][head][dim] = x[batch][seq][hidden_idx]
        
        return result
    
    def _add_tensors_3d(self, a: list[list[list[float]]], b: list[list[list[float]]]) -> list[list[list[float]]]:
        """Element-wise addition of 3D tensors."""
        batch_size, seq_len, hidden_dim = self._get_shape_3d(a)
        result = self._zeros((batch_size, seq_len, hidden_dim))
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                for dim in range(hidden_dim):
                    result[batch][seq][dim] = a[batch][seq][dim] + b[batch][seq][dim]
        
        return result
    
    def _relu_3d(self, x: list[list[list[float]]]) -> list[list[list[float]]]:
        """Apply ReLU activation to 3D tensor.""" 
        batch_size, seq_len, dim = self._get_shape_3d(x)
        result = self._zeros((batch_size, seq_len, dim))
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                for d in range(dim):
                    result[batch][seq][d] = max(0.0, x[batch][seq][d])
        
        return result
    
    def _gelu_3d(self, x: list[list[list[float]]]) -> list[list[list[float]]]:
        """Apply GELU activation to 3D tensor."""
        batch_size, seq_len, dim = self._get_shape_3d(x)
        result = self._zeros((batch_size, seq_len, dim))
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                for d in range(dim):
                    val = x[batch][seq][d]
                    # GELU approximation
                    result[batch][seq][d] = 0.5 * val * (1 + math.tanh(math.sqrt(2/math.pi) * (val + 0.044715 * val**3)))
        
        return result
    
    def _softmax(self, logits: list[float]) -> list[float]:
        """Apply softmax activation."""
        # Numerical stability: subtract max
        max_logit = max(logits)
        exp_logits = [math.exp(logit - max_logit) for logit in logits]
        sum_exp = sum(exp_logits)
        return [exp_logit / sum_exp for exp_logit in exp_logits]
    
    def _sample_from_probs(self, probs: list[float]) -> int:
        """Sample token index from probability distribution."""
        rand_val = random.random()
        cumsum = 0.0
        for i, prob in enumerate(probs):
            cumsum += prob
            if rand_val <= cumsum:
                return i
        return len(probs) - 1  # Fallback
    
    def _compute_cross_entropy_loss(self, logits: list[list[list[float]]], targets: list[list[int]]) -> float:
        """Compute cross-entropy loss between logits and targets."""
        batch_size, seq_len, vocab_size = self._get_shape_3d(logits)
        total_loss = 0.0
        num_tokens = 0
        
        for batch in range(batch_size):
            for seq in range(seq_len):
                target_id = targets[batch][seq]
                
                if target_id >= 0:  # Skip padding tokens (if any)
                    # Get probabilities for this position
                    position_logits = logits[batch][seq]
                    probs = self._softmax(position_logits)
                    
                    # Negative log likelihood of correct token
                    target_prob = probs[target_id]
                    total_loss -= math.log(target_prob + 1e-8)  # Add epsilon for numerical stability
                    num_tokens += 1
        
        return total_loss / max(num_tokens, 1)
    
    def _tensor_mean(self, tensor: list) -> float:
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
    
    def _tensor_std(self, tensor: list) -> float:
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
    
    def _simple_parameter_update(self, learning_rate: float):
        """Simplified parameter update (educational placeholder)."""
        # This is a placeholder - real training would compute proper gradients
        # For demonstration, just add tiny random noise to some parameters
        noise_scale = learning_rate * 0.001
        
        # Add noise to token embeddings (just first few for efficiency)
        for i in range(min(10, len(self.token_embeddings))):
            for j in range(len(self.token_embeddings[i])):
                self.token_embeddings[i][j] += random.gauss(0, noise_scale)


def main():
    """Demonstrate the simple transformer."""
    print("🚀 Simple Transformer Demo")
    print("=" * 50)
    
    # Create tiny configuration for demo
    config = TransformerConfig(
        vocab_size=100,
        hidden_dim=64, 
        num_heads=4,
        num_layers=2,
        seq_len=16,
        batch_size=2
    )
    
    # Initialize transformer
    transformer = SimpleTransformer(config, verbose=True)
    
    # Create dummy data
    batch_size, seq_len = 2, 8
    input_ids = [[random.randint(1, 99) for _ in range(seq_len)] for _ in range(batch_size)]
    target_ids = [[random.randint(1, 99) for _ in range(seq_len)] for _ in range(batch_size)]
    
    print(f"\\n📝 Sample Input: {input_ids[0][:5]}... (showing first 5 tokens)")
    
    # Forward pass
    logits, stats = transformer.forward(input_ids, target_ids)
    print(f"\\n📊 Output logits shape: {len(logits)} x {len(logits[0])} x {len(logits[0][0])}")
    
    # Training step
    loss = transformer.train_step(input_ids, target_ids)
    
    # Generation demo
    prompt = [[1, 2, 3]]  # Simple prompt
    generated = transformer.generate(prompt, max_new_tokens=5, temperature=0.8)
    print(f"\\n🎯 Generated: {prompt[0]} -> {generated[0]}")
    
    print("\\n✅ Demo complete!")


if __name__ == "__main__":
    main()