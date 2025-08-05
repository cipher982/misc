"""
Simple character-level tokenizer for the Transformer Intuition Lab.
"""

import numpy as np
from typing import List, Tuple, Dict


class CharTokenizer:
    """Character-level tokenizer for educational purposes."""
    
    def __init__(self, text: str):
        """Initialize tokenizer with text corpus."""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        return ''.join([self.idx_to_char.get(idx, '?') for idx in indices])
    
    def encode_batch(self, texts: List[str], max_length: int = None) -> np.ndarray:
        """Encode a batch of texts to a 2D array."""
        if max_length is None:
            max_length = max(len(text) for text in texts)
        
        batch = np.zeros((len(texts), max_length), dtype=np.int32)
        for i, text in enumerate(texts):
            tokens = self.encode(text)
            batch[i, :len(tokens)] = tokens[:max_length]
        
        return batch
    
    def get_vocab_stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'vocab_size': self.vocab_size,
            'chars': self.chars,
            'sample_chars': self.chars[:10] + ['...'] if len(self.chars) > 10 else self.chars
        }


def load_corpus(filename: str) -> Tuple[str, CharTokenizer]:
    """Load corpus and create tokenizer."""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    return text, tokenizer