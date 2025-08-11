"""
Tests for the tokenizer module.
"""

from transformerlab.core.tokenizer import CharTokenizer


def test_char_tokenizer_initialization():
    """Test tokenizer initialization."""
    text = "hello world"
    tokenizer = CharTokenizer(text)

    assert tokenizer.vocab_size == len(set(text))
    assert len(tokenizer.char_to_idx) == tokenizer.vocab_size
    assert len(tokenizer.idx_to_char) == tokenizer.vocab_size


def test_char_tokenizer_encode_decode():
    """Test encoding and decoding."""
    text = "hello world"
    tokenizer = CharTokenizer(text)

    # Test encoding
    encoded = tokenizer.encode(text)
    assert len(encoded) == len(text)
    assert all(isinstance(idx, int) for idx in encoded)

    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_char_tokenizer_encode_batch():
    """Test batch encoding."""
    texts = ["hello", "world", "test"]
    tokenizer = CharTokenizer("".join(texts))

    batch = tokenizer.encode_batch(texts)
    assert batch.shape == (3, 5)  # 3 texts, max length 5

    # Test with specified max_length
    batch = tokenizer.encode_batch(texts, max_length=10)
    assert batch.shape == (3, 10)


def test_char_tokenizer_vocab_stats():
    """Test vocabulary statistics."""
    text = "hello world"
    tokenizer = CharTokenizer(text)

    stats = tokenizer.get_vocab_stats()
    assert "vocab_size" in stats
    assert "chars" in stats
    assert "sample_chars" in stats
    assert stats["vocab_size"] == len(set(text))
