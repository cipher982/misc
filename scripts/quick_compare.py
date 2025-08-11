#!/usr/bin/env python3
"""
Quick backend comparison script for Makefile.
"""

from transformerlab.backends.factory import create_transformer, list_backends
import time

def main():
    config = {
        'vocab_size': 20, 
        'hidden_dim': 16, 
        'num_layers': 1, 
        'num_heads': 2, 
        'ff_dim': 32
    }
    sample_input = [[1, 2, 3, 4, 5]]
    sample_targets = [[2, 3, 4, 5, 6]]
    
    print('Backend Comparison:')
    print('==================')
    
    for backend in list_backends():
        try:
            model = create_transformer(backend, **config)
            start = time.time()
            logits, stats = model.forward(sample_input, sample_targets)
            duration = (time.time() - start) * 1000
            params = model.get_parameter_count()
            loss = stats.get('loss', 0)
            print(f'{backend:>7}: {duration:6.2f}ms, {params:>6,} params, loss={loss:.4f}')
        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            print(f'{backend:>7}: ❌ {error_msg}')

if __name__ == '__main__':
    main()