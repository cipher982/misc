#!/usr/bin/env python3
"""
Quick training test script for Makefile.
"""

from transformerlab.backends.factory import create_transformer
from transformerlab.backends.numpy_backend.optimizer import create_numpy_optimizer
import numpy as np

def main():
    model = create_transformer(
        'numpy', 
        vocab_size=20, 
        hidden_dim=16, 
        num_layers=1, 
        num_heads=2, 
        ff_dim=32
    )
    optimizer = create_numpy_optimizer('sgd', learning_rate=0.01)
    
    print('Training 5 steps:')
    print('================')
    
    for step in range(5):
        x = np.random.randint(1, 19, size=(2, 5))
        targets = np.random.randint(1, 19, size=(2, 5))
        try:
            loss = model.train_step(x.tolist(), targets.tolist(), optimizer)
            print(f'Step {step+1}: Loss = {loss:.4f}')
        except Exception as e:
            print(f'Step {step+1}: ❌ {e}')
            break

if __name__ == '__main__':
    main()