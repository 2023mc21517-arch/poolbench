# PoolBench — Community Examples

This folder contains templates for contributing new pooling strategies to the PoolBench leaderboard.

## Adding a new strategy

1. **Implement your strategy** using `my_strategy.py` as a template.  
   - Your function must accept `(h: np.ndarray) -> np.ndarray` where `h` is `(seq_len, d_model)`.  
   - Return a `(d_model,)` float32 vector.  
   - If your strategy needs extra inputs (text, token ids, attention weights), document them clearly.

2. **Evaluate it** using `evaluate.py` against the pre-extracted PoolBench activations on HuggingFace (`agarwalayushi/poolbench`).  
   - No GPU required — activations are pre-computed.  
   - Evaluation takes ~5 minutes on CPU.

3. **Submit to the leaderboard** by opening a pull request against this repository with:  
   - Your strategy function (in `examples/`)  
   - A filled-in `leaderboard/community/<your_strategy>.json` (see `leaderboard/schema.json`)

## Requirements

```
pip install poolbench[dev]
```

## Quick test

```python
import numpy as np
from poolbench import STRATEGY_REGISTRY, pool_mean

# Toy hidden states: 50 tokens, 64 dims
h = np.random.randn(50, 64).astype(np.float32)
vec = pool_mean(h)
print(vec.shape)   # (64,)
print(len(STRATEGY_REGISTRY))  # 20
```
