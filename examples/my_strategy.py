"""
examples/my_strategy.py
~~~~~~~~~~~~~~~~~~~~~~~
Template for contributing a new pooling strategy to PoolBench.

Copy this file, rename the function, implement your logic, then run evaluate.py
to get your AUROC/SCP/D3 scores for leaderboard submission.
"""

from __future__ import annotations
import numpy as np


def pool_my_strategy(h: np.ndarray, **kwargs) -> np.ndarray:
    """
    Replace this docstring with a description of your strategy.

    Parameters
    ----------
    h : np.ndarray
        Hidden-state matrix of shape (seq_len, d_model), float32.
        Each row is the hidden state of one token at the target layer.
    **kwargs
        Optional extra inputs. Supported keys:
          - text           : str  — raw passage text (for spaCy-based strategies)
          - token_ids      : list[int] — HuggingFace token IDs
          - offset_mapping : list[tuple[int,int]] — char-offset pairs per token
          - attn_weights   : np.ndarray (n_heads, seq_len, seq_len) or None

    Returns
    -------
    np.ndarray
        Pooled representation, shape (d_model,), dtype float32.
    """
    # ── YOUR IMPLEMENTATION HERE ──────────────────────────────────────────
    # Example: weighted mean based on token position (trivial, replace with yours)
    seq_len = h.shape[0]
    weights = np.arange(1, seq_len + 1, dtype=np.float32)
    weights /= weights.sum()
    return (weights[:, None] * h).sum(axis=0)


# ── Registration metadata ──────────────────────────────────────────────────────
# Fill this in before submitting your leaderboard JSON.

STRATEGY_METADATA = {
    "id":          "MY_strategy_name",      # e.g. "W5_position_weighted"
    "family":      "window",                # position_anchored | uniform_aggregation |
                                            # window | saliency_weighted | structural_linguistic
    "supervision": "unsupervised",          # unsupervised | supervised
    "source":      "community",             # community (always for external submissions)
    "description": "Short plain-English description of what makes this strategy different.",
    "citation":    "Author et al., Venue YYYY — link if applicable, or 'original'",
}


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    h = rng.standard_normal((50, 64)).astype(np.float32)
    vec = pool_my_strategy(h)
    assert vec.shape == (64,), f"Expected (64,), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
    print(f"Smoke test passed: output shape {vec.shape}, dtype {vec.dtype}")
