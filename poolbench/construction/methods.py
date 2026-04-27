"""
src/construction_methods.py
Five methods for constructing the concept direction vector d from a layer's activations.

C1  DifMean  — default, no fitting
C2  PCA      — data-whitened, principal direction
C3  LogReg   — max-margin discriminant
C4  RepE     — Reading-the-Mind-in-the-Eyes CCS variant (Zou et al. 2023)
C5  SAE      — Sparse AutoEncoder most-activated feature direction

All methods return a (d_model,) unit float32 direction vector.
Callers: src/probe_training.py → compute_auroc_for_strategy()
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


# ── C1 — DifMean (default) ────────────────────────────────────────────────────

def construct_difmean(pos_pooled: np.ndarray, neg_pooled: np.ndarray) -> np.ndarray:
    """
    C1  DifMean (default construction method).
    d = mean(pos_pooled) - mean(neg_pooled), normalised to unit length.
    No fitting — properties:
      • Closed-form, O(N)
      • Numerically stable (no matrix inversion)
      • Directly interpretable as a "concept direction"
    """
    d = pos_pooled.mean(axis=0) - neg_pooled.mean(axis=0)
    return _unit(d)


# ── C2 — PCA ─────────────────────────────────────────────────────────────────

def construct_pca(pos_pooled: np.ndarray, neg_pooled: np.ndarray) -> np.ndarray:
    """
    C2  PCA concept direction.
    Concatenate [pos; neg] and return the first principal component.
    When positive and negative examples are equally represented, PC1 aligns with
    the max-variance direction; on well-separated data this closely approximates
    the concept axis.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415
    X = np.vstack([pos_pooled, neg_pooled])
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X)
    # Canonical sign: PC1 should point from neg centroid toward pos centroid
    d = pca.components_[0]
    if np.dot(d, pos_pooled.mean(0) - neg_pooled.mean(0)) < 0:
        d = -d
    return _unit(d)


# ── C3 — Logistic Regression discriminant ────────────────────────────────────

def construct_logreg(pos_pooled: np.ndarray, neg_pooled: np.ndarray,
                     C: float = 1.0) -> np.ndarray:
    """
    C3  LogReg concept direction.
    Train an L2-regularised logistic regression on [pos; neg] activations.
    Direction d = normalised weight vector w.

    Note on train-test leakage: This function trains on whatever data the caller
    passes. The caller (compute_auroc_for_strategy) applies cross-validation so
    that this method is ONLY called on the current CV *training* fold.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    X = np.vstack([pos_pooled, neg_pooled])
    y = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled))
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs", random_state=42)
    clf.fit(X_norm, y)
    return _unit(clf.coef_[0])


# ── C4 — RepE (Reading Vectors / CCS-style) ──────────────────────────────────

def construct_repe(pos_pooled: np.ndarray, neg_pooled: np.ndarray) -> np.ndarray:
    """
    C4  RepE concept direction (Representation Engineering, Zou et al. 2023).
    Algorithm:
      1. Form contrast pairs (pos_i, neg_i). If |pos| != |neg|, pair by random
         sample from the smaller set.
      2. diff_i = h_pos_i - h_neg_i
      3. Apply PCA on {diff_i} → PC1 = RepE direction
    Unlike C1 (mean diff), this uses PCA on individual pair differences so it
    captures the mode of the distribution of differences rather than the mean,
    making it more robust to outlier passages.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415
    n = min(len(pos_pooled), len(neg_pooled))
    rng = np.random.default_rng(42)
    pos_idx = rng.choice(len(pos_pooled), n, replace=False) if len(pos_pooled) > n else np.arange(n)
    neg_idx = rng.choice(len(neg_pooled), n, replace=False) if len(neg_pooled) > n else np.arange(n)
    diffs = pos_pooled[pos_idx] - neg_pooled[neg_idx]   # (n, d_model)
    pca = PCA(n_components=1, random_state=42)
    pca.fit(diffs)
    d = pca.components_[0]
    # Canonical sign: should point from neg toward pos on average
    if np.dot(d, diffs.mean(0)) < 0:
        d = -d
    return _unit(d)


# ── C5 — SAE feature direction ────────────────────────────────────────────────

def construct_sae_feature(pos_pooled: np.ndarray, neg_pooled: np.ndarray,
                           sae_model, k_features: int = 10) -> Optional[np.ndarray]:
    """
    C5  SAE concept direction.
    Algorithm:
      1. Encode all [pos; neg] pooled vectors through a pre-trained SAE encoder.
         SAE encoder maps (batch, d_model) → (batch, d_sae) with non-negative activations.
      2. Compute mean activation delta per SAE feature:
         delta_j = mean(F_pos[:, j]) - mean(F_neg[:, j])
      3. Select top k_features by |delta_j|.
      4. Retrieve their decoder columns (d_sae, d_model) as candidate directions.
      5. d = mean of the top-k decoder columns (signs aligned to delta).
         Returns unit-normalised (d_model,) vector.

    `sae_model` is a SAE object from the sae-lens library exposing:
      .encode(h)          → (batch, d_sae) feature activations
      .W_dec              → (d_sae, d_model) decoder weight tensor

    Falls back to None if sae_model is None — caller uses DifMean instead.
    """
    if sae_model is None:
        return None

    import torch  # noqa: PLC0415
    X = np.vstack([pos_pooled, neg_pooled])
    X_t = torch.from_numpy(X).float()
    with torch.no_grad():
        F = sae_model.encode(X_t).cpu().numpy()  # (N, d_sae)
    n_pos = len(pos_pooled)
    delta = F[:n_pos].mean(0) - F[n_pos:].mean(0)  # (d_sae,)
    top_k = np.argsort(np.abs(delta))[-k_features:]
    W_dec = sae_model.W_dec.detach().cpu().numpy()  # (d_sae, d_model)
    cols   = W_dec[top_k]                           # (k, d_model)
    signs  = np.sign(delta[top_k])[:, None]
    d      = (cols * signs).mean(axis=0)
    return _unit(d)


# ── Registry ──────────────────────────────────────────────────────────────────

CONSTRUCTION_METHODS: dict[str, callable] = {
    "C1_difmean":     construct_difmean,
    "C2_pca":         construct_pca,
    "C3_logreg":      construct_logreg,
    "C4_repe":        construct_repe,
    "C5_sae_feature": construct_sae_feature,
}

DEFAULT_CONSTRUCTION = "C1_difmean"


def get_construction_method(name: str = DEFAULT_CONSTRUCTION) -> callable:
    if name not in CONSTRUCTION_METHODS:
        raise KeyError(f"Unknown construction method '{name}'. "
                       f"Valid: {list(CONSTRUCTION_METHODS)}")
    return CONSTRUCTION_METHODS[name]
