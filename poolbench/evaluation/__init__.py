"""
poolbench.evaluation
~~~~~~~~~~~~~~~~~~~~
AUROC probing, statistical tests, linearity validation, ICC, and D2/D3 metrics.
"""
from poolbench.evaluation.probe import (
    compute_auroc_for_strategy,
    compute_all_auroc,
    build_nemenyi_auroc_matrix,
    nemenyi_strategy_significance,
    check_linearity_assumption,
    compute_layer_icc,
    keyword_ablation_check,
    LINEARITY_GAP_THRESHOLD,
    N_BOOTSTRAP,
    N_FOLDS,
    RANDOM_SEED,
)

__all__ = [
    "compute_auroc_for_strategy",
    "compute_all_auroc",
    "build_nemenyi_auroc_matrix",
    "nemenyi_strategy_significance",
    "check_linearity_assumption",
    "compute_layer_icc",
    "keyword_ablation_check",
    "LINEARITY_GAP_THRESHOLD",
    "N_BOOTSTRAP",
    "N_FOLDS",
    "RANDOM_SEED",
    # D2 / D3 — imported lazily to avoid heavy HF imports at package import time
    "classifier_b",
    "scp_eval",
    "disentanglement",
]
