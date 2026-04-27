"""
poolbench/probe_training.py — backward-compatibility shim.
All symbols are canonical in poolbench.evaluation.probe.
This module re-exports everything so existing imports keep working.
"""
from __future__ import annotations  # noqa: F401
from poolbench.evaluation.probe import *  # noqa: F401, F403
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
