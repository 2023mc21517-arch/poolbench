"""
poolbench/construction_methods.py — backward-compatibility shim.
All symbols are canonical in poolbench.construction.methods.
This module re-exports everything so existing imports keep working.
"""
from __future__ import annotations  # noqa: F401  (keep for mypy compat)
from poolbench.construction.methods import *  # noqa: F401, F403
from poolbench.construction.methods import (
    construct_difmean,
    construct_pca,
    construct_logreg,
    construct_repe,
    construct_sae_feature,
    CONSTRUCTION_METHODS,
    DEFAULT_CONSTRUCTION,
    get_construction_method,
)
