"""
PoolBench — public top-level API.

Quick start
-----------
    from poolbench import STRATEGY_REGISTRY, CONCEPTS
    from poolbench import compute_all_pooling_strategies
    from poolbench import compute_auroc_for_strategy
    from poolbench import get_construction_method

Subpackages
-----------
    poolbench.strategies   — all 19 ranked pooling functions + registry
    poolbench.construction — C1–C5 concept-direction construction methods
    poolbench.evaluation   — AUROC probing, Nemenyi tests, linearity checks
    poolbench.data         — CONCEPTS dict, filters, rewriters
"""

from poolbench.data.concepts import CONCEPTS, CONCEPT_NAMES, MATCHED_PAIR_CONCEPTS
from poolbench.construction.methods import (
    construct_difmean,
    get_construction_method,
    DEFAULT_CONSTRUCTION,
    CONSTRUCTION_METHODS,
)
from poolbench.pooling_strategies import (
    STRATEGY_REGISTRY,
    RANKED_STRATEGIES,
    OFF_LEADERBOARD,
    compute_all_pooling_strategies,
    pool_mean,
    pool_last_token,
    pool_first_token,
)
from poolbench.evaluation.probe import (
    compute_auroc_for_strategy,
    compute_all_auroc,
    nemenyi_strategy_significance,
    check_linearity_assumption,
    compute_layer_icc,
)

__version__ = "0.1.0"
__all__ = [
    # Data
    "CONCEPTS", "CONCEPT_NAMES", "MATCHED_PAIR_CONCEPTS",
    # Strategies
    "STRATEGY_REGISTRY", "RANKED_STRATEGIES", "OFF_LEADERBOARD",
    "compute_all_pooling_strategies",
    "pool_mean", "pool_last_token", "pool_first_token",
    # Construction
    "construct_difmean", "get_construction_method",
    "DEFAULT_CONSTRUCTION", "CONSTRUCTION_METHODS",
    # Evaluation
    "compute_auroc_for_strategy", "compute_all_auroc",
    "nemenyi_strategy_significance", "check_linearity_assumption", "compute_layer_icc",
    # Meta
    "__version__",
]
