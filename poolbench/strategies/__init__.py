"""
poolbench.strategies
~~~~~~~~~~~~~~~~~~~~
All 19 ranked pooling strategies plus oracle G1_IxG.

Re-exported from the canonical implementation in the parent package
(poolbench.pooling_strategies) to avoid duplicating potentially large functions.
"""
from poolbench.pooling_strategies import (
    # Position-anchored
    pool_last_token,
    pool_first_token,
    pool_cls_token,
    # Uniform aggregation
    pool_mean,
    pool_max,
    pool_min,
    pool_sum,
    pool_median,
    pool_random,
    pool_normalised_mean,
    pool_first_last_concat,
    # Window
    pool_mean_last_4,
    pool_mean_last_8,
    pool_mean_last_16,
    pool_hierarchical,
    # Saliency-weighted
    pool_attention_weighted,
    pool_SIF_adapted,
    pool_attn_head_ITI_exact,
    # Structural-linguistic
    pool_POS_filtered,
    pool_dependency_relation,
    pool_named_entity,
    pool_subword_root_only,
    pool_SVO,
    # Oracle
    pool_IxG,
    # Registry
    STRATEGY_REGISTRY,
    RANKED_STRATEGIES,
    OFF_LEADERBOARD,
    SUPERVISED_STRATEGIES,
    STRATEGY_SUPERVISION,
    STRATEGY_SOURCE,
    compute_all_pooling_strategies,
)

__all__ = [
    "pool_last_token", "pool_first_token", "pool_cls_token",
    "pool_mean", "pool_max", "pool_min", "pool_sum",
    "pool_median", "pool_random", "pool_normalised_mean", "pool_first_last_concat",
    "pool_mean_last_4", "pool_mean_last_8", "pool_mean_last_16", "pool_hierarchical",
    "pool_attention_weighted", "pool_SIF_adapted", "pool_attn_head_ITI_exact",
    "pool_POS_filtered", "pool_dependency_relation", "pool_named_entity",
    "pool_subword_root_only", "pool_SVO",
    "pool_IxG",
    "STRATEGY_REGISTRY", "RANKED_STRATEGIES", "OFF_LEADERBOARD",
    "SUPERVISED_STRATEGIES", "STRATEGY_SUPERVISION", "STRATEGY_SOURCE",
    "compute_all_pooling_strategies",
]
