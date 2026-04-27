"""
poolbench/concepts.py — backward-compatibility shim.
All symbols are canonical in poolbench.data.concepts.
This module re-exports everything so existing imports keep working.
"""
from poolbench.data.concepts import *  # noqa: F401, F403
from poolbench.data.concepts import (
    CONCEPTS,
    CONCEPT_NAMES,
    MATCHED_PAIR_CONCEPTS,
    DETERMINISTIC_CONCEPTS,
)
