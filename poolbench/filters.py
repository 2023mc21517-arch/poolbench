"""
poolbench/filters.py — backward-compatibility shim.
All symbols are canonical in poolbench.data.filters.
This module re-exports everything so existing imports keep working.
"""
from poolbench.data.filters import *  # noqa: F401, F403
