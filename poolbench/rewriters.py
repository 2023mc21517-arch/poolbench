"""
poolbench/rewriters.py — backward-compatibility shim.
All symbols are canonical in poolbench.data.rewriters.
This module re-exports everything so existing imports keep working.
"""
from poolbench.data.rewriters import *  # noqa: F401, F403
