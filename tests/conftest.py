"""
tests/conftest.py
Shared pytest fixtures available to all test modules.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_hidden(rng):
    """A small (10, 16) hidden state array for fast unit tests."""
    return rng.standard_normal((10, 16)).astype(np.float32)


@pytest.fixture
def medium_hidden(rng):
    """A (50, 64) hidden state array for strategy/construction tests."""
    return rng.standard_normal((50, 64)).astype(np.float32)


@pytest.fixture
def separable_pos_neg(rng):
    """Linearly separable (100, 32) pos/neg arrays, mean offset = 2.0."""
    pos = (rng.standard_normal((100, 32)) + 2.0).astype(np.float32)
    neg = (rng.standard_normal((100, 32)) - 2.0).astype(np.float32)
    return pos, neg
