"""
tests/test_pooling.py
Unit tests for poolbench.pooling_strategies.
Run from project root: pytest tests/ -v
"""

import numpy as np
import pytest
from poolbench.pooling_strategies import (
    pool_last_token, pool_first_token, pool_mean, pool_sum, pool_max,
    pool_min, pool_median, pool_random, pool_mean_last_4, pool_mean_last_8,
    pool_mean_last_16, pool_hierarchical, pool_first_last_concat,
    STRATEGY_REGISTRY, RANKED_STRATEGIES, OFF_LEADERBOARD,
)


@pytest.fixture
def h():
    """A deterministic (10, 16) hidden-state array for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 16)).astype(np.float32)


# ── Output shape ──────────────────────────────────────────────────────────────

def test_last_token_shape(h):
    assert pool_last_token(h).shape == (16,)


def test_first_token_shape(h):
    assert pool_first_token(h).shape == (16,)


def test_mean_shape(h):
    assert pool_mean(h).shape == (16,)


def test_sum_shape(h):
    assert pool_sum(h).shape == (16,)


def test_max_shape(h):
    assert pool_max(h).shape == (16,)


def test_min_shape(h):
    assert pool_min(h).shape == (16,)


def test_median_shape(h):
    assert pool_median(h).shape == (16,)


def test_random_shape(h):
    assert pool_random(h, seed=42).shape == (16,)


def test_mean_last_4_shape(h):
    assert pool_mean_last_4(h).shape == (16,)


def test_mean_last_8_shape(h):
    assert pool_mean_last_8(h).shape == (16,)


def test_mean_last_16_shape(h):
    assert pool_mean_last_16(h).shape == (16,)


def test_hierarchical_shape(h):
    assert pool_hierarchical(h).shape == (16,)


def test_first_last_concat_shape(h):
    vec = pool_first_last_concat(h)
    assert vec.shape == (32,), "P3 should produce 2×d_model"


# ── Correctness ───────────────────────────────────────────────────────────────

def test_last_token_value(h):
    np.testing.assert_array_equal(pool_last_token(h), h[-1])


def test_first_token_value(h):
    np.testing.assert_array_equal(pool_first_token(h), h[0])


def test_mean_value(h):
    np.testing.assert_allclose(pool_mean(h), h.mean(axis=0))


def test_max_value(h):
    np.testing.assert_array_equal(pool_max(h), h.max(axis=0))


def test_random_deterministic(h):
    """Same seed → identical result."""
    assert np.allclose(pool_random(h, seed=7), pool_random(h, seed=7))


def test_random_different_seeds(h):
    """Different seeds → different results (almost certainly)."""
    v1 = pool_random(h, seed=1)
    v2 = pool_random(h, seed=2)
    assert not np.allclose(v1, v2)


def test_mean_last_4_uses_last_4(h):
    expected = h[-4:].mean(axis=0)
    np.testing.assert_allclose(pool_mean_last_4(h), expected)


def test_first_last_concat_values(h):
    expected = np.concatenate([h[0], h[-1]])
    np.testing.assert_array_equal(pool_first_last_concat(h), expected)


# ── Short-sequence edge cases ─────────────────────────────────────────────────

def test_mean_last_4_short_sequence():
    """Input shorter than window → should not raise, uses all tokens."""
    h_short = np.ones((3, 8), dtype=np.float32)
    result = pool_mean_last_4(h_short)
    assert result.shape == (8,)


def test_hierarchical_single_token():
    h_1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = pool_hierarchical(h_1)
    assert result.shape == (3,)


# ── Registry completeness ─────────────────────────────────────────────────────

def test_ranked_strategies_count():
    assert len(RANKED_STRATEGIES) == 19, (
        f"Expected 19 ranked strategies, got {len(RANKED_STRATEGIES)}"
    )


def test_off_leaderboard_not_in_ranked():
    for s in OFF_LEADERBOARD:
        assert s not in RANKED_STRATEGIES, f"{s} should not be in RANKED_STRATEGIES"


def test_all_ranked_in_registry():
    for s in RANKED_STRATEGIES:
        assert s in STRATEGY_REGISTRY, f"{s} missing from STRATEGY_REGISTRY"


def test_registry_callables():
    for name, tup in STRATEGY_REGISTRY.items():
        fn, family = tup[0], tup[1]
        assert callable(fn), f"{name}: pool function is not callable"
        assert isinstance(family, str), f"{name}: family must be a string"
