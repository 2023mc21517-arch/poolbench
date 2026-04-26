"""
tests/test_construction.py
Unit tests for poolbench.construction_methods.
"""

import numpy as np
import pytest
from poolbench.construction_methods import (
    construct_difmean, construct_pca, construct_logreg, construct_repe,
    CONSTRUCTION_METHODS, DEFAULT_CONSTRUCTION, get_construction_method,
)


@pytest.fixture
def pos_neg():
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((50, 32)).astype(np.float32) + 1.0
    neg = rng.standard_normal((50, 32)).astype(np.float32) - 1.0
    return pos, neg


def test_difmean_shape(pos_neg):
    pos, neg = pos_neg
    d = construct_difmean(pos, neg)
    assert d.shape == (32,)


def test_difmean_unit_norm(pos_neg):
    pos, neg = pos_neg
    d = construct_difmean(pos, neg)
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-5)


def test_pca_unit_norm(pos_neg):
    pos, neg = pos_neg
    d = construct_pca(pos, neg)
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-5)


def test_logreg_unit_norm(pos_neg):
    pos, neg = pos_neg
    d = construct_logreg(pos, neg)
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-5)


def test_repe_unit_norm(pos_neg):
    pos, neg = pos_neg
    d = construct_repe(pos, neg)
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-5)


def test_all_methods_registered():
    assert set(CONSTRUCTION_METHODS.keys()) == {
        "C1_difmean", "C2_pca", "C3_logreg", "C4_repe", "C5_sae_feature"
    }


def test_default_construction():
    assert DEFAULT_CONSTRUCTION == "C1_difmean"


def test_get_construction_method_valid():
    fn = get_construction_method("C1_difmean")
    assert callable(fn)


def test_get_construction_method_invalid():
    with pytest.raises(KeyError):
        get_construction_method("C99_unknown")


def test_direction_discriminates(pos_neg):
    """The concept direction from any method should project pos > neg on average."""
    pos, neg = pos_neg
    for name, fn in CONSTRUCTION_METHODS.items():
        if name == "C5_sae_feature":
            continue  # requires SAE model object
        d = fn(pos, neg)
        pos_proj = (pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-9)) @ d
        neg_proj = (neg / (np.linalg.norm(neg, axis=1, keepdims=True) + 1e-9)) @ d
        assert pos_proj.mean() > neg_proj.mean(), (
            f"{name}: pos projections not > neg projections — direction sign is flipped"
        )
