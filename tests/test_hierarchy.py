"""
Tests for hierarchical reconciliation — the defining property is coherence:
leaf forecasts must sum to their parent total.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hierarchy import (
    bottom_up, top_down, historical_proportions, is_coherent,
)


def test_bottom_up_total_is_leaf_sum():
    leaves = {"a": np.array([1.0, 2, 3]), "b": np.array([4.0, 5, 6])}
    r = bottom_up(leaves)
    assert np.allclose(r["total"], [5, 7, 9])
    assert is_coherent(r)


def test_proportions_sum_to_one():
    hist = {"a": np.array([10, 10]), "b": np.array([30, 0]), "c": np.array([0, 0])}
    props = historical_proportions(hist)
    assert abs(sum(props.values()) - 1.0) < 1e-9


def test_proportions_reflect_history_share():
    hist = {"a": np.array([75.0]), "b": np.array([25.0])}
    props = historical_proportions(hist)
    assert abs(props["a"] - 0.75) < 1e-9
    assert abs(props["b"] - 0.25) < 1e-9


def test_top_down_is_coherent():
    total = np.array([100.0, 200.0])
    hist = {"a": np.array([3.0]), "b": np.array([1.0])}
    r = top_down(total, hist)
    assert is_coherent(r)
    # a has 75% share
    assert np.allclose(r["leaves"]["a"], [75, 150])


def test_zero_history_falls_back_to_equal_split():
    hist = {"a": np.array([0.0]), "b": np.array([0.0])}
    props = historical_proportions(hist)
    assert abs(props["a"] - 0.5) < 1e-9


def test_incoherent_set_detected():
    bad = {"leaves": {"a": np.array([1.0]), "b": np.array([1.0])},
           "total": np.array([5.0])}  # 1+1 != 5
    assert not is_coherent(bad)
