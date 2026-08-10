"""
Tests for rolling-origin backtesting.

The load-bearing test is the no-leakage property: every training window must end
exactly where its test window begins, with no overlap and no future data.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_series import make_panel
from src.baselines import BASELINE_FORECASTERS
from src.backtest import rolling_origin_splits, backtest_panel


def test_splits_have_no_leakage():
    """train_end == test_start for every window (no overlap, no gap, no peek)."""
    splits = rolling_origin_splits(n=400, horizon=28, n_windows=3)
    for train_end, test_start, test_end in splits:
        assert train_end == test_start          # no overlap / no gap
        assert test_end - test_start == 28       # horizon length
        assert test_end <= 400                   # never past the series end


def test_splits_are_chronological_and_step_back():
    splits = rolling_origin_splits(n=400, horizon=28, n_windows=3, step=28)
    ends = [s[2] for s in splits]
    assert ends == sorted(ends)                  # chronological
    # Successive windows differ by the step
    assert ends[-1] - ends[-2] == 28


def test_min_train_respected():
    splits = rolling_origin_splits(n=200, horizon=28, n_windows=10,
                                   step=28, min_train=100)
    assert all(train_end >= 100 for train_end, _, _ in splits)


def test_backtest_panel_shapes():
    panel = make_panel(n_items=4, n_stores=2, n_days=300, seed=0)
    results, summary = backtest_panel(panel, BASELINE_FORECASTERS,
                                      horizon=28, n_windows=2)
    # 8 series x 3 methods x 2 windows
    assert len(results) == 8 * 3 * 2
    assert set(summary["method"]) == set(BASELINE_FORECASTERS.keys())
    assert {"mae", "rmse", "rmsse"}.issubset(summary.columns)


def test_summary_sorted_by_rmsse():
    panel = make_panel(n_items=4, n_stores=2, n_days=300, seed=1)
    _, summary = backtest_panel(panel, BASELINE_FORECASTERS,
                                horizon=28, n_windows=2)
    assert list(summary["rmsse"]) == sorted(summary["rmsse"])
