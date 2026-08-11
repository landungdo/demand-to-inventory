"""
Tests for prediction intervals and coverage.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_series import make_series
from src.baselines import MovingAverageForecaster
from src.intervals import (
    collect_residuals_by_step, residual_quantiles, make_interval,
    coverage, conformal_scale,
)


def test_residuals_shape():
    s = make_series(n_days=400, seed=0)
    resid = collect_residuals_by_step(
        s, lambda: MovingAverageForecaster(28), horizon=14, n_windows=5)
    assert resid.shape[1] == 14
    assert resid.shape[0] <= 5


def test_quantiles_ordered():
    resid = np.random.default_rng(0).normal(0, 2, size=(50, 14))
    lo, hi = residual_quantiles(resid, alpha=0.1)
    assert (lo <= hi).all()


def test_interval_lower_below_upper_and_nonneg():
    point = np.array([5.0, 3.0, 8.0])
    lo, hi = make_interval(point, np.array([-2, -2, -2]), np.array([2, 2, 2]))
    assert (lo <= hi).all()
    assert (lo >= 0).all()


def test_coverage_computes_fraction_inside():
    actual = np.array([1, 5, 9])
    lower = np.array([0, 0, 0])
    upper = np.array([2, 4, 10])   # 5 is outside [0,4]
    assert abs(coverage(actual, lower, upper) - 2/3) < 1e-9


def test_perfect_interval_full_coverage():
    actual = np.array([3, 4, 5])
    assert coverage(actual, actual, actual) == 1.0


def test_conformal_scale_at_least_one():
    resid = np.random.default_rng(1).normal(0, 2, size=(60, 14))
    lo, hi = residual_quantiles(resid, alpha=0.1)
    s = conformal_scale(resid, lo, hi, target=0.9)
    assert s >= 1.0


def test_conformal_widening_raises_coverage():
    """A larger scale factor should not reduce calibration coverage."""
    resid = np.random.default_rng(2).normal(0, 2, size=(80, 14))
    lo, hi = residual_quantiles(resid, alpha=0.1)
    cov_1 = ((resid >= 1.0*lo) & (resid <= 1.0*hi)).mean()
    cov_2 = ((resid >= 2.0*lo) & (resid <= 2.0*hi)).mean()
    assert cov_2 >= cov_1


def test_split_conformal_returns_valid_interval():
    from src.intervals import split_conformal_interval
    resid = np.random.default_rng(0).normal(0, 2, size=(40, 14))
    point = np.full(14, 5.0)
    lo, hi = split_conformal_interval(resid, point, alpha=0.1)
    assert (lo <= hi).all()
    assert (lo >= 0).all()
    assert len(lo) == 14


def test_split_conformal_wider_for_smaller_alpha():
    from src.intervals import split_conformal_interval
    resid = np.random.default_rng(1).normal(0, 2, size=(60, 10))
    point = np.full(10, 8.0)
    lo90, hi90 = split_conformal_interval(resid, point, alpha=0.1)
    lo99, hi99 = split_conformal_interval(resid, point, alpha=0.01)
    # 99% interval should be at least as wide as 90%
    assert (hi99 - lo99).mean() >= (hi90 - lo90).mean()
