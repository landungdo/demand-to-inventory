"""Tests for forecast metrics (MAE, RMSE, RMSSE)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import mae, rmse, rmsse, evaluate_forecast, mean_rmsse


def test_perfect_forecast_zero_error():
    y = np.array([1, 2, 3, 4])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_mae_and_rmse_values():
    y_true = np.array([0, 0, 10])
    y_pred = np.array([0, 0, 7])
    assert abs(mae(y_true, y_pred) - 1.0) < 1e-9        # (0+0+3)/3
    assert abs(rmse(y_true, y_pred) - np.sqrt(3.0)) < 1e-9  # sqrt(9/3)


def test_rmsse_less_than_one_when_beating_naive():
    # Train has big day-to-day swings (large naive scale); forecast is near-perfect
    y_train = np.array([0, 10, 0, 10, 0, 10, 0, 10], dtype=float)
    y_true = np.array([5, 5, 5])
    y_pred = np.array([5, 5, 5])   # perfect -> RMSSE 0
    assert rmsse(y_true, y_pred, y_train, period=1) == 0.0


def test_rmsse_scales_by_training_volatility():
    y_true = np.array([5, 5])
    y_pred = np.array([6, 6])       # constant error of 1
    calm = np.array([5, 5, 5, 5, 5], dtype=float)     # zero naive scale -> epsilon
    wild = np.array([0, 10, 0, 10, 0], dtype=float)   # large naive scale
    r_calm = rmsse(y_true, y_pred, calm, period=1)
    r_wild = rmsse(y_true, y_pred, wild, period=1)
    assert r_calm > r_wild          # same error is "worse" on a calm series


def test_mean_rmsse_ignores_nan_and_weights():
    vals = [1.0, 2.0, np.nan]
    assert abs(mean_rmsse(vals) - 1.5) < 1e-9
    # Weighted: put all weight on the 2.0
    assert abs(mean_rmsse([1.0, 2.0], weights=[0, 1]) - 2.0) < 1e-9
