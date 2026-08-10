"""
Tests for the global gradient-boosted forecaster.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_series import make_panel
from src.global_model import GlobalForecaster


def _split(panel, test_days=28):
    cutoff = panel["date"].max() - pd.Timedelta(days=test_days)
    return panel[panel["date"] <= cutoff], panel[panel["date"] > cutoff]


def test_forecast_shape_and_nonneg():
    panel = make_panel(n_items=5, n_stores=2, n_days=300, seed=0)
    train, test = _split(panel)
    gm = GlobalForecaster(max_iter=80).fit(train)
    sid = panel["series_id"].iloc[0]
    hist = train[train["series_id"] == sid]
    pred = gm.forecast(hist, horizon=28)
    assert pred.shape == (28,)
    assert (pred >= 0).all()          # demand can't be negative
    assert np.isfinite(pred).all()


def test_fit_learns_something_better_than_zero():
    """The model should beat forecasting all-zeros on a non-trivial series."""
    panel = make_panel(n_items=6, n_stores=2, n_days=400, seed=2)
    train, test = _split(panel)
    gm = GlobalForecaster(max_iter=120).fit(train)
    errs_model, errs_zero = [], []
    for sid, g in test.groupby("series_id"):
        hist = train[train["series_id"] == sid]
        actual = g.sort_values("date")["sales"].to_numpy()
        pred = gm.forecast(hist, len(actual))
        errs_model.append(np.mean(np.abs(actual - pred)))
        errs_zero.append(np.mean(np.abs(actual - 0)))
    assert np.mean(errs_model) < np.mean(errs_zero)
