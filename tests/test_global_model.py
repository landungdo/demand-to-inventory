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


def _m5_shaped_panel(n_series=6, n_days=300, seed=0):
    """Build a panel that looks like real M5 output: has wday/month/snap columns
    already present (this is what triggered the P0 fallback bug)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=n_days)
    frames = []
    for s in range(n_series):
        sales = rng.poisson(4 + s, n_days)
        frames.append(pd.DataFrame({
            "series_id": f"CA_1__ITEM_{s:03d}",
            "item_id": f"ITEM_{s:03d}", "store_id": "CA_1", "state_id": "CA",
            "date": dates, "sales": sales,
            "wday": dates.weekday + 1, "month": dates.month,
            "snap_CA": rng.integers(0, 2, n_days),
            "snap_TX": 0, "snap_WI": 0,
        }))
    return pd.concat(frames, ignore_index=True)


def test_global_model_actually_called_on_m5_shaped_data(monkeypatch):
    """
    Regression test for the P0 bug: on M5-shaped data (wday/month already present)
    the model's predict() must actually be invoked, not silently bypassed by the
    short-history fallback.
    """
    panel = _m5_shaped_panel(n_series=6, n_days=300, seed=1)
    cutoff = panel["date"].max() - pd.Timedelta(days=28)
    train = panel[panel["date"] <= cutoff]
    gm = GlobalForecaster(max_iter=60).fit(train)

    # Count how many times the underlying model.predict is called
    calls = {"n": 0}
    orig_predict = gm.model.predict

    def counting_predict(X):
        calls["n"] += 1
        return orig_predict(X)

    monkeypatch.setattr(gm.model, "predict", counting_predict)

    sid = panel["series_id"].iloc[0]
    hist = train[train["series_id"] == sid]
    future_cal = panel[(panel["series_id"] == sid) & (panel["date"] > cutoff)]
    pred = gm.forecast(hist, horizon=28, future_calendar=future_cal)

    assert len(pred) == 28
    # The model must be called for (nearly) every step — not zero times
    assert calls["n"] >= 27, f"model.predict called only {calls['n']} times (fallback bug)"


def test_future_snap_used_from_calendar(monkeypatch):
    """The forecast should read future SNAP from the provided calendar and pass
    it into the model — verified by spying on model.predict and checking the snap
    feature column matches the future calendar's SNAP for each step."""
    panel = _m5_shaped_panel(n_series=4, n_days=200, seed=2)
    cutoff = panel["date"].max() - pd.Timedelta(days=14)
    train = panel[panel["date"] <= cutoff]
    gm = GlobalForecaster(max_iter=50).fit(train)
    sid = panel["series_id"].iloc[0]
    hist = train[train["series_id"] == sid]
    future_cal = panel[(panel["series_id"] == sid) & (panel["date"] > cutoff)].sort_values("date")

    # SNAP is the last feature column (see FEATURE_COLS order); capture it per call
    from src.features import FEATURE_COLS
    snap_idx = FEATURE_COLS.index("snap")
    seen_snap = []
    orig = gm.model.predict

    def spy(X):
        seen_snap.append(float(X.iloc[0, snap_idx]))
        return orig(X)

    monkeypatch.setattr(gm.model, "predict", spy)
    gm.forecast(hist, horizon=14, future_calendar=future_cal)

    expected_snap = future_cal["snap_CA"].to_numpy(dtype=float)
    # Each step's snap feature should match the future calendar's snap_CA
    assert len(seen_snap) >= 13
    for i, s in enumerate(seen_snap):
        assert s == expected_snap[i], f"step {i}: snap {s} != calendar {expected_snap[i]}"
