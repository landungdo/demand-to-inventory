"""
Integration smoke test: the forecast -> interval -> inventory -> hierarchy chain
runs end-to-end and stays coherent, without needing the (gitignored) M5 CSVs.

This guards the wiring the reproduce script depends on: that a global model can
be fit, forecast with a future calendar, have a calibrated interval built, feed
an inventory simulation, and roll up coherently.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_series import make_panel
from src.baselines import BASELINE_FORECASTERS, MovingAverageForecaster
from src.backtest import backtest_panel
from src.global_model import GlobalForecaster
from src.metrics import rmsse_m5
from src.inventory import simulate_inventory
from src.intervals import (
    collect_residuals_by_step, residual_quantiles, make_interval,
    conformal_scale, coverage, split_conformal_interval,
)
from src.hierarchy import bottom_up, is_coherent


def test_end_to_end_pipeline_runs_and_is_coherent():
    HORIZON = 28
    panel = make_panel(n_items=8, n_stores=2, n_days=400, seed=0)
    panel["snap_CA"] = 0; panel["snap_TX"] = 0; panel["snap_WI"] = 0
    panel["state_id"] = "CA"

    results, summary = backtest_panel(panel, BASELINE_FORECASTERS,
                                      horizon=HORIZON, n_windows=2)
    assert {"mae", "rmse", "rmsse"}.issubset(summary.columns)

    cutoff = panel["date"].max() - pd.Timedelta(days=HORIZON)
    train_panel = panel[panel["date"] <= cutoff]
    test_panel = panel[panel["date"] > cutoff]
    gm = GlobalForecaster(max_iter=60).fit(train_panel)

    leaf_fc = {}
    for sid, g in test_panel.groupby("series_id"):
        hist = train_panel[train_panel["series_id"] == sid]
        gs = g.sort_values("date")
        actual = gs["sales"].to_numpy()
        tv = hist.sort_values("date")["sales"].to_numpy()

        pred = gm.forecast(hist, len(actual), future_calendar=gs)
        assert len(pred) == len(actual)
        r = rmsse_m5(actual, pred, tv)
        assert np.isfinite(r) or np.isnan(r)

        resid = collect_residuals_by_step(
            tv, lambda: MovingAverageForecaster(28), horizon=HORIZON, n_windows=3)
        lo, hi = split_conformal_interval(resid, pred, alpha=0.1)
        assert (lo <= hi).all()
        cov = coverage(actual, lo, hi)
        assert 0.0 <= cov <= 1.0

        isig = float(np.mean(hi - lo) / 3.29) if len(resid) >= 2 else float(tv.std())
        inv = simulate_inventory(actual, float(np.mean(pred)), isig,
                                 holding_cost=1.0, stockout_cost=8.0)
        assert inv["total_cost"] >= 0
        leaf_fc[sid] = pred

    bu = bottom_up(leaf_fc)
    assert is_coherent(bu)
