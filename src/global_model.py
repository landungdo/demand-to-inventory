"""
Global gradient-boosted forecaster.

One model is trained on features from ALL series pooled together, then used to
forecast each series `horizon` days ahead. Multi-step forecasts are produced
recursively: predict day t+1, append it to the history, recompute lags, predict
t+2, and so on. This keeps the model causal at every step.

The model plugs into the same backtest harness as the baselines, but because it
needs the whole panel (not a single array) it exposes a panel-level interface:
`GlobalForecaster.fit(train_panel)` then `.forecast(series_history, horizon)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import (
    make_features, add_features, FEATURE_COLS, LAGS, ROLL_WINDOWS,
)


class GlobalForecaster:
    """Gradient-boosted regressor trained across all series."""

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.05,
                 max_depth: int = 6, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state,
        )

    def fit(self, train_panel: pd.DataFrame, value_col: str = "sales",
            id_col: str = "series_id") -> "GlobalForecaster":
        X, y, _ = make_features(train_panel, value_col=value_col, id_col=id_col)
        self.model.fit(X, y)
        self._value_col = value_col
        self._id_col = id_col
        return self

    def forecast(self, series_history: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Recursively forecast one series `horizon` days ahead.

        series_history must contain the columns needed by add_features
        (date, sales, wday/month optional, snap columns/state optional) up to the
        cutoff. Predicted values are appended and features recomputed each step.
        """
        hist = series_history.sort_values("date").copy()
        vcol = self._value_col
        preds = []
        last_date = hist["date"].max()

        for step in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(days=step)
            new_row = {
                "series_id": hist["series_id"].iloc[0] if "series_id" in hist else "s",
                "date": next_date,
                vcol: np.nan,
            }
            # Carry state_id and snap columns forward if present
            for c in ("state_id", "store_id", "item_id",
                      "snap_CA", "snap_TX", "snap_WI"):
                if c in hist.columns:
                    new_row[c] = hist[c].iloc[-1]
            work = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
            feat = add_features(work, value_col=vcol)
            x_row = feat[FEATURE_COLS].iloc[[-1]].astype(float)
            # If lags are unavailable (very short history), fall back to last mean
            if x_row.isna().any(axis=1).iloc[0]:
                yhat = float(np.nanmean(hist[vcol].to_numpy()[-28:]))
            else:
                yhat = float(self.model.predict(x_row)[0])
            yhat = max(0.0, yhat)  # demand is non-negative
            preds.append(yhat)
            # Append the prediction as the realized value for the next step
            appended = new_row.copy()
            appended[vcol] = yhat
            hist = pd.concat([hist, pd.DataFrame([appended])], ignore_index=True)

        return np.array(preds)


if __name__ == "__main__":
    from src.synthetic_series import make_panel
    from src.metrics import evaluate_forecast

    panel = make_panel(n_items=10, n_stores=2, n_days=500, seed=0)
    # Split: last 28 days as test, everything before as train
    cutoff = panel["date"].max() - pd.Timedelta(days=28)
    train_panel = panel[panel["date"] <= cutoff]
    test_panel = panel[panel["date"] > cutoff]

    model = GlobalForecaster(max_iter=150).fit(train_panel)

    # Evaluate on a few series
    rmsses = []
    for sid, g in test_panel.groupby("series_id"):
        hist = train_panel[train_panel["series_id"] == sid]
        actual = g.sort_values("date")["sales"].to_numpy()
        pred = model.forecast(hist, horizon=len(actual))
        train_vals = hist.sort_values("date")["sales"].to_numpy()
        m = evaluate_forecast(actual, pred, train_vals, period=7)
        rmsses.append(m["rmsse"])
    print("Global gradient-boosted forecaster (synthetic panel)")
    print(f"  series evaluated: {len(rmsses)}")
    print(f"  mean RMSSE: {np.nanmean(rmsses):.3f}")
    print("  (compare against the baselines' RMSSE from backtest.py)")
