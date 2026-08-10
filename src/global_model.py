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

    def forecast(self, series_history: pd.DataFrame, horizon: int,
                 future_calendar: pd.DataFrame = None) -> np.ndarray:
        """
        Recursively forecast one series `horizon` days ahead.

        series_history : the series' rows up to the cutoff (date, sales, and the
            snap_CA/TX/WI + state_id columns if available).
        future_calendar : optional DataFrame with one row per forecast day,
            carrying the real calendar for those days — at minimum `date`, and
            ideally `snap_CA/snap_TX/snap_WI` (and state_id). Supplying it is what
            lets the model use the correct future SNAP/weekday instead of copying
            the last observed day. If omitted, dates are generated and SNAP is
            carried forward (a documented approximation).

        wday and month are always derived from each row's date inside
        add_features, so future rows get correct calendar values and the model is
        actually invoked (no silent fallback).
        """
        hist = series_history.sort_values("date").copy()
        vcol = self._value_col
        preds = []
        last_date = hist["date"].max()

        fc = None
        if future_calendar is not None:
            fc = future_calendar.sort_values("date").reset_index(drop=True)

        for step in range(1, horizon + 1):
            if fc is not None and step - 1 < len(fc):
                cal_row = fc.iloc[step - 1]
                next_date = pd.Timestamp(cal_row["date"])
            else:
                cal_row = None
                next_date = last_date + pd.Timedelta(days=step)

            new_row = {
                "series_id": hist["series_id"].iloc[0] if "series_id" in hist else "s",
                "date": next_date,
                vcol: np.nan,
            }
            # Static identifiers carried forward
            for c in ("state_id", "store_id", "item_id"):
                if c in hist.columns:
                    new_row[c] = hist[c].iloc[-1]
            # SNAP: take the real future value from the calendar if provided,
            # otherwise carry the last observed value forward (approximation).
            for c in ("snap_CA", "snap_TX", "snap_WI"):
                if cal_row is not None and c in fc.columns:
                    new_row[c] = cal_row[c]
                elif c in hist.columns:
                    new_row[c] = hist[c].iloc[-1]

            work = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
            feat = add_features(work, value_col=vcol)
            x_row = feat[FEATURE_COLS].iloc[[-1]].astype(float)
            if x_row.isna().any(axis=1).iloc[0]:
                # Only reached when lags are genuinely unavailable (very short
                # history), not because of missing calendar features.
                yhat = float(np.nanmean(hist[vcol].to_numpy()[-28:]))
            else:
                yhat = float(self.model.predict(x_row)[0])
            yhat = max(0.0, yhat)
            preds.append(yhat)

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
