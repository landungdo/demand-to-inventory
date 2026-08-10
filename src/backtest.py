"""
Rolling-origin backtesting.

A single train/test split gives one noisy accuracy number and is easy to get
lucky (or unlucky) on. Rolling-origin evaluation (a.k.a. time-series
cross-validation) slides the forecast origin forward through several cutoffs,
forecasts the next `horizon` days at each, and averages the errors. Crucially it
never trains on data after the cutoff — the forecasting analogue of the
out-of-time split used in the credit and uplift projects.

    |-------- train --------|== horizon ==|                 cutoff 1
       |-------- train --------|== horizon ==|              cutoff 2
          |-------- train --------|== horizon ==|           cutoff 3

Given a long tidy panel (series_id, date, sales), `backtest_panel` runs every
forecaster in a registry across every series and every cutoff, and returns a
tidy results frame plus per-method aggregates. `rolling_origin_splits` exposes
the split indices so the no-leakage property can be unit-tested directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import evaluate_forecast


def rolling_origin_splits(n: int, horizon: int, n_windows: int,
                          step: int = None, min_train: int = None):
    """
    Yield (train_end, test_start, test_end) index triples for rolling origins.

    n         : length of the series
    horizon   : forecast length at each origin
    n_windows : number of origins
    step      : gap between successive origins (default = horizon)
    min_train : minimum training length before the first origin

    The last window ends at n; earlier windows step back by `step`.
    train is always [0, train_end); test is [test_start, test_end) with
    test_start == train_end (no gap, no overlap, no peeking).
    """
    step = step or horizon
    cutoffs = []
    end = n
    for _ in range(n_windows):
        test_end = end
        test_start = test_end - horizon
        train_end = test_start
        if min_train is not None and train_end < min_train:
            break
        if train_end <= 0:
            break
        cutoffs.append((train_end, test_start, test_end))
        end -= step
    return list(reversed(cutoffs))


def backtest_series(sales: np.ndarray, forecaster_factory, horizon: int,
                    n_windows: int, period: int = 7, step: int = None,
                    min_train: int = None):
    """Backtest one forecaster on one series; return a list of per-window dicts."""
    sales = np.asarray(sales, dtype=float)
    splits = rolling_origin_splits(len(sales), horizon, n_windows,
                                   step=step, min_train=min_train)
    rows = []
    for (train_end, test_start, test_end) in splits:
        train = sales[:train_end]
        actual = sales[test_start:test_end]
        f = forecaster_factory().fit(train)
        pred = f.predict(horizon)
        m = evaluate_forecast(actual, pred, train, period=period)
        m["train_end"] = int(train_end)
        rows.append(m)
    return rows


def backtest_panel(panel: pd.DataFrame, forecasters: dict, horizon: int = 28,
                   n_windows: int = 3, period: int = 7,
                   value_col: str = "sales", id_col: str = "series_id"):
    """
    Run every forecaster across every series in the panel with rolling origins.

    Returns (results_df, summary_df):
      results_df : one row per (series, method, window) with metrics
      summary_df : mean metrics per method across all series and windows
    """
    records = []
    for series_id, g in panel.groupby(id_col):
        sales = g.sort_values("date")[value_col].to_numpy()
        for method, factory in forecasters.items():
            for r in backtest_series(sales, factory, horizon, n_windows, period=period):
                r["series_id"] = series_id
                r["method"] = method
                records.append(r)
    results = pd.DataFrame.from_records(records)

    summary = (results.groupby("method")[["mae", "rmse", "rmsse"]]
               .mean().reset_index()
               .sort_values("rmsse").reset_index(drop=True))
    return results, summary


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.synthetic_series import make_panel
    from src.baselines import BASELINE_FORECASTERS

    # Synthetic panel so the demo runs without the M5 files
    panel = make_panel(n_items=8, n_stores=2, n_days=400, seed=0)
    results, summary = backtest_panel(panel, BASELINE_FORECASTERS,
                                      horizon=28, n_windows=3)
    print("Rolling-origin backtest (synthetic panel, 16 series, 3 windows)\n")
    print(summary.to_string(index=False))
    print(f"\nEvaluations: {len(results)} "
          f"({results['series_id'].nunique()} series x "
          f"{results['method'].nunique()} methods x 3 windows)")
    print("Lower RMSSE is better; the winning baseline depends on the demand shape.")
