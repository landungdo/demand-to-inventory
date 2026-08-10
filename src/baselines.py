"""
Baseline forecasters for demand.

Strong baselines are non-negotiable in forecasting: on intermittent retail
demand (here ~54% zero-sales days), simple methods are surprisingly hard to
beat, and many "sophisticated" models quietly lose to them. Reporting a model
without a seasonal-naive baseline is how forecasting projects fool themselves.

Each forecaster takes a 1-D history array and a horizon h, and returns an
h-step-ahead forecast using ONLY the history — no peeking at future values.

  NaiveForecaster          : repeat the last observed value.
  SeasonalNaiveForecaster  : repeat the value from one season ago (period=7 for
                             weekly retail seasonality), tiling across horizon.
  MovingAverageForecaster  : forecast the mean of the last `window` observations,
                             flat across the horizon. Robust for low-count /
                             intermittent series.

All are point forecasters; prediction intervals are added later. The interface
(fit/predict on a single series) is intentionally simple so the backtester can
call any of them uniformly.
"""

from __future__ import annotations

import numpy as np


class NaiveForecaster:
    """Forecast = last observed value, repeated across the horizon."""

    def fit(self, history: np.ndarray) -> "NaiveForecaster":
        self._last = float(np.asarray(history)[-1])
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.full(h, self._last)


class SeasonalNaiveForecaster:
    """Forecast = value from one season ago (default weekly, period=7)."""

    def __init__(self, period: int = 7):
        self.period = period

    def fit(self, history: np.ndarray) -> "SeasonalNaiveForecaster":
        self._history = np.asarray(history, dtype=float)
        return self

    def predict(self, h: int) -> np.ndarray:
        hist = self._history
        p = self.period
        if len(hist) < p:
            # Not enough history for a full season: fall back to the last value
            return np.full(h, hist[-1])
        # The last full season, tiled to cover the horizon
        last_season = hist[-p:]
        reps = int(np.ceil(h / p))
        return np.tile(last_season, reps)[:h]


class MovingAverageForecaster:
    """Forecast = mean of the last `window` observations, flat across horizon."""

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, history: np.ndarray) -> "MovingAverageForecaster":
        hist = np.asarray(history, dtype=float)
        w = min(self.window, len(hist))
        self._mean = float(hist[-w:].mean())
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.full(h, self._mean)


# Registry so the backtester and reports can iterate uniformly
BASELINE_FORECASTERS = {
    "naive": lambda: NaiveForecaster(),
    "seasonal_naive": lambda: SeasonalNaiveForecaster(period=7),
    "moving_avg_28": lambda: MovingAverageForecaster(window=28),
}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.synthetic_series import make_series

    # A weekly-seasonal series: seasonal-naive should track it best
    series = make_series(n_days=140, seed=1, base=10, weekly_amp=6,
                         noise=0.0, intermittent=0.0)
    train, test = series[:-14], series[-14:]

    print("14-day-ahead forecast on a weekly-seasonal series\n")
    print(f"{'method':<18} {'MAE':>8}")
    print("-" * 28)
    for name, make in BASELINE_FORECASTERS.items():
        f = make().fit(train)
        pred = f.predict(14)
        mae = np.mean(np.abs(pred - test))
        print(f"{name:<18} {mae:>8.3f}")
    print("\nOn a purely seasonal series, seasonal_naive should win clearly.")
