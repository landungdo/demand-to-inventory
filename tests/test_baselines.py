"""
Tests for baseline forecasters.

The key properties: each forecaster uses only history (no leakage), shapes are
correct, and on a purely seasonal series the seasonal-naive method beats the
plain naive method.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_series import make_series
from src.baselines import (
    NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster,
    BASELINE_FORECASTERS,
)


def test_naive_repeats_last_value():
    f = NaiveForecaster().fit(np.array([1, 2, 3, 9]))
    assert np.allclose(f.predict(3), [9, 9, 9])


def test_seasonal_naive_repeats_last_season():
    hist = np.arange(14)  # two weeks: 0..13
    f = SeasonalNaiveForecaster(period=7).fit(hist)
    pred = f.predict(7)
    # Should repeat days 7..13
    assert np.allclose(pred, np.arange(7, 14))


def test_seasonal_naive_tiles_across_long_horizon():
    hist = np.arange(14)
    f = SeasonalNaiveForecaster(period=7).fit(hist)
    pred = f.predict(10)
    assert len(pred) == 10
    # First 7 are the last season, next 3 wrap around
    assert np.allclose(pred[:7], np.arange(7, 14))
    assert np.allclose(pred[7:], np.arange(7, 10))


def test_moving_average_is_mean_of_window():
    f = MovingAverageForecaster(window=4).fit(np.array([10, 0, 0, 0, 4, 4, 4, 4]))
    assert np.allclose(f.predict(2), [4.0, 4.0])


def test_all_forecasters_output_correct_shape():
    hist = make_series(n_days=100, seed=0)
    for name, make in BASELINE_FORECASTERS.items():
        pred = make().fit(hist).predict(14)
        assert pred.shape == (14,), name
        assert np.isfinite(pred).all(), name


def test_seasonal_naive_beats_naive_on_seasonal_series():
    series = make_series(n_days=210, seed=5, base=10, weekly_amp=7,
                         noise=0.0, intermittent=0.0)
    train, test = series[:-28], series[-28:]
    sn = SeasonalNaiveForecaster(period=7).fit(train).predict(28)
    nv = NaiveForecaster().fit(train).predict(28)
    mae_sn = np.mean(np.abs(sn - test))
    mae_nv = np.mean(np.abs(nv - test))
    assert mae_sn < mae_nv
