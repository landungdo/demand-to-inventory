"""
Tests for the synthetic time-series generator (used to validate forecasting
logic against series with known structure).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_series import make_series, make_panel


def test_series_length_and_nonneg():
    s = make_series(n_days=365, seed=0)
    assert len(s) == 365
    assert (s >= 0).all()


def test_trend_increases_level():
    up = make_series(n_days=400, seed=1, base=5, trend=0.02, noise=0.0, intermittent=0.0)
    # Later part should have higher mean than earlier part
    assert up[-100:].mean() > up[:100].mean()


def test_weekly_seasonality_has_period_7_structure():
    s = make_series(n_days=700, seed=2, base=10, weekly_amp=5, noise=0.0, intermittent=0.0)
    # Average by weekday index should not be flat
    by_wday = np.array([s[i::7].mean() for i in range(7)])
    assert by_wday.std() > 0.5


def test_intermittent_creates_zeros():
    s = make_series(n_days=500, seed=3, base=5, intermittent=0.4)
    zero_frac = (s == 0).mean()
    assert zero_frac > 0.2


def test_panel_shape_and_columns():
    panel = make_panel(n_items=4, n_stores=2, n_days=100, seed=0)
    assert panel["item_id"].nunique() == 4
    assert panel["store_id"].nunique() == 2
    assert set(["item_id", "store_id", "date", "day_index", "sales"]).issubset(panel.columns)
    # One row per item x store x day
    assert len(panel) == 4 * 2 * 100
