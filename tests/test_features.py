"""
Tests for feature engineering — the load-bearing property is no leakage:
a feature for day t must use only data strictly before t.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_series import make_panel
from src.features import add_features, make_features, FEATURE_COLS, LAGS


def test_lag_features_use_only_past():
    panel = make_panel(n_items=1, n_stores=1, n_days=100, seed=0)
    feat = add_features(panel)
    g = feat.sort_values("date").reset_index(drop=True)
    # lag_7 at row i should equal sales at row i-7
    for i in range(7, len(g)):
        if not np.isnan(g.loc[i, "lag_7"]):
            assert g.loc[i, "lag_7"] == g.loc[i - 7, "sales"]


def test_rolling_mean_excludes_current_day():
    panel = make_panel(n_items=1, n_stores=1, n_days=100, seed=1)
    feat = add_features(panel).sort_values("date").reset_index(drop=True)
    i = 50
    # roll_mean_7 at i is the mean of sales[i-7:i] (ending the day before i)
    expected = feat.loc[i - 7:i - 1, "sales"].mean()
    assert abs(feat.loc[i, "roll_mean_7"] - expected) < 1e-9


def test_make_features_drops_unavailable_lags():
    panel = make_panel(n_items=2, n_stores=1, n_days=60, seed=0)
    X, y, meta = make_features(panel)
    # No NaNs in the feature matrix after dropping
    assert not X.isna().any().any()
    assert len(X) == len(y) == len(meta)
    assert list(X.columns) == FEATURE_COLS


def test_feature_matrix_has_expected_columns():
    panel = make_panel(n_items=2, n_stores=2, n_days=120, seed=3)
    X, _, _ = make_features(panel)
    for l in LAGS:
        assert f"lag_{l}" in X.columns
