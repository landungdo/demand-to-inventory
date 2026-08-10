"""
Feature engineering for a global forecasting model.

A *global* model trains one learner across all series at once, so it can borrow
strength across items and stores — far better than fitting 2,469 tiny per-series
models. The learner is a plain gradient-boosted regressor; the intelligence is
in the features.

All features are causal: every feature for day t uses only information available
strictly before t. The lag and rolling-window features are computed per series
and then shifted so no target leaks into its own predictors.

Feature groups:
  - Lags: sales at t-7, t-14, t-28 (recent history at weekly offsets).
  - Rolling means: mean sales over the previous 7 / 28 days (shifted by 1).
  - Calendar: weekday, month, and the SNAP flag for the store's state.

`make_features` returns a design matrix aligned to a target vector, with the
rows where any required lag is unavailable dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [7, 14, 28]
ROLL_WINDOWS = [7, 28]
FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + ["wday", "month", "snap"]
)


def _state_snap_column(df: pd.DataFrame) -> pd.Series:
    """Pick the SNAP flag matching each row's state (CA/TX/WI)."""
    if "state_id" not in df.columns:
        return pd.Series(0, index=df.index)
    snap = pd.Series(0, index=df.index, dtype=float)
    for state in ("CA", "TX", "WI"):
        col = f"snap_{state}"
        if col in df.columns:
            mask = df["state_id"] == state
            snap.loc[mask] = df.loc[mask, col].to_numpy()
    return snap


def add_features(panel: pd.DataFrame, value_col: str = "sales",
                 id_col: str = "series_id") -> pd.DataFrame:
    """
    Add causal lag / rolling / calendar features to a tidy long panel.

    Returns the panel with feature columns added; rows lacking the longest lag
    are kept as NaN here and dropped in `make_features`.
    """
    df = panel.sort_values([id_col, "date"]).copy()
    g = df.groupby(id_col)[value_col]

    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    for w in ROLL_WINDOWS:
        # shift(1) first so the window ends the day BEFORE the target (no leak)
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)

    # Calendar features — always recomputed from the date so future rows (whose
    # wday/month may be missing or NaN) get correct values. Recomputing is cheap
    # and avoids the bug where an existing-but-NaN column silently disables the
    # feature and forces a fallback.
    df["wday"] = df["date"].dt.weekday + 1
    df["month"] = df["date"].dt.month
    df["snap"] = _state_snap_column(df)
    return df


def make_features(panel: pd.DataFrame, value_col: str = "sales",
                  id_col: str = "series_id"):
    """
    Build (X, y, meta) for modelling.

    Returns:
      X    : DataFrame of FEATURE_COLS
      y    : target Series (sales)
      meta : DataFrame with series_id and date for each row (for splitting)
    """
    df = add_features(panel, value_col=value_col, id_col=id_col)
    needed = [f"lag_{max(LAGS)}"] + [f"roll_mean_{max(ROLL_WINDOWS)}"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    X = df[FEATURE_COLS].astype(float)
    y = df[value_col].astype(float)
    meta = df[[id_col, "date"]].copy()
    return X, y, meta
