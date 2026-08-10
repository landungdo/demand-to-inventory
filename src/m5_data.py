"""
M5 data pipeline: load, subset, reshape to a tidy long panel, join calendar.

The raw M5 sales file is "wide": one row per item x store, one column per day
(d_1 ... d_1941). Forecasting code wants a "long tidy" panel: one row per
(series, date) with the sales value and calendar features attached.

This module:
  1. loads the raw sales, calendar (and optionally prices) CSVs,
  2. selects a configurable subset (by category and store) so we can iterate
     fast while keeping the pipeline able to scale to the full 30k series,
  3. melts wide -> long and attaches real dates + calendar features,
  4. runs a short data audit (series count, date span, zero-sales share, etc.).

The subset is parameterised on purpose: the scope calls for a pipeline that can
scale, not a hand-picked set of easy series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
SALES_FILE = "sales_train_evaluation.csv"   # 1941 days, the fuller of the two
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

# Calendar feature columns worth carrying into the panel
CALENDAR_FEATURES = [
    "date", "d", "wday", "month", "year",
    "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
]


def load_raw(data_dir: Path = DATA_DIR):
    """Load the raw sales and calendar frames."""
    sales = pd.read_csv(data_dir / SALES_FILE)
    calendar = pd.read_csv(data_dir / CALENDAR_FILE)
    return sales, calendar


def build_panel(cat_id: str = "FOODS",
                dept_id: str | None = "FOODS_3",
                store_ids=("CA_1", "CA_2", "CA_3"),
                data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Build a tidy long panel for a configurable subset.

    cat_id    : category to keep (e.g. "FOODS", "HOBBIES", "HOUSEHOLD")
    dept_id   : optional finer department filter (e.g. "FOODS_3"); None = all
                departments within the category
    store_ids : which stores to keep
    Returns columns:
      id, item_id, dept_id, cat_id, store_id, state_id, d, date, sales,
      wday, month, year, event_name_1, event_type_1, snap_CA/TX/WI
    """
    sales, calendar = load_raw(data_dir)

    mask = (sales["cat_id"] == cat_id) & (sales["store_id"].isin(store_ids))
    if dept_id is not None:
        mask &= (sales["dept_id"] == dept_id)
    sub = sales[mask].copy()
    if sub.empty:
        raise ValueError(f"No rows for cat={cat_id}, dept={dept_id}, stores={store_ids}")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sub.columns if c.startswith("d_")]

    long = sub.melt(id_vars=id_cols, value_vars=day_cols,
                    var_name="d", value_name="sales")

    # Attach real dates + calendar features
    cal = calendar[CALENDAR_FEATURES].copy()
    long = long.merge(cal, on="d", how="left")
    long["date"] = pd.to_datetime(long["date"])
    long["sales"] = long["sales"].astype("int32")

    # A stable per-series key and chronological order
    long["series_id"] = long["store_id"].astype(str) + "__" + long["item_id"].astype(str)
    long = long.sort_values(["series_id", "date"]).reset_index(drop=True)
    return long


def audit(panel: pd.DataFrame) -> dict:
    """Short data-quality audit of a tidy panel."""
    n_series = panel["series_id"].nunique()
    per_series_len = panel.groupby("series_id").size()
    zero_share = float((panel["sales"] == 0).mean())
    return {
        "n_series": int(n_series),
        "n_rows": int(len(panel)),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "days_per_series_min": int(per_series_len.min()),
        "days_per_series_max": int(per_series_len.max()),
        "zero_sales_share": zero_share,
        "mean_daily_sales": float(panel["sales"].mean()),
        "median_daily_sales": float(panel["sales"].median()),
        "any_missing_dates": bool(per_series_len.nunique() > 1),
    }


if __name__ == "__main__":
    panel = build_panel()
    info = audit(panel)
    print("M5 subset panel — data audit\n")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.3f}")
        else:
            print(f"  {k:22s}: {v}")
    print("\nSample rows:")
    cols = ["series_id", "date", "sales", "wday", "snap_CA", "event_name_1"]
    print(panel[cols].head(5).to_string(index=False))
