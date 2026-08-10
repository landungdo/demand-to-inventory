"""
Synthetic intermittent-demand series generator — for TESTING ONLY.

Unlike the uplift project, the synthetic data here is *not* a headline artifact:
the real M5 dataset is the centrepiece. This generator exists so unit tests can
check forecasting and backtesting logic against series whose trend, seasonality,
and noise are known by construction — e.g. that a seasonal-naive baseline nails a
purely seasonal series, or that a backtest never peeks into the future.

Retail demand at the item x store level is typically:
  - intermittent (many zero-sales days),
  - low-count (small integers),
  - weekly-seasonal (weekend peaks),
  - sometimes trending or promotion-driven.

`make_series` builds one such series with controllable components; `make_panel`
stacks several into a long tidy frame shaped like the M5 data we will load.
"""

import numpy as np
import pandas as pd


def make_series(n_days: int = 730, seed: int = 0,
                base: float = 5.0, trend: float = 0.0,
                weekly_amp: float = 0.0, intermittent: float = 0.0,
                noise: float = 0.3) -> np.ndarray:
    """
    Generate one daily demand series of length n_days.

    base         : average daily demand level
    trend        : additive change in level per day (can be negative)
    weekly_amp   : amplitude of a weekly (period-7) seasonal cycle
    intermittent : probability a given day is forced to zero demand
    noise        : multiplicative noise scale (Gamma-ish via clipping)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    level = base + trend * t
    seasonal = weekly_amp * np.sin(2 * np.pi * (t % 7) / 7)
    mean = np.clip(level + seasonal, 0.01, None)
    demand = rng.poisson(mean * (1 + noise * rng.standard_normal(n_days)).clip(0.1))
    if intermittent > 0:
        zero_mask = rng.random(n_days) < intermittent
        demand = demand * (~zero_mask)
    return demand.astype(int)


def make_panel(n_items: int = 5, n_stores: int = 2, n_days: int = 730,
               seed: int = 0) -> pd.DataFrame:
    """
    Build a long tidy panel of item x store daily demand, shaped like the M5
    data we will load later: columns [item_id, store_id, date, day_index, sales].
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2013-01-01")
    dates = start + pd.to_timedelta(np.arange(n_days), unit="D")

    rows = []
    s = 0
    for i in range(n_items):
        for st in range(n_stores):
            s += 1
            series = make_series(
                n_days=n_days, seed=seed + s,
                base=rng.uniform(2, 12),
                trend=rng.uniform(-0.005, 0.01),
                weekly_amp=rng.uniform(0, 3),
                intermittent=rng.uniform(0, 0.3),
                noise=0.3,
            )
            rows.append(pd.DataFrame({
                "item_id": f"ITEM_{i:03d}",
                "store_id": f"STORE_{st+1}",
                "date": dates,
                "day_index": np.arange(n_days),
                "sales": series,
            }))
    result = pd.concat(rows, ignore_index=True)
    result["series_id"] = (result["store_id"].astype(str) + "__"
                           + result["item_id"].astype(str))
    return result


if __name__ == "__main__":
    panel = make_panel()
    print("Synthetic demand panel (for testing)")
    print(f"  series: {panel.groupby(['item_id','store_id']).ngroups}")
    print(f"  rows:   {len(panel)}")
    print(f"  date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"  overall mean daily sales: {panel['sales'].mean():.2f}")
    print(f"  zero-sales fraction: {(panel['sales'] == 0).mean():.1%}")
    print()
    # Show a purely seasonal series recovers its period
    s = make_series(n_days=70, seed=1, base=10, weekly_amp=5, noise=0.0, intermittent=0.0)
    print("Purely weekly-seasonal sample (first 14 days):")
    print(" ", s[:14].tolist())
