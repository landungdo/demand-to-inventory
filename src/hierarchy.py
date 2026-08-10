"""
Hierarchical forecast reconciliation.

Retail demand is hierarchical: items roll up into stores, stores into a category
total. If we forecast each level independently, the numbers won't agree — the
item forecasts won't sum to the store forecast. Reconciliation makes a set of
forecasts *coherent* (children sum to their parent) so downstream planning at
different levels tells one consistent story.

Two standard approaches implemented here:

  Bottom-up
    Forecast the bottom level (item x store), then obtain every parent by
    summation. Coherent by construction; ignores any signal in aggregate series.

  Top-down
    Forecast the top level (the total), then split it down to the leaves using
    historical proportions (each leaf's share of the total). Good when the
    aggregate is easier to forecast than noisy leaves.

Both take a leaf-level structure and return coherent forecasts at leaf and
aggregate levels. The proportions for top-down are estimated from history and
are guaranteed to sum to 1, so coherence holds exactly.
"""

from __future__ import annotations

import numpy as np


def bottom_up(leaf_forecasts: dict) -> dict:
    """
    Bottom-up reconciliation.

    leaf_forecasts : {leaf_id: np.ndarray(horizon)}
    Returns {"leaves": {...}, "total": np.ndarray} where total is the leaf sum.
    """
    leaves = {k: np.asarray(v, dtype=float) for k, v in leaf_forecasts.items()}
    total = np.sum(list(leaves.values()), axis=0)
    return {"leaves": leaves, "total": total}


def historical_proportions(leaf_history: dict) -> dict:
    """
    Each leaf's share of the total, from historical sums. Shares sum to 1.
    leaf_history : {leaf_id: np.ndarray(history)}
    """
    sums = {k: float(np.sum(v)) for k, v in leaf_history.items()}
    grand = sum(sums.values())
    if grand == 0:
        n = len(sums)
        return {k: 1.0 / n for k in sums}
    return {k: s / grand for k, s in sums.items()}


def top_down(total_forecast: np.ndarray, leaf_history: dict) -> dict:
    """
    Top-down reconciliation: split a total forecast across leaves by their
    historical proportions. Coherent by construction.
    """
    total_forecast = np.asarray(total_forecast, dtype=float)
    props = historical_proportions(leaf_history)
    leaves = {k: total_forecast * p for k, p in props.items()}
    return {"leaves": leaves, "total": total_forecast, "proportions": props}


def is_coherent(reconciled: dict, tol: float = 1e-6) -> bool:
    """Check that leaf forecasts sum to the stated total at every horizon step."""
    leaf_sum = np.sum(list(reconciled["leaves"].values()), axis=0)
    return bool(np.allclose(leaf_sum, reconciled["total"], atol=tol))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.synthetic_series import make_panel
    from src.baselines import MovingAverageForecaster

    # Build a small hierarchy: several item x store leaves under one total
    panel = make_panel(n_items=6, n_stores=2, n_days=400, seed=0)
    horizon = 14

    leaf_hist, leaf_fc = {}, {}
    for sid, g in panel.groupby("series_id"):
        sales = g.sort_values("date")["sales"].to_numpy()
        train = sales[:-horizon]
        leaf_hist[sid] = train
        leaf_fc[sid] = MovingAverageForecaster(28).fit(train).predict(horizon)

    bu = bottom_up(leaf_fc)
    print("Bottom-up reconciliation")
    print(f"  leaves: {len(bu['leaves'])}, total[day0] = {bu['total'][0]:.1f}")
    print(f"  coherent: {is_coherent(bu)}")

    # Top-down: forecast the total directly, split by history
    total_hist = np.sum(list(leaf_hist.values()), axis=0)
    total_fc = MovingAverageForecaster(28).fit(total_hist).predict(horizon)
    td = top_down(total_fc, leaf_hist)
    print("\nTop-down reconciliation")
    print(f"  total[day0] = {td['total'][0]:.1f}, split across {len(td['leaves'])} leaves")
    print(f"  proportions sum to {sum(td['proportions'].values()):.3f}")
    print(f"  coherent: {is_coherent(td)}")
    print("\nBoth are coherent by construction: leaf forecasts sum to the total.")
