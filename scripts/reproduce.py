"""
Reproduce the full demand-to-inventory pipeline on the M5 subset.

One command runs the project end to end on real M5 data and writes the headline
tables as artifacts:

  results/backtest_summary.csv  - RMSSE/MAE/RMSE per method (rolling backtest)
  results/inventory_summary.csv - inventory cost per forecasting method
  results/metrics.json          - key numbers, including the accuracy-vs-cost
                                  comparison that is the project's main point

Because the raw M5 CSVs are large and gitignored, this script requires them in
data/ (see README). It uses a configurable subset so it runs in minutes, and the
same code scales to more series by widening the subset in src/m5_data.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.m5_data import build_panel, audit
from src.baselines import BASELINE_FORECASTERS
from src.backtest import backtest_panel
from src.global_model import GlobalForecaster
from src.metrics import evaluate_forecast
from src.inventory import simulate_inventory

OUTDIR = Path("results")
HORIZON = 28
N_WINDOWS = 3
# Keep the reproduce run quick: one store, a capped number of series.
MAX_SERIES = 100


def main():
    OUTDIR.mkdir(exist_ok=True)

    # 1. Load a subset and audit it
    panel = build_panel(cat_id="FOODS", dept_id="FOODS_3", store_ids=("CA_1",))
    info = audit(panel)
    # Cap the number of series for a fast, reproducible run
    keep = sorted(panel["series_id"].unique())[:MAX_SERIES]
    panel = panel[panel["series_id"].isin(keep)].reset_index(drop=True)

    # Hold out the final `HORIZON` days as an untouched test set. Everything
    # before it is used to (a) backtest baselines and (b) pick the best baseline,
    # so the final holdout never influences model/method selection.
    cutoff = panel["date"].max() - pd.Timedelta(days=HORIZON)
    train_panel = panel[panel["date"] <= cutoff]
    test_panel = panel[panel["date"] > cutoff]

    # 2. Backtest the baselines on the PRE-HOLDOUT data only (rolling origin).
    results, summary = backtest_panel(train_panel, BASELINE_FORECASTERS,
                                      horizon=HORIZON, n_windows=N_WINDOWS)
    summary.to_csv(OUTDIR / "backtest_summary.csv", index=False)

    # 3. Global model + best baseline (selected from the pre-holdout backtest,
    #    not from the final holdout) evaluated once on the untouched holdout.
    gm = GlobalForecaster(max_iter=150).fit(train_panel)

    gm_rmsse, best_base_rmsse = [], []
    best_base = summary.iloc[0]["method"]
    base_factory = BASELINE_FORECASTERS[best_base]

    # For the accuracy-vs-cost story, also accumulate inventory cost per method
    inv_rows = []
    for sid, g in test_panel.groupby("series_id"):
        hist = train_panel[train_panel["series_id"] == sid]
        g_sorted = g.sort_values("date")
        actual = g_sorted["sales"].to_numpy()
        train_vals = hist.sort_values("date")["sales"].to_numpy()
        sigma = float(train_vals.std())

        # Pass the real future calendar (dates + SNAP) so the global model uses
        # correct future features instead of carrying the last day forward.
        gm_pred = gm.forecast(hist, len(actual), future_calendar=g_sorted)
        base_pred = base_factory().fit(train_vals).predict(len(actual))

        gm_rmsse.append(evaluate_forecast(actual, gm_pred, train_vals, period=7)["rmsse"])
        best_base_rmsse.append(evaluate_forecast(actual, base_pred, train_vals, period=7)["rmsse"])

        for name, pred in [("global", gm_pred), (best_base, base_pred)]:
            fc_daily = float(np.mean(pred))
            r = simulate_inventory(actual, fc_daily, sigma, z=1.04,
                                   holding_cost=1.0, stockout_cost=8.0)
            inv_rows.append({"series_id": sid, "method": name,
                             "total_cost": r["total_cost"],
                             "fill_rate": r["fill_rate"]})

    inv = pd.DataFrame(inv_rows)
    inv_summary = inv.groupby("method")[["total_cost", "fill_rate"]].mean().reset_index()
    inv_summary.to_csv(OUTDIR / "inventory_summary.csv", index=False)

    metrics = {
        "audit": info,
        "n_series_used": int(panel["series_id"].nunique()),
        "backtest_rmsse": {r["method"]: float(r["rmsse"]) for _, r in summary.iterrows()},
        "global_rmsse_mean": float(np.nanmean(gm_rmsse)),
        "best_baseline": best_base,
        "best_baseline_rmsse_mean": float(np.nanmean(best_base_rmsse)),
        "inventory_cost": {r["method"]: float(r["total_cost"])
                           for _, r in inv_summary.iterrows()},
    }
    with open(OUTDIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary
    print("Demand-to-inventory pipeline — M5 subset\n")
    print(f"Series used: {metrics['n_series_used']}  "
          f"(zero-sales share {info['zero_sales_share']:.0%})\n")
    print("Backtest RMSSE (rolling origin, lower better):")
    print(summary.to_string(index=False))
    print(f"\nGlobal model mean RMSSE:   {metrics['global_rmsse_mean']:.3f}")
    print(f"Best baseline ({best_base}) mean RMSSE: {metrics['best_baseline_rmsse_mean']:.3f}")
    print("\nInventory cost per method (holding=1, stockout=8):")
    print(inv_summary.to_string(index=False))
    print("\nKey point: compare the RMSSE ranking with the inventory-cost ranking —")
    print("the most accurate forecaster is not always the cheapest policy.")


if __name__ == "__main__":
    main()
