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
from src.baselines import BASELINE_FORECASTERS, MovingAverageForecaster
from src.backtest import backtest_panel
from src.global_model import GlobalForecaster
from src.metrics import evaluate_forecast, rmsse_m5
from src.inventory import simulate_inventory
from src.intervals import (
    collect_residuals_by_step, residual_quantiles, make_interval,
    conformal_scale, coverage,
)
from src.hierarchy import bottom_up, is_coherent

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

    # For the accuracy-vs-cost story, also accumulate inventory cost per method.
    # The inventory buffer uses uncertainty from a CALIBRATED prediction interval
    # (residual quantiles + split-conformal), not just the raw historical stdev,
    # so forecast -> uncertainty -> inventory is one connected pipeline.
    inv_rows = []
    coverage_records = []
    leaf_forecasts_gm = {}   # for hierarchy reconciliation
    leaf_history = {}
    for sid, g in test_panel.groupby("series_id"):
        hist = train_panel[train_panel["series_id"] == sid]
        g_sorted = g.sort_values("date")
        actual = g_sorted["sales"].to_numpy()
        train_vals = hist.sort_values("date")["sales"].to_numpy()

        gm_pred = gm.forecast(hist, len(actual), future_calendar=g_sorted)
        base_pred = base_factory().fit(train_vals).predict(len(actual))

        gm_rmsse.append(rmsse_m5(actual, gm_pred, train_vals))
        best_base_rmsse.append(rmsse_m5(actual, base_pred, train_vals))

        leaf_forecasts_gm[sid] = gm_pred
        leaf_history[sid] = train_vals

        # Calibrated interval on the training history -> sigma-equivalent buffer
        resid = collect_residuals_by_step(
            train_vals, lambda: MovingAverageForecaster(28),
            horizon=HORIZON, n_windows=4)
        cov_here = np.nan
        if len(resid) >= 2:
            q_lo, q_hi = residual_quantiles(resid, alpha=0.1)
            s = conformal_scale(resid, q_lo, q_hi, target=0.9)
            base_fc = base_factory().fit(train_vals).predict(len(actual))
            lower, upper = make_interval(base_fc, s * q_lo[:len(actual)],
                                         s * q_hi[:len(actual)])
            cov_here = coverage(actual, lower, upper)
            coverage_records.append(cov_here)
            # Interval half-width as an uncertainty proxy for safety stock
            interval_sigma = float(np.mean(upper - lower) / 3.29)  # ~90% z-range
        else:
            interval_sigma = float(train_vals.std())

        for name, pred in [("global", gm_pred), (best_base, base_pred)]:
            fc_daily = float(np.mean(pred))
            r = simulate_inventory(actual, fc_daily, interval_sigma, z=1.04,
                                   holding_cost=1.0, stockout_cost=8.0)
            inv_rows.append({"series_id": sid, "method": name,
                             "total_cost": r["total_cost"],
                             "fill_rate": r["fill_rate"]})

    inv = pd.DataFrame(inv_rows)
    inv_summary = inv.groupby("method")[["total_cost", "fill_rate"]].mean().reset_index()
    inv_summary.to_csv(OUTDIR / "inventory_summary.csv", index=False)

    # 4. Hierarchical reconciliation: bottom-up the leaf (global) forecasts to a
    #    coherent total, and confirm coherence.
    bu = bottom_up(leaf_forecasts_gm)
    reconciliation_coherent = bool(is_coherent(bu))
    total_forecast_day0 = float(bu["total"][0])

    mean_coverage = float(np.nanmean(coverage_records)) if coverage_records else float("nan")

    metrics = {
        "audit": info,
        "n_series_used": int(panel["series_id"].nunique()),
        "backtest_rmsse": {r["method"]: float(r["rmsse"]) for _, r in summary.iterrows()},
        "global_rmsse_mean": float(np.nanmean(gm_rmsse)),
        "best_baseline": best_base,
        "best_baseline_rmsse_mean": float(np.nanmean(best_base_rmsse)),
        "inventory_cost": {r["method"]: float(r["total_cost"])
                           for _, r in inv_summary.iterrows()},
        "interval_mean_coverage_nominal_90": mean_coverage,
        "hierarchy_bottom_up_coherent": reconciliation_coherent,
        "hierarchy_total_forecast_day0": total_forecast_day0,
    }
    with open(OUTDIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary
    print("Demand-to-inventory pipeline — M5 subset\n")
    print(f"Series used: {metrics['n_series_used']}  "
          f"(zero-sales share {info['zero_sales_share']:.0%})\n")
    print("Backtest RMSSE (rolling origin, lower better):")
    print(summary.to_string(index=False))
    print(f"\nGlobal model mean RMSSE (M5-style): {metrics['global_rmsse_mean']:.3f}")
    print(f"Best baseline ({best_base}) mean RMSSE: {metrics['best_baseline_rmsse_mean']:.3f}")
    print(f"\nInterval mean coverage (nominal 90%): {mean_coverage:.0%}")
    print(f"Hierarchy bottom-up coherent: {reconciliation_coherent}")
    print("\nInventory cost per method (holding=1, stockout=8, buffer from calibrated interval):")
    print(inv_summary.to_string(index=False))
    print("\nKey point: compare the RMSSE ranking with the inventory-cost ranking —")
    print("the most accurate forecaster is not always the cheapest policy.")


if __name__ == "__main__":
    main()
