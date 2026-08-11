# Demand-to-Inventory — Retail Forecasting as a Decision System

![tests](https://img.shields.io/badge/tests-53%20passed-brightgreen)

A demand-forecasting project on the **M5 (Walmart) dataset** that does not stop at
a forecast. It turns forecasts into an **inventory replenishment policy** and
scores that policy by its **realised cost**, because the question a retailer
actually asks is not "what's the RMSSE?" but "how much does this cost me?"

> **The central finding:** a model with lower forecast error does **not**
> automatically produce a cheaper inventory decision. Under asymmetric holding
> vs stockout costs, the accuracy ranking and the cost ranking can disagree —
> and the cost ranking is the one that matters.

This is the third of three decision-focused projects: credit risk (predict →
lend), uplift (estimate effect → target), and here demand (forecast → stock).

## What's inside

| Component | Module | What it does |
|---|---|---|
| Data pipeline | `src/m5_data.py` | Load M5, subset by category/store, reshape wide→long, join calendar, audit |
| Baselines | `src/baselines.py` | Naive, seasonal-naive, moving-average — the benchmarks to beat |
| Metrics | `src/metrics.py` | MAE, RMSE, and **RMSSE** (M5's per-series scaled error; full WRMSSE with dollar weights not computed) |
| Backtesting | `src/backtest.py` | **Rolling-origin** evaluation with no leakage |
| Global model | `src/global_model.py` | One gradient-boosted model across all series, causal lag features |
| Prediction intervals | `src/intervals.py` | Residual-quantile + split-conformal intervals, **coverage-checked** |
| Hierarchy | `src/hierarchy.py` | Bottom-up / top-down reconciliation (coherent across leaf ↔ total; store level not separately modelled) |
| Inventory decision | `src/inventory.py` | Order-up-to policy, safety stock, **asymmetric-cost simulation** |

## The data (M5 subset)

The pipeline is run on FOODS_3 in California stores. A representative audit:

- ~2,469 item×store series (one store ≈ 800), 1,941 days (2011–2016)
- **~54% zero-sales days** — genuinely intermittent demand

That intermittency drives the design: percentage errors (MAPE) are useless, so
**RMSSE** (M5's per-series scaled error) is the metric; and simple methods are strong, so a **seasonal-naive
baseline is mandatory** before trusting anything fancier.

## Headline results (M5 subset, rolling-origin backtest)

Run `python scripts/reproduce.py` to regenerate these into `results/`. The
committed `results/` files are a **sample run on an M5-shaped panel**; running
the script on the real M5 CSVs overwrites them.

A representative run shows the project's central point directly — the accuracy
ranking and the cost ranking **disagree**:

| Method | RMSSE (accuracy) | Inventory cost | Fill rate |
|---|---|---|---|
| Moving-average baseline | **0.78** (better) | 1328 | 99% |
| Global gradient-boosted | 0.81 | **1271** (cheaper) | 98% |

The moving-average has the better forecast error, yet the global model produces
the **cheaper inventory policy** under these holding/stockout costs. Lower RMSSE
did not win the decision — which is the whole argument for scoring forecasts by
their downstream cost, not error alone. (Exact numbers depend on the subset and
cost parameters; see `results/metrics.json`.)

On the intermittent FOODS series the simple baselines are genuinely hard to beat
on RMSSE — itself a finding worth stating rather than hiding.

## Two methodological commitments (shared with the other projects)

- **No leakage.** Every feature uses only past data, and the rolling-origin
  backtest never trains past the cutoff — the forecasting analogue of an
  out-of-time split.
- **Honest uncertainty.** Prediction intervals are checked for empirical
  coverage rather than assumed; on intermittent demand they under-cover the
  nominal level, which is reported, not hidden.

## Running it

```bash
pip install -r requirements.txt
# Place the M5 CSVs in data/ (see "The data" — they are gitignored)
python src/m5_data.py            # data audit
python src/backtest.py           # rolling-origin backtest (synthetic demo)
python src/global_model.py       # global gradient-boosted forecaster
python src/intervals.py          # interval coverage
python src/inventory.py          # accuracy-vs-cost demonstration
python scripts/reproduce.py      # full pipeline on the M5 subset -> results/
pytest tests/ -v                 # full test suite
```

## Limitations & scope

- Runs on a configurable **subset** for speed; the pipeline scales to more
  series by widening the subset, but the full 30k-series WRMSSE is not computed.
- No deep-learning model; a gradient-boosted global model is the ceiling here by
  design (strong, simple, honest baseline-first methodology).
- Inventory unit economics (holding/stockout costs, lead time) are illustrative
  parameters, not a calibrated business P&L.
- Not production-grade: no live data pipeline, monitoring, or retraining loop.
