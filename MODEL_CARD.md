# Model Card — Demand Forecasting & Inventory Policy

## Intended use
- **Task:** forecast daily item×store demand and choose replenishment quantities
  that minimise total holding + stockout cost.
- **Users:** demand planners / operations data scientists.
- **Out of scope:** promotions optimisation, pricing, new-item cold-start
  (no sales history), and any single-customer decision.

## Data
- M5 (Walmart) daily unit sales, FOODS subset in California stores; 2011–2016.
- Highly intermittent (~54% zero-sales days), low-count integer demand.

## Assumptions
1. **No leakage / causal features.** Every feature and backtest split uses only
   information available before the forecast date.
2. **Stationarity.** Demand structure is stable enough between training and the
   forecast horizon; large regime shifts (e.g. COVID-scale) would degrade it.
3. **SUTVA-like independence.** Each series is modelled on its own history plus
   calendar; cross-item cannibalisation/substitution is not modelled.
4. **Cost structure is known.** The inventory recommendation depends on holding
   and stockout costs and lead time being specified correctly.

## Evaluation
- Rolling-origin backtest with RMSSE (M5 metric), plus MAE/RMSE.
- Prediction intervals evaluated by empirical **coverage**, not assumed.
- Forecasts scored not only on error but on **simulated inventory cost**.

## Known limitations
- Runs on a subset; full-catalogue WRMSSE not computed.
- Intervals under-cover on the most intermittent items (reported honestly).
- Illustrative cost parameters; not a calibrated business P&L.
- No deep-learning model, no production monitoring/retraining.

## Ethical / responsible-use notes
- Inventory decisions affect availability and waste; over-forecasting perishable
  goods raises spoilage. Cost parameters should reflect real waste, not only
  financial holding cost.
