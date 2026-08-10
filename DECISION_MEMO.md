# Decision Memo — Demand Forecasting for Inventory Replenishment

**To:** Operations / Supply Planning
**From:** Data Science
**Re:** Which forecasting approach to deploy for FOODS replenishment

## Bottom line

Choose the forecaster by **total inventory cost**, not by forecast accuracy
alone. Our backtest shows the most accurate model (lowest RMSSE) is **not**
always the one that minimises cost once holding and stockout penalties are
applied. The right choice depends on how expensive a stockout is relative to
holding a unit.

## What we did

1. Built a leakage-free rolling-origin backtest on M5 FOODS demand (~54% of days
   have zero sales — genuinely intermittent).
2. Compared simple baselines (seasonal-naive, moving-average) against a global
   gradient-boosted model, scored with RMSSE (the M5 metric).
3. Converted each forecast into an order-up-to inventory policy and simulated the
   realised holding + stockout cost.

## What we found

- **Simple baselines are strong.** On intermittent demand a moving-average is
  competitive with the gradient-boosted model. A complex model is not worth
  deploying unless it beats this baseline in *cost*, not just error.
- **Accuracy ≠ cost.** As the stockout penalty rises, a forecaster that carries
  more buffer (slightly worse point accuracy) becomes the cheaper policy. The
  cost curve, not the error metric, should drive the deployment decision.

## Recommendation

- Adopt the moving-average / seasonal baseline as the default, and only promote
  the global model where it demonstrably lowers **simulated cost** for that
  segment.
- Set the service level (safety-stock multiplier) from the actual holding and
  stockout economics per category, and re-run the cost simulation when those
  costs change.
- Track interval coverage in production; our intervals under-cover on the most
  intermittent items, so safety stock there should lean conservative.

## Caveats

Results are on a FOODS subset with illustrative cost parameters. Before rollout,
calibrate holding/stockout costs and lead times per category and re-run the
simulation.
