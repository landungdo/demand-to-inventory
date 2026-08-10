"""
From forecast to inventory decision — and the cost that actually matters.

A forecast is an input; the decision is how much stock to hold. This module
turns a forecast (plus its uncertainty) into an order-up-to policy and then
simulates the realised cost against the true demand. The central point of the
whole project lives here:

    A model with lower RMSSE does not necessarily produce a cheaper inventory
    policy. Under asymmetric holding vs stockout costs, a slightly worse point
    forecast with better-calibrated uncertainty (or a helpful bias) can win on
    total cost.

Policy — periodic order-up-to level:
  target stock S = forecast demand over the lead-time + review period
                   + safety stock z * sigma
  where z is the service-level multiplier (e.g. 1.64 for ~95%) and sigma is the
  demand standard deviation over the protection interval.

Simulation:
  Walk day by day. Each day: receive orders due, meet demand from on-hand stock
  (unmet demand is a stockout), then reorder up to S on review days. Accrue
  holding cost on end-of-day stock and stockout cost on unmet units. Return the
  total cost decomposed into holding and stockout.
"""

from __future__ import annotations

import numpy as np


def safety_stock(sigma: float, z: float, protection_days: int) -> float:
    """Safety stock = z * sigma * sqrt(protection interval)."""
    return float(z * sigma * np.sqrt(max(protection_days, 1)))


def order_up_to_level(forecast_daily: float, sigma: float, z: float,
                      lead_time: int, review_period: int) -> float:
    """
    Order-up-to level S for a periodic-review policy.

    Covers expected demand over (lead_time + review_period) plus safety stock
    sized to the same protection interval.
    """
    protection = lead_time + review_period
    return float(forecast_daily * protection
                 + safety_stock(sigma, z, protection))


def simulate_inventory(demand_true: np.ndarray, forecast_daily: float,
                       sigma: float, z: float = 1.64,
                       lead_time: int = 2, review_period: int = 7,
                       holding_cost: float = 1.0,
                       stockout_cost: float = 5.0,
                       init_stock: float = None) -> dict:
    """
    Simulate a periodic-review order-up-to policy against true demand.

    Costs are asymmetric by design: stockout_cost >> holding_cost is the usual
    retail reality (a lost sale hurts more than a day of holding). Returns total
    cost and its decomposition, plus service level (fill rate).
    """
    demand_true = np.asarray(demand_true, dtype=float)
    n = len(demand_true)
    S = order_up_to_level(forecast_daily, sigma, z, lead_time, review_period)

    on_hand = float(init_stock if init_stock is not None else S)
    pipeline = {}  # day_index -> quantity arriving
    holding = 0.0
    stockout_units = 0.0
    demand_total = 0.0

    for day in range(n):
        # Receive any orders arriving today
        on_hand += pipeline.pop(day, 0.0)

        # Meet demand
        d = demand_true[day]
        demand_total += d
        sold = min(on_hand, d)
        unmet = d - sold
        on_hand -= sold
        stockout_units += unmet

        # Holding cost on end-of-day stock
        holding += holding_cost * on_hand

        # Reorder on review days: order up to S, arriving after lead_time
        if day % review_period == 0:
            in_pipeline = sum(pipeline.values())
            position = on_hand + in_pipeline
            order = max(0.0, S - position)
            if order > 0:
                pipeline[day + lead_time] = pipeline.get(day + lead_time, 0.0) + order

    stockout = stockout_cost * stockout_units
    return {
        "order_up_to": S,
        "holding_cost": holding,
        "stockout_cost": stockout,
        "total_cost": holding + stockout,
        "stockout_units": stockout_units,
        "fill_rate": float(1 - stockout_units / demand_total) if demand_total > 0 else 1.0,
    }


if __name__ == "__main__":
    import numpy as np

    # Demand with occasional spikes (typical retail): a point forecast tuned to
    # the usual day misses the spikes. Model A nails typical demand (better point
    # accuracy); model B over-forecasts and carries more buffer.
    rng = np.random.default_rng(1)
    future = rng.poisson(3, 56)
    future[[10, 25, 40]] = [30, 35, 28]      # demand spikes
    sigma = float(future.std())
    fc_A, fc_B = 3.0, 5.0                     # A accurate, B biased-high

    print("Inventory cost vs point accuracy (holding=1)\n")
    print("A has the better point forecast (closer to typical demand);")
    print("B over-forecasts and holds more buffer. Who wins depends on the")
    print("stockout/holding ratio — so we sweep the stockout cost:\n")
    print(f"{'stockout$':>9} | {'A total':>9} {'A fill':>7} | {'B total':>9} {'B fill':>7} | winner")
    print("-" * 66)
    for so in (4, 8, 16, 32):
        rA = simulate_inventory(future, fc_A, sigma, z=0.0, lead_time=2,
                                review_period=7, holding_cost=1.0, stockout_cost=so)
        rB = simulate_inventory(future, fc_B, sigma, z=0.0, lead_time=2,
                                review_period=7, holding_cost=1.0, stockout_cost=so)
        winner = "A" if rA["total_cost"] < rB["total_cost"] else "B"
        print(f"{so:>9} | {rA['total_cost']:>9.0f} {rA['fill_rate']:>7.0%} | "
              f"{rB['total_cost']:>9.0f} {rB['fill_rate']:>7.0%} | {winner}")
    print("\nAs the stockout penalty rises, the 'less accurate' buffering model B")
    print("overtakes the more accurate model A. Lower RMSSE does not guarantee the")
    print("cheaper inventory decision — the cost structure decides.")
