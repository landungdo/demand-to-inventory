"""
Tests for the inventory decision simulation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inventory import (
    safety_stock, order_up_to_level, simulate_inventory,
)


def test_safety_stock_scales_with_sigma_and_z():
    assert safety_stock(sigma=2, z=1.64, protection_days=9) > safety_stock(
        sigma=1, z=1.64, protection_days=9)
    assert safety_stock(sigma=2, z=2.0, protection_days=9) > safety_stock(
        sigma=2, z=1.0, protection_days=9)


def test_order_up_to_covers_protection_demand():
    S = order_up_to_level(forecast_daily=5, sigma=0, z=0,
                          lead_time=2, review_period=7)
    # With no uncertainty, S is just demand over the protection interval
    assert abs(S - 5 * 9) < 1e-9


def test_no_stockout_when_stock_ample():
    demand = np.full(30, 4.0)
    r = simulate_inventory(demand, forecast_daily=4, sigma=1, z=3.0,
                           holding_cost=1.0, stockout_cost=5.0)
    assert r["stockout_units"] == 0.0
    assert r["fill_rate"] == 1.0


def test_understock_creates_stockout():
    demand = np.full(30, 10.0)
    # Forecast far too low, no safety buffer -> stockouts
    r = simulate_inventory(demand, forecast_daily=1, sigma=0, z=0.0,
                           lead_time=1, review_period=7,
                           holding_cost=1.0, stockout_cost=5.0,
                           init_stock=0.0)
    assert r["stockout_units"] > 0
    assert r["fill_rate"] < 1.0


def test_total_cost_is_holding_plus_stockout():
    demand = np.random.default_rng(0).poisson(5, 40).astype(float)
    r = simulate_inventory(demand, forecast_daily=5, sigma=2, z=1.0)
    assert abs(r["total_cost"] - (r["holding_cost"] + r["stockout_cost"])) < 1e-6


def test_higher_stockout_penalty_favors_more_buffer():
    """The key project insight: the cost-minimising forecast shifts with the
    stockout/holding ratio, so a higher-buffer forecast wins as the penalty rises."""
    rng = np.random.default_rng(1)
    demand = rng.poisson(3, 56).astype(float)
    demand[[10, 25, 40]] = [30, 35, 28]
    sigma = demand.std()

    def total(fc, so):
        return simulate_inventory(demand, fc, sigma, z=0.0, lead_time=2,
                                  review_period=7, holding_cost=1.0,
                                  stockout_cost=so)["total_cost"]

    fc_accurate, fc_buffer = 3.0, 5.0
    # Low penalty: accurate wins; high penalty: buffer wins
    assert total(fc_accurate, 4) < total(fc_buffer, 4)
    assert total(fc_buffer, 32) < total(fc_accurate, 32)
