"""
Forecast accuracy metrics.

On intermittent demand (many zero-sales days) percentage errors (MAPE/sMAPE)
are useless — they divide by zero or explode. The right tools are absolute
errors and, above all, *scaled* errors:

  MAE, RMSE
    Standard absolute / squared errors, in units of sales.

  RMSSE (Root Mean Squared Scaled Error) — the official M5 metric
    Scales the forecast's squared error by the in-sample one-step seasonal-naive
    squared error, so series of different volumes are comparable and a value of
    1.0 means "as good as a naive random walk on the training data":

        RMSSE = sqrt( mean_h (y_hat - y)^2  /  mean_train (y_t - y_{t-1})^2 )

    < 1 beats the naive scale; > 1 is worse. Averaging RMSSE across series (M5
    weights by dollar volume) gives the WRMSSE; here we expose per-series RMSSE
    and a simple/weighted mean.
"""

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(d ** 2)))


def rmsse(y_true, y_pred, y_train, period: int = 1) -> float:
    """
    Root Mean Squared Scaled Error.

    y_train is the in-sample history used to compute the naive scale (the
    denominator). period=1 uses a random-walk naive; period=7 scales by the
    weekly-seasonal naive instead.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= period:
        return float("nan")
    # In-sample one-step (seasonal) naive squared error
    scale = np.mean((y_train[period:] - y_train[:-period]) ** 2)
    if scale == 0:
        # Flat training history (e.g. all zeros): fall back to a tiny epsilon so
        # a perfect forecast scores 0 and any error is large but finite.
        scale = 1e-8
    num = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(num / scale))


def evaluate_forecast(y_true, y_pred, y_train, period: int = 1) -> dict:
    """All three metrics for one forecast."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "rmsse": rmsse(y_true, y_pred, y_train, period=period),
    }


def mean_rmsse(rmsse_values, weights=None) -> float:
    """
    Aggregate per-series RMSSE. With weights (e.g. dollar volume) this is the
    WRMSSE-style weighted mean; without, a simple mean. NaNs are ignored.
    """
    r = np.asarray(rmsse_values, dtype=float)
    mask = ~np.isnan(r)
    r = r[mask]
    if len(r) == 0:
        return float("nan")
    if weights is not None:
        w = np.asarray(weights, dtype=float)[mask]
        if w.sum() == 0:
            return float(np.mean(r))
        return float(np.average(r, weights=w))
    return float(np.mean(r))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.synthetic_series import make_series
    from src.baselines import BASELINE_FORECASTERS

    series = make_series(n_days=200, seed=3, base=8, weekly_amp=4, intermittent=0.2)
    train, test = series[:-28], series[-28:]

    print("Baseline metrics on a 28-day holdout (intermittent series)\n")
    print(f"{'method':<18} {'MAE':>7} {'RMSE':>7} {'RMSSE':>7}")
    print("-" * 42)
    for name, make in BASELINE_FORECASTERS.items():
        pred = make().fit(train).predict(28)
        m = evaluate_forecast(test, pred, train, period=7)
        print(f"{name:<18} {m['mae']:>7.3f} {m['rmse']:>7.3f} {m['rmsse']:>7.3f}")
    print("\nRMSSE < 1 beats the in-sample seasonal-naive scale.")
