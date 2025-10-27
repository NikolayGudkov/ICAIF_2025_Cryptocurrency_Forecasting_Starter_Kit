"""
Cross-sectional trading strategies
Aligned with evaluate.py evaluation logic
"""

import numpy as np
import pandas as pd

# --------------------------- Core Utility ---------------------------
def _simple_returns_base(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame):
    """
    Simple return relative to previous time step:
      true_ret(t) = true_close(t) / true_close(t-1) - 1
      pred_ret(t) = pred_close(t) / true_close(t-1) - 1
    """
    # Merge y_true and x_test to get both current and previous prices
    tr = y_true.copy().sort_values(["window_id", "time_step"])
    pr = y_pred.copy().sort_values(["window_id", "time_step"])

    # Get all x_test data (historical prices)
    x_all = x_test[["window_id", "time_step", "close"]].copy()
    x_all = x_all.sort_values(["window_id", "time_step"])

    # For each row in y_true/y_pred, find the previous time step's true price
    # Combine x_test and y_true to get full price history
    full_prices = pd.concat([
        x_all.rename(columns={"close": "price"}),
        tr[["window_id", "time_step", "close"]].rename(columns={"close": "price"})
    ]).sort_values(["window_id", "time_step"])

    # Create previous price column
    full_prices["prev_price"] = full_prices.groupby("window_id")["price"].shift(1)

    # Merge previous prices back to true and pred
    prev_prices = full_prices[["window_id", "time_step", "prev_price"]].dropna()

    tr = tr.merge(prev_prices, on=["window_id", "time_step"], how="inner")
    pr = pr.merge(prev_prices, on=["window_id", "time_step"], how="inner")

    eps = 1e-12
    tr["true_ret"] = tr["close"] / (tr["prev_price"] + eps) - 1.0
    pr["pred_ret"] = pr["pred_close"] / (pr["prev_price"] + eps) - 1.0

    pr = pr[["window_id", "time_step", "pred_ret"]]
    return tr, pr

# --------------------------- Portfolio Strategies per Timestamp ---------------------------
def _portfolio_csm(g: pd.DataFrame, top_decile: float = 0.10) -> float:
    """
    Cross-Sectional Momentum: long top 10%, short bottom 10% (equal-weighted)
    """
    if len(g) < max(10, int(1/top_decile) + int(1/top_decile)):
        return 0.0
    q_hi = np.quantile(g["pred_ret"], 1.0 - top_decile)
    q_lo = np.quantile(g["pred_ret"], top_decile)
    long_ret  = g.loc[g["pred_ret"] >= q_hi, "true_ret"].mean()
    short_ret = g.loc[g["pred_ret"] <= q_lo, "true_ret"].mean()
    val = float(long_ret - short_ret)
    return val if np.isfinite(val) else 0.0

def _portfolio_lotq(g: pd.DataFrame, top_q: float = 0.20) -> float:
    """
    Long-Only Top Quantile: long top 20% (equal-weighted)
    """
    if len(g) < max(5, int(1/top_q)):
        return 0.0
    thr = np.quantile(g["pred_ret"], 1.0 - top_q)
    long_ret = g.loc[g["pred_ret"] >= thr, "true_ret"].mean()
    val = float(long_ret)
    return val if np.isfinite(val) else 0.0

def _portfolio_pw(g: pd.DataFrame) -> float:
    """
    Proportional Weighting: long-only, weighted by positive pred_ret
    """
    rhat = np.maximum(g["pred_ret"].to_numpy(), 0.0)
    denom = rhat.sum()
    if denom <= 0.0:
        return 0.0
    w = rhat / denom
    true = g["true_ret"].to_numpy()
    val = float(np.sum(w * true))
    return val if np.isfinite(val) else 0.0

# --------------------------- Strategy Time Series Builder ---------------------------
def _portfolio_series(y_true: pd.DataFrame, y_pred: pd.DataFrame,
                      x_test: pd.DataFrame, strat: str) -> np.ndarray:
    """
    Build time series of portfolio returns using specified strategy.
    Groups by event_datetime and computes cross-sectional portfolio return per timestamp.
    """
    tr, pr = _simple_returns_base(y_true, y_pred, x_test)
    if "event_datetime" not in tr.columns:
        return np.array([])
    df = tr.merge(pr, on=["window_id","time_step"], how="inner")
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True, errors="coerce")

    rets = []
    for ts, g in df.groupby("event_datetime", dropna=True):
        if len(g) < 3:
            continue
        if strat == "csm":
            r = _portfolio_csm(g)
        elif strat == "lotq":
            r = _portfolio_lotq(g)
        elif strat == "pw":
            r = _portfolio_pw(g)
        else:
            raise ValueError(f"Unknown strategy: {strat}")
        rets.append((ts, r))

    if not rets:
        return np.array([])
    # Sort by timestamp and return the series
    rets = np.array([r for _, r in sorted(rets, key=lambda x: x[0])], dtype=float)
    return rets

# --------------------------- Public Strategy Functions ---------------------------
def CSM(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame,
        top_decile: float = 0.10) -> np.ndarray:
    """
    Cross-Sectional Momentum (long-short) strategy.

    Args:
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]
        top_decile: Top/bottom quantile to trade (default 0.10 for 10%)

    Returns:
        Array of portfolio returns over time
    """
    return _portfolio_series(y_true, y_pred, x_test, "csm")

def LOTQ(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame,
         topq: float = 0.20) -> np.ndarray:
    """
    Long-Only Top Quantile strategy (equal-weighted).

    Args:
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]
        topq: Top quantile to go long (default 0.20 for top 20%)

    Returns:
        Array of portfolio returns over time
    """
    return _portfolio_series(y_true, y_pred, x_test, "lotq")

def PW(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> np.ndarray:
    """
    Proportional Weighting strategy (long-only, weighted by predicted return magnitude).

    Args:
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]

    Returns:
        Array of portfolio returns over time
    """
    return _portfolio_series(y_true, y_pred, x_test, "pw")

# --------------------------- Dispatcher ---------------------------
def run_strategy(name: str, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                 x_test: pd.DataFrame) -> np.ndarray:
    """
    Run a strategy by name.

    Args:
        name: Strategy name ('CSM', 'LOTQ', 'PW')
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]

    Returns:
        Array of portfolio returns over time
    """
    name = name.upper()
    if name == 'CSM':
        return CSM(y_true, y_pred, x_test)
    elif name == 'LOTQ':
        return LOTQ(y_true, y_pred, x_test)
    elif name in ('PW', 'PROPORTIONAL'):
        return PW(y_true, y_pred, x_test)
    else:
        raise ValueError(f"Unknown strategy: {name}")