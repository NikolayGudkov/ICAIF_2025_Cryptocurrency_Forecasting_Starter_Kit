# =========================================================
# Metrics: MSE, MAE, IC, IR, SharpeRatio, MDD, VaR, ES
# Aligned with evaluate.py evaluation logic
# =========================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict
from trading import CSM, LOTQ, PW


# ----------------- Utility -----------------
def _simple_returns_base(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame):
    """
    Simple return relative to previous time step:
      true_ret(t) = true_close(t) / true_close(t-1) - 1
      pred_ret(t) = pred_close(t) / true_close(t-1) - 1
    y_true must contain event_datetime; function preserves original columns (including event_datetime)
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


# ----------------- Error Metrics -----------------
def MSE(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Mean Squared Error on close prices"""
    df = y_true.merge(y_pred, on=["window_id", "time_step"], how="inner")
    return float(np.mean((df["close"] - df["pred_close"]) ** 2))


def MAE(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Mean Absolute Error on close prices"""
    df = y_true.merge(y_pred, on=["window_id", "time_step"], how="inner")
    return float(np.mean(np.abs(df["close"] - df["pred_close"])))


# ----------------- Rank-based Metrics (IC/IR) -----------------
def IC(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> float:
    """
    Information Coefficient: average Spearman correlation across event_datetime timestamps
    """
    tr, pr = _simple_returns_base(y_true, y_pred, x_test)
    if "event_datetime" not in tr.columns:
        return 0.0
    df = tr.merge(pr, on=["window_id", "time_step"], how="inner")
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True, errors="coerce")

    ics = []
    for _, g in df.groupby("event_datetime", dropna=True):
        if len(g) >= 3:
            ic = spearmanr(g["pred_ret"], g["true_ret"], nan_policy="omit")[0]
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def IR(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> float:
    """
    Information Ratio: IC / std(IC), requires at least 2 timestamps
    """
    tr, pr = _simple_returns_base(y_true, y_pred, x_test)
    if "event_datetime" not in tr.columns:
        return 0.0
    df = tr.merge(pr, on=["window_id", "time_step"], how="inner")
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True, errors="coerce")

    ics = []
    for _, g in df.groupby("event_datetime", dropna=True):
        if len(g) >= 3:
            ic = spearmanr(g["pred_ret"], g["true_ret"], nan_policy="omit")[0]
            if np.isfinite(ic):
                ics.append(ic)
    if len(ics) < 2:
        return 0.0
    return float(np.mean(ics) / (np.std(ics) + 1e-12))


# ----------------- Risk / Performance Metrics -----------------
def _sharpe_ratio(rets: np.ndarray) -> float:
    """Sharpe ratio: mean / std"""
    if not rets.size:
        return 0.0
    return float(np.mean(rets) / (np.std(rets) + 1e-12))


def _max_drawdown(rets: np.ndarray) -> float:
    """
    Relative MDD: max_t (peak - equity)/peak, where equity = ∏(1+r)
    """
    if not rets.size:
        return 0.0
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-12)
    return float(np.max(dd)) if dd.size else 0.0


def _var(rets: np.ndarray, alpha: float = 0.05) -> float:
    """Value at Risk at level alpha"""
    if not rets.size:
        return 0.0
    return float(np.nanpercentile(rets, 100 * alpha))


def _es(rets: np.ndarray, alpha: float = 0.05) -> float:
    """Expected Shortfall (CVaR) at level alpha"""
    if not rets.size:
        return 0.0
    v = np.nanpercentile(rets, 100 * alpha)
    tail = rets[rets <= v]
    return float(np.mean(tail)) if tail.size else float(v)


# ----------------- One-call Wrapper -----------------
def evaluate_all_metrics(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        x_test: pd.DataFrame,
        alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute all metrics: MSE, MAE, IC, IR, SharpeRatio, MDD, VaR, ES
    Averages portfolio metrics across three strategies (CSM, LOTQ, PW)

    Args:
        y_true: Must contain columns [window_id, time_step, close, event_datetime]
        y_pred: Must contain columns [window_id, time_step, pred_close]
        x_test: Must contain columns [window_id, time_step, close]
        alpha: Significance level for VaR and ES (default 0.05)

    Returns:
        Dictionary with metric names and values
    """
    # Basic error metrics
    results = {
        "MSE": MSE(y_true, y_pred),
        "MAE": MAE(y_true, y_pred),
        "IC": IC(y_true, y_pred, x_test),
        "IR": IR(y_true, y_pred, x_test),
    }

    # Portfolio strategy metrics
    sharpe_list, mdd_list, var_list, es_list = [], [], [], []

    for strategy_fn in (CSM, LOTQ, PW):
        rets = strategy_fn(y_true, y_pred, x_test)
        sharpe_list.append(_sharpe_ratio(rets))
        mdd_list.append(_max_drawdown(rets))
        var_list.append(_var(rets, alpha))
        es_list.append(_es(rets, alpha))

    # Average across three strategies
    results["SharpeRatio"] = float(np.mean(sharpe_list))
    results["MDD"] = float(np.mean(mdd_list))
    results["VaR"] = float(np.mean(var_list))
    results["ES"] = float(np.mean(es_list))

    return results