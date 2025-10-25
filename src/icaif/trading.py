"""
Cross-sectional strategies for the 60->10 task that DO NOT require x_test.
Requirements:
  y_true must contain columns: ['window_id','time_step','close','base_close']
  y_pred must contain columns: ['window_id','time_step','pred_close']
Strategies:
  - CSM (Cross-Sectional Momentum): long top decile by predicted returns,
    short bottom decile (equal-weight).
  - LOTQ (Long-Only Top-Quantile): long-only top 20% by predicted returns
    (equal-weight), zero elsewhere.
  - PW (Proportional-Weighting): long-only; weights are proportional to
    predicted returns at the horizon. Emphasizes magnitude instead of rank.
"""

from typing import List
import numpy as np
import pandas as pd

# --------------------------- Utilities ---------------------------

def _need(df: pd.DataFrame, cols: List[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}. Found: {df.columns.tolist()}")

def _returns_truth_pred_at_h(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizon_step: int,
) -> pd.DataFrame:
    """
    Build a dataframe for a single horizon_step with columns:
      ['window_id','pred_ret','true_ret'].
    Returns are computed relative to base_close supplied in y_true.
    """
    if horizon_step < 0:
        raise ValueError("horizon_step must be >= 0")
    _need(y_true, ['window_id','time_step','close','base_close'], 'y_true')
    _need(y_pred, ['window_id','time_step','pred_close'], 'y_pred')

    t = y_true[y_true['time_step'] == horizon_step][['window_id','close','base_close']].copy()
    p = y_pred[y_pred['time_step'] == horizon_step][['window_id','pred_close']].copy()
    if t.empty or p.empty:
        raise ValueError("No rows found at the given horizon_step in y_true/y_pred.")

    t['true_ret'] = t['close'] / (t['base_close'] + 1e-12) - 1.0
    p = p.merge(t[['window_id','base_close']], on='window_id', how='inner')
    p['pred_ret'] = p['pred_close'] / (p['base_close'] + 1e-12) - 1.0

    df = p[['window_id','pred_ret']].merge(t[['window_id','true_ret']], on='window_id', how='inner')
    return df  # one row per window_id at the chosen horizon

# --------------------------- Strategies ---------------------------

def CSM(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizon_step: int = 0,
    top_decile: float = 0.10,
) -> np.ndarray:
    """
    Cross-Sectional Momentum (long-short):
      - Long equal-weight the top decile by predicted returns.
      - Short equal-weight the bottom decile by predicted returns.
      - Portfolio return = mean(true_ret of long) - mean(true_ret of short).
    Returns an array of realized portfolio returns across the cross section
    (one scalar here for the chosen horizon).
    """
    df = _returns_truth_pred_at_h(y_true, y_pred, horizon_step)
    q_hi = np.quantile(df['pred_ret'].to_numpy(), 1.0 - top_decile)
    q_lo = np.quantile(df['pred_ret'].to_numpy(), top_decile)

    long_leg = df[df['pred_ret'] >= q_hi]
    short_leg = df[df['pred_ret'] <= q_lo]

    long_ret = float(long_leg['true_ret'].mean()) if len(long_leg) else 0.0
    short_ret = float(short_leg['true_ret'].mean()) if len(short_leg) else 0.0

    port_ret = long_ret - short_ret
    return np.array([port_ret], dtype=np.float64)

def LOTQ(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizon_step: int = 0,
    top_quantile: float = 0.20,
) -> np.ndarray:
    """
    Long-Only Top-Quantile:
      - Long equal-weight the top 'top_quantile' by predicted returns.
      - Zero exposure to others.
    Returns an array with one realized portfolio return for the chosen horizon.
    """
    df = _returns_truth_pred_at_h(y_true, y_pred, horizon_step)
    thr = np.quantile(df['pred_ret'].to_numpy(), 1.0 - top_quantile)
    long_leg = df[df['pred_ret'] >= thr]
    long_ret = float(long_leg['true_ret'].mean()) if len(long_leg) else 0.0
    return np.array([long_ret], dtype=np.float64)

def PW(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizon_step: int = 0,
    clip_negative: bool = False,   # default: signed weights
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Proportional-Weighting (PW):
      - Allocate portfolio weights proportional to predicted returns.
      - Default uses signed weights (allowing long/short).
      - Portfolio return = weighted sum of true returns with proportional weights.
    Returns an array with one realized portfolio return for the chosen horizon.
    """
    df = _returns_truth_pred_at_h(y_true, y_pred, horizon_step)
    rhat = df['pred_ret'].to_numpy().astype(np.float64)

    if clip_negative:
        rhat = np.maximum(rhat, 0.0)
        denom = rhat.sum()
        if denom <= eps:
            return np.array([0.0], dtype=np.float64)
        w = rhat / denom
    else:
        # Signed weights: normalize by L1 norm to avoid degenerate denominator
        denom = np.sum(np.abs(rhat))
        if denom <= eps:
            return np.array([0.0], dtype=np.float64)
        w = rhat / denom

    realized = float((w * df['true_ret'].to_numpy().astype(np.float64)).sum())
    return np.array([realized], dtype=np.float64)

# --------------------------- Convenience ---------------------------

def run_strategy(
    name: str,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizon_step: int = 0,
) -> np.ndarray:
    """
    Dispatch a strategy by name: 'CSM', 'LOTQ', or 'PW'.
    Returns the realized portfolio return series (np.ndarray).
    """
    key = name.upper()
    if key == 'CSM':
        return CSM(y_true, y_pred, horizon_step=horizon_step)
    elif key == 'LOTQ':
        return LOTQ(y_true, y_pred, horizon_step=horizon_step)
    elif key in ('PW', 'PROPORTIONAL', 'S3'):
        return PW(y_true, y_pred, horizon_step=horizon_step)
    else:
        raise ValueError(f"Unknown strategy: {name}")
