# Metrics: MSE, MAE, IC, IR, SharpeRatio, MDD, VaR, ES

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, Tuple

# ----------------- Utility -----------------s
def _base_from_x_test(x_test: pd.DataFrame) -> pd.DataFrame:
    last_ts = int(x_test['time_step'].max())  # should be 59
    base = x_test.loc[x_test['time_step'] == last_ts, ['window_id','close']].copy()
    base = base.rename(columns={'close':'base_close'})
    return base

def _returns_truth_pred(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame):
    base = _base_from_x_test(x_test)
    t = y_true.merge(base, on='window_id', how='inner')
    p = y_pred.merge(base, on='window_id', how='inner')
    t['true_ret'] = t['close'] / (t['base_close'] + 1e-12) - 1.0
    p['pred_ret'] = p['pred_close'] / (p['base_close'] + 1e-12) - 1.0
    return t, p

# ----------------- Metrics -----------------
def MSE(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    df = y_true.merge(y_pred, on=['window_id','time_step'])
    return float(np.mean((df['close'] - df['pred_close']) ** 2))

def MAE(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    df = y_true.merge(y_pred, on=['window_id','time_step'])
    return float(np.mean(np.abs(df['close'] - df['pred_close'])))

def IC(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> float:
    tr, pr = _returns_truth_pred(y_true, y_pred, x_test)
    df = tr.merge(pr, on=['window_id','time_step'])
    ics = []
    for h, g in df.groupby('time_step'):
        if len(g) >= 3:
            ic = spearmanr(g['pred_ret'], g['true_ret'], nan_policy='omit')[0]
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0

def IR(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> float:
    tr, pr = _returns_truth_pred(y_true, y_pred, x_test)
    df = tr.merge(pr, on=['window_id','time_step'])
    ics = []
    for h, g in df.groupby('time_step'):
        if len(g) >= 3:
            ic = spearmanr(g['pred_ret'], g['true_ret'], nan_policy='omit')[0]
            if np.isfinite(ic):
                ics.append(ic)
    if len(ics) < 2: return 0.0
    return float(np.mean(ics) / (np.std(ics) + 1e-12))

def SharpeRatio(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> float:
    tr, pr = _returns_truth_pred(y_true, y_pred, x_test)
    df = tr.merge(pr, on=['window_id','time_step'])
    # portfolio: top 20% long
    rets = []
    for h, g in df.groupby('time_step'):
        thr = np.quantile(g['pred_ret'], 0.8)
        sel = g[g['pred_ret'] >= thr]
        r = sel['true_ret'].mean() if len(sel) else 0.0
        rets.append(r)
    rets = np.array(rets)
    return float(np.mean(rets) / (np.std(rets) + 1e-12)) if len(rets) else 0.0

def MDD(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> float:
    tr, pr = _returns_truth_pred(y_true, y_pred, x_test)
    df = tr.merge(pr, on=['window_id','time_step'])
    # simple path: equally weighted portfolio per horizon
    rets = []
    for h, g in df.groupby('time_step'):
        rets.append(g['true_ret'].mean())
    if not rets: return 0.0
    equity = np.cumprod(1.0 + np.array(rets))
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if dd.size else 0.0

def VaR(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame, alpha: float=0.05) -> float:
    tr, pr = _returns_truth_pred(y_true, y_pred, x_test)
    df = tr.merge(pr, on=['window_id','time_step'])
    rets = df['true_ret'].to_numpy()
    return float(np.nanpercentile(rets, 100*alpha)) if len(rets) else 0.0

def ES(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame, alpha: float=0.05) -> float:
    tr, pr = _returns_truth_pred(y_true, y_pred, x_test)
    df = tr.merge(pr, on=['window_id','time_step'])
    rets = df['true_ret'].to_numpy()
    if not len(rets): return 0.0
    var = np.nanpercentile(rets, 100*alpha)
    tail = rets[rets <= var]
    return float(np.mean(tail)) if len(tail) else var

# ----------------- One-call wrapper -----------------
def evaluate_all_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame) -> Dict[str, float]:
    return {
        "MSE": MSE(y_true, y_pred),
        "MAE": MAE(y_true, y_pred),
        "IC": IC(y_true, y_pred, x_test),
        "IR": IR(y_true, y_pred, x_test),
        "SharpeRatio": SharpeRatio(y_true, y_pred, x_test),
        "MDD": MDD(y_true, y_pred, x_test),
        "VaR": VaR(y_true, y_pred, x_test),
        "ES": ES(y_true, y_pred, x_test),
    }
