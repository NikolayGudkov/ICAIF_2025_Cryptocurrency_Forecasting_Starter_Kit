# Metrics: MSE, MAE, IC, IR, SharpeRatio, MDD, VaR, ES

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, Tuple
from trading import CSM, LOTQ, PW

# ----------------- Utility -----------------s
def _base_from_x_test(x_test: pd.DataFrame) -> pd.DataFrame:
    last_ts = int(x_test['time_step'].max())  # should be 59
    base = x_test.loc[x_test['time_step'] == last_ts, ['window_id','close']].copy()
    base = base.rename(columns={'close':'base_close'})
    return base

def _returns_truth_pred(    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = _base_from_x_test(x_test)

    # sort to make groupby-shift deterministic
    tr = y_true.merge(base, on='window_id', how='inner').sort_values(['window_id','time_step']).copy()
    pr = y_pred.merge(base, on='window_id', how='inner').sort_values(['window_id','time_step']).copy()
    eps = 1e-12

    # true step-wise returns
    tr['prev_close'] = tr.groupby('window_id')['close'].shift(1)
    tr['den_true']   = tr['prev_close'].fillna(tr['base_close'])
    tr['true_ret']   = tr['close'] / (tr['den_true'] + eps) - 1.0
    tr = tr.drop(columns=['prev_close','den_true'])

    # pred step-wise returns
    pr['prev_pred'] = pr.groupby('window_id')['pred_close'].shift(1)
    pr['den_pred']  = pr['prev_pred'].fillna(pr['base_close'])
    pr['pred_ret']  = pr['pred_close'] / (pr['den_pred'] + eps) - 1.0
    pr = pr.drop(columns=['prev_pred','den_pred'])

    return tr, pr
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

def _sharpe_ratio(rets: np.ndarray) -> float:
    if len(rets) == 0: return 0.0
    return float(np.mean(rets) / (np.std(rets) + 1e-12))

def _max_drawdown(rets: np.ndarray) -> float:
    if len(rets) == 0: return 0.0
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if dd.size else 0.0

def _var(rets: np.ndarray, alpha: float = 0.05) -> float:
    if len(rets) == 0: return 0.0
    return float(np.nanpercentile(rets, 100 * alpha))

def _es(rets: np.ndarray, alpha: float = 0.05) -> float:
    if len(rets) == 0: return 0.0
    var = np.nanpercentile(rets, 100 * alpha)
    tail = rets[rets <= var]
    return float(np.mean(tail)) if len(tail) else var
# ----------------- One-call wrapper -----------------
def evaluate_all_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    x_test: pd.DataFrame,
    y_true_with_base: pd.DataFrame,
    horizon_step: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Return a flat dict with error metrics and the AVERAGE of trading-based metrics
    (SharpeRatio, MDD, VaR, ES) across strategies (CSM, LOTQ, PW).
    """
    results = {
        "MSE": MSE(y_true, y_pred),
        "MAE": MAE(y_true, y_pred),
        "IC": IC(y_true, y_pred, x_test),
        "IR": IR(y_true, y_pred, x_test),
    }

    # collect per-strategy metrics, then average
    sharpe_list, mdd_list, var_list, es_list = [], [], [], []

    for fn in (CSM, LOTQ, PW):
        rets = fn(y_true_with_base, y_pred, horizon_step=horizon_step)
        sharpe_list.append(_sharpe_ratio(rets))
        mdd_list.append(_max_drawdown(rets))
        var_list.append(_var(rets, alpha))
        es_list.append(_es(rets, alpha))

    # simple averages (fallback to 0.0 if any list is empty)
    results["SharpeRatio"] = float(np.mean(sharpe_list)) if sharpe_list else 0.0
    results["MDD"]         = float(np.mean(mdd_list))    if mdd_list    else 0.0
    results["VaR"]         = float(np.mean(var_list))    if var_list    else 0.0
    results["ES"]          = float(np.mean(es_list))     if es_list     else 0.0

    return results

