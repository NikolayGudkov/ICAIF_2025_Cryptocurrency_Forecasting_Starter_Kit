import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import time

# ---------- tiny profiler ----------
class _Profiler:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.records = []  # list[(name, seconds)]
    def mark(self, name: str) -> None:
        t = time.perf_counter()
        self.records.append((name, t - self.last))
        self.last = t
    def report(self) -> Dict[str, float]:
        total = time.perf_counter() - self.t0
        out = {name: dur for name, dur in self.records}
        out["total"] = total
        return out

# -------------------------------------------------------------------------
# The _prev_price_from_union_fast function has been REMOVED entirely
# -------------------------------------------------------------------------

def evaluate_all_metrics_vectorized(
    y_true: Optional[pd.DataFrame],
    y_pred: Optional[pd.DataFrame],
    *,
    alpha: float = 0.05,
    top_decile: float = 0.10,   # for CSM
    top_q: float = 0.20,        # for LOTQ
    profile: bool = False,
    # Optional: pass a prebuilt base_df to skip merges/prev_price entirely
    base_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fast, vectorized evaluation... [docstring]

    OPTIMIZED: Assumes y_true contains 'prev_close' if base_df is not provided.
    """
    P = _Profiler() if profile else None

    # ---------- 1) Build/accept base price frame ----------
    if base_df is None:
        # OPTIMIZATION 1: Use y_true['prev_close'] and skip x_test
        if "prev_close" not in y_true.columns:
            raise ValueError("y_true must contain 'prev_close' column to use this optimized function.")

        # Merge once, pulling in 'prev_close' from y_true
        df = y_true.loc[:, ["window_id", "time_step", "close", "event_datetime", "prev_close"]].merge(
            y_pred.loc[:, ["window_id", "time_step", "pred_close"]],
            on=["window_id", "time_step"],
            how="inner",
            copy=False,
        )
        # Normalize dtype once (cheap no-op if already correct)
        if not pd.api.types.is_datetime64_any_dtype(df["event_datetime"]):
            df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True, errors="coerce")

        # MSE / MAE on price levels (no filtering)
        close_arr  = df["close"].to_numpy(dtype=np.float64)
        pclose_arr = df["pred_close"].to_numpy(dtype=np.float64)
        mse = float(np.nanmean((close_arr - pclose_arr) ** 2))
        mae = float(np.nanmean(np.abs(close_arr - pclose_arr)))
        if P: P.mark("merge+MSE/MAE")

        # The 6.3s bottleneck is GONE.
        # We just rename the column to match the rest of the function's expectations.
        df.rename(columns={"prev_close": "prev_price"}, inplace=True)
        if P: P.mark("prev_price_from_y_true")
    else:
        # Fast path: user provided
        required = {"event_datetime", "close", "pred_close", "prev_price"}
        missing = required.difference(base_df.columns)
        if missing:
            raise ValueError(f"base_df missing: {sorted(missing)}")
        df = base_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["event_datetime"]):
            df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True, errors="coerce")
        close_arr  = df["close"].to_numpy(dtype=np.float64)
        pclose_arr = df["pred_close"].to_numpy(dtype=np.float64)
        mse = float(np.mean((close_arr - pclose_arr) ** 2))
        mae = float(np.mean(np.abs(close_arr - pclose_arr)))
        if P: P.mark("fastpath:MSE/MAE")

    # Keep only rows usable for return-based metrics
    df = df.dropna(subset=["prev_price", "event_datetime"])
    if df.empty:
        print(df)
        metrics = {
            "MSE": mse, "MAE": mae, "IC": 0.0, "IR": 0.0,
            "SharpeRatio": 0.0, "MDD": 0.0, "VaR": 0.0, "ES": 0.0,
        }
        return (metrics, P.report()) if P else metrics

    # ---------- 2) Returns (reused everywhere) ----------
    eps = 1e-12
    inv_prev = 1.0 / (df["prev_price"].to_numpy(dtype=np.float64) + eps)
    true_ret = df["close"].to_numpy(dtype=np.float64) * inv_prev - 1.0
    pred_ret = df["pred_close"].to_numpy(dtype=np.float64) * inv_prev - 1.0
    df["true_ret"] = true_ret
    df["pred_ret"] = pred_ret
    if P: P.mark("returns")

    # ---------- 3) Group encoding and sizes ----------
    # Faster to group by int codes than datetimes
    codes, uniq_events = pd.factorize(df["event_datetime"], sort=True)
    G = len(uniq_events)
    if G == 0:
        metrics = {
            "MSE": mse, "MAE": mae, "IC": 0.0, "IR": 0.0,
            "SharpeRatio": 0.0, "MDD": 0.0, "VaR": 0.0, "ES": 0.0,
        }
        return (metrics, P.report()) if P else metrics

    size_total = np.bincount(codes, minlength=G)
    valid_pred = ~np.isnan(pred_ret)
    size_pred  = np.bincount(codes[valid_pred], minlength=G)
    size_total_row = size_total[codes]
    size_pred_row  = size_pred[codes]
    if P: P.mark("group_sizes")

    rx = pd.Series(pred_ret).groupby(codes, sort=False).rank(method="average").to_numpy(np.float64)
    ry = pd.Series(true_ret).groupby(codes, sort=False).rank(method="average").to_numpy(np.float64)
    
    if P: P.mark("ranks")

    # ---------- 5) IC / IR (Spearman via Pearson on ranks) ----------
    keep_row_ge3 = size_total_row >= 3
    ic_valid = keep_row_ge3 & ~np.isnan(rx) & ~np.isnan(ry)
    if not np.any(ic_valid):
        IC = 0.0
        IR = 0.0
    else:
        c_ic = codes[ic_valid]
        rxv, ryv = rx[ic_valid], ry[ic_valid]
        n   = np.bincount(c_ic, minlength=G).astype(np.float64)
        sx  = np.bincount(c_ic, weights=rxv, minlength=G)
        sy  = np.bincount(c_ic, weights=ryv, minlength=G)
        sxx = np.bincount(c_ic, weights=rxv * rxv, minlength=G)
        syy = np.bincount(c_ic, weights=ryv * ryv, minlength=G)
        sxy = np.bincount(c_ic, weights=rxv * ryv, minlength=G)

        num = n * sxy - sx * sy
        den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy)) + 1e-12
        corr = num / den
        corr = corr[np.isfinite(corr)]
        if corr.size == 0:
            IC, IR = 0.0, 0.0
        else:
            IC = float(np.mean(corr))
            IR = float(IC / (np.std(corr) + 1e-12)) if corr.size >= 2 else 0.0
    if P: P.mark("IC/IR")

    # ---------- 6) Strategies (CSM / LOTQ / PW), reusing ranks ----------
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = rx / np.maximum(size_pred_row.astype(np.float64), 1.0)

    long_csm  = pct > (1.0 - top_decile)
    short_csm = pct <= top_decile
    long_lotq = pct >= (1.0 - top_q)

    def _mean_masked(mask: np.ndarray, vals: np.ndarray) -> np.ndarray:
        m = mask & ~np.isnan(vals)
        if not np.any(m):
            return np.zeros(G, dtype=np.float64)
        c = codes[m]
        w = vals[m]
        s = np.bincount(c, weights=w, minlength=G)
        k = np.bincount(c, minlength=G).astype(np.float64)
        out = np.zeros(G, dtype=np.float64)
        np.divide(s, np.maximum(k, 1.0), out=out, where=(k > 0))
        return out

    long_mean_csm  = _mean_masked(long_csm,  true_ret)
    short_mean_csm = _mean_masked(short_csm, true_ret)
    rets_csm  = long_mean_csm - short_mean_csm
    rets_lotq = _mean_masked(long_lotq, true_ret)

    w = np.where(np.isnan(pred_ret), 0.0, np.maximum(pred_ret, 0.0))
    if np.any(w > 0):
        num_pw = np.bincount(codes, weights=w * np.where(np.isnan(true_ret), 0.0, true_ret), minlength=G)
        den_pw = np.bincount(codes, weights=w, minlength=G)
        rets_pw = np.zeros(G, dtype=np.float64)
        np.divide(num_pw, np.maximum(den_pw, 1e-12), out=rets_pw, where=(den_pw > 0))
    else:
        rets_pw = np.zeros(G, dtype=np.float64)
    if P: P.mark("strategies")

    # Cross-sectional minima (match earlier guards)
    min_csm  = max(10, int(np.ceil(1 / top_decile)) + int(np.ceil(1 / top_decile)))
    min_lotq = max(5, int(np.ceil(1 / top_q)))

    keep_csm  = (size_total >= 3) & (size_pred >= min_csm)
    keep_lotq = (size_total >= 3) & (size_pred >= min_lotq)
    keep_pw   = (size_total >= 3)

    rets_csm  = rets_csm[keep_csm]
    rets_lotq = rets_lotq[keep_lotq]
    rets_pw   = rets_pw[keep_pw]

    # ---------- 7) Risk/perf & average ----------
    def _sharpe(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.mean(x) / (np.std(x) + 1e-12)) if x.size else 0.0
    def _mdd(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        eq = np.cumprod(1.0 + x)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-12)
        return float(np.max(dd)) if dd.size else 0.0
    def _var(x: np.ndarray, a: float) -> float:
        x = x[np.isfinite(x)]
        return float(np.nanpercentile(x, 100 * a)) if x.size else 0.0
    def _es(x: np.ndarray, a: float) -> float:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        v = np.nanpercentile(x, 100 * a)
        tail = x[x <= v]
        return float(np.mean(tail)) if tail.size else float(v)

    sharpe_vals = (_sharpe(rets_csm), _sharpe(rets_lotq), _sharpe(rets_pw))
    mdd_vals    = (_mdd(rets_csm),    _mdd(rets_lotq),    _mdd(rets_pw))
    var_vals    = (_var(rets_csm, alpha), _var(rets_lotq, alpha), _var(rets_pw, alpha))
    es_vals     = (_es(rets_csm, alpha),  _es(rets_lotq, alpha),  _es(rets_pw, alpha))

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "IC": IC,
        "IR": IR,
        "SharpeRatio": float(np.mean(sharpe_vals)),
        "MDD": float(np.mean(mdd_vals)),
        "VaR": float(np.mean(var_vals)),
        "ES": float(np.mean(es_vals)),
    }
    if P: P.mark("risk/aggregation")

    return (metrics, P.report()) if P else metrics