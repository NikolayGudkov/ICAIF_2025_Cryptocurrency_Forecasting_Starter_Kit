"""
Cross-sectional trading strategies — fully vectorized
Aligned with evaluate.py evaluation logic
"""

import numpy as np
import pandas as pd


# --------------------------- Core Utility (vectorized) ---------------------------
def _simple_returns_base(y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame):
    """
    Simple return relative to previous time step:
      true_ret(t) = true_close(t) / true_close(t-1) - 1
      pred_ret(t) = pred_close(t) / true_close(t-1) - 1

    Returns a single merged DataFrame with columns:
      [window_id, time_step, event_datetime, close, pred_close, prev_price, true_ret, pred_ret]
    """
    # Keep only necessary columns and inner-join on (window_id, time_step)
    tr = y_true.loc[:, ["window_id", "time_step", "close", "event_datetime"]].copy()
    pr = y_pred.loc[:, ["window_id", "time_step", "pred_close"]].copy()
    df = tr.merge(pr, on=["window_id", "time_step"], how="inner", copy=False)

    # Build *full* price history to compute previous true price (from x_test + y_true)
    hist = pd.concat(
        [
            x_test.loc[:, ["window_id", "time_step", "close"]],
            y_true.loc[:, ["window_id", "time_step", "close"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["window_id", "time_step"], keep="last")

    hist = hist.sort_values(["window_id", "time_step"], kind="mergesort")
    hist["prev_price"] = hist.groupby("window_id", sort=False)["close"].shift(1)

    # Map prev_price back to the working frame
    prev = (
        hist.set_index(["window_id", "time_step"])[["prev_price"]]
        .astype({"prev_price": "float64"})
    )
    df = df.join(prev, on=["window_id", "time_step"], how="left")

    # Compute returns (drop rows without a previous price)
    df = df.dropna(subset=["prev_price"]).copy()
    eps = 1e-12
    inv_prev = 1.0 / (df["prev_price"].to_numpy() + eps)
    df["true_ret"] = df["close"].to_numpy() * inv_prev - 1.0
    df["pred_ret"] = df["pred_close"].to_numpy() * inv_prev - 1.0

    # Normalize event_datetime
    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True, errors="coerce")

    return df


# --------------------------- Vectorized strategy reducers ---------------------------
def _csm_vectorized(df: pd.DataFrame, top_decile: float = 0.10) -> np.ndarray:
    """
    Cross-Sectional Momentum: long top q, short bottom q (equal-weighted) per timestamp.
    Vectorized using percentile ranks.
    """
    if "event_datetime" not in df.columns:
        return np.array([], dtype=float)

    # Group sizes & minimum cross-section guard
    sizes = df.groupby("event_datetime", sort=False).size()
    min_n = max(10, int(1 / top_decile) + int(1 / top_decile))
    valid_groups = sizes.index[sizes >= min_n]

    if valid_groups.empty:
        return np.array([], dtype=float)

    gkey = df["event_datetime"]
    # Percentile ranks of predicted returns within each timestamp
    pct = df.groupby(gkey, sort=False)["pred_ret"].rank(pct=True, method="first")

    long_mask = pct >= (1.0 - top_decile)
    short_mask = pct <= top_decile

    # Means within each group (NaNs ignored)
    long_mean = df["true_ret"].where(long_mask).groupby(gkey).mean()
    short_mean = df["true_ret"].where(short_mask).groupby(gkey).mean()

    # Long-short
    rets = (long_mean - short_mean).loc[valid_groups]

    # Sorted by timestamp as np.ndarray
    return rets.sort_index().to_numpy(dtype=float)


def _lotq_vectorized(df: pd.DataFrame, top_q: float = 0.20) -> np.ndarray:
    """
    Long-Only Top Quantile: long top q (equal-weighted) per timestamp.
    """
    if "event_datetime" not in df.columns:
        return np.array([], dtype=float)

    sizes = df.groupby("event_datetime", sort=False).size()
    min_n = max(5, int(1 / top_q))
    valid_groups = sizes.index[sizes >= min_n]
    if valid_groups.empty:
        return np.array([], dtype=float)

    gkey = df["event_datetime"]
    pct = df.groupby(gkey, sort=False)["pred_ret"].rank(pct=True, method="first")
    long_mask = pct >= (1.0 - top_q)

    long_mean = df["true_ret"].where(long_mask).groupby(gkey).mean().loc[valid_groups]
    return long_mean.sort_index().to_numpy(dtype=float)


def _pw_vectorized(df: pd.DataFrame) -> np.ndarray:
    """
    Proportional Weighting: long-only, weights proportional to positive predicted returns.
    """
    if "event_datetime" not in df.columns:
        return np.array([], dtype=float)

    gkey = df["event_datetime"]
    w = np.maximum(df["pred_ret"].to_numpy(), 0.0)

    num = pd.Series(w * df["true_ret"].to_numpy()).groupby(gkey).sum()
    den = pd.Series(w).groupby(gkey).sum()

    # Safe division: groups with den <= 0 -> 0 return
    out = np.divide(
        num.to_numpy(dtype=float),
        den.to_numpy(dtype=float),
        out=np.zeros_like(num.to_numpy(dtype=float)),
        where=(den.to_numpy(dtype=float) > 0.0),
    )

    # Ensure chronological order
    return pd.Series(out, index=num.index).sort_index().to_numpy(dtype=float)


# --------------------------- Strategy Time Series Builder ---------------------------
def _portfolio_series(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    x_test: pd.DataFrame,
    strat: str,
    *,
    top_decile: float = 0.10,
    top_q: float = 0.20,
) -> np.ndarray:
    """
    Build time series of portfolio returns using specified strategy in a vectorized way.
    Groups by event_datetime and computes cross-sectional portfolio return per timestamp.
    """
    df = _simple_returns_base(y_true, y_pred, x_test)
    if "event_datetime" not in df.columns:
        return np.array([], dtype=float)
    df = df.dropna(subset=["event_datetime"])

    # Optional global small-sample guard (kept to mirror original behavior)
    cs_sizes = df.groupby("event_datetime", sort=False).size()
    df = df[df["event_datetime"].isin(cs_sizes.index[cs_sizes >= 3])]
    if df.empty:
        return np.array([], dtype=float)

    strat = strat.lower()
    if strat == "csm":
        return _csm_vectorized(df, top_decile=top_decile)
    elif strat == "lotq":
        return _lotq_vectorized(df, top_q=top_q)
    elif strat == "pw":
        return _pw_vectorized(df)
    else:
        raise ValueError(f"Unknown strategy: {strat}")


# --------------------------- Public Strategy Functions ---------------------------
def CSM(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    x_test: pd.DataFrame,
    top_decile: float = 0.10,
) -> np.ndarray:
    """
    Cross-Sectional Momentum (long-short) strategy.
    Args:
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]
        top_decile: Top/bottom quantile to trade (default 0.10)
    Returns:
        Array of portfolio returns over time (chronological)
    """
    return _portfolio_series(
        y_true, y_pred, x_test, "csm", top_decile=top_decile
    )


def LOTQ(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    x_test: pd.DataFrame,
    topq: float = 0.20,
) -> np.ndarray:
    """
    Long-Only Top Quantile strategy (equal-weighted).
    Args:
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]
        topq: Top quantile to go long (default 0.20)
    Returns:
        Array of portfolio returns over time (chronological)
    """
    return _portfolio_series(
        y_true, y_pred, x_test, "lotq", top_q=topq
    )


def PW(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    x_test: pd.DataFrame,
) -> np.ndarray:
    """
    Proportional Weighting strategy (long-only, weighted by predicted return magnitude).
    Args:
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]
    Returns:
        Array of portfolio returns over time (chronological)
    """
    return _portfolio_series(y_true, y_pred, x_test, "pw")


# --------------------------- Dispatcher ---------------------------
def run_strategy(
    name: str, y_true: pd.DataFrame, y_pred: pd.DataFrame, x_test: pd.DataFrame,
    **kwargs
) -> np.ndarray:
    """
    Run a strategy by name.
    Args:
        name: Strategy name ('CSM', 'LOTQ', 'PW')
        y_true: Must contain [window_id, time_step, close, event_datetime]
        y_pred: Must contain [window_id, time_step, pred_close]
        x_test: Must contain [window_id, time_step, close]
        kwargs: pass-through (e.g., top_decile=..., topq=...)
    """
    name_u = name.upper()
    if name_u == "CSM":
        return CSM(y_true, y_pred, x_test, **kwargs)
    elif name_u == "LOTQ":
        # allow both "topq" and "top_q"
        if "top_q" in kwargs and "topq" not in kwargs:
            kwargs = {**kwargs, "topq": kwargs["top_q"]}
        return LOTQ(y_true, y_pred, x_test, **kwargs)
    elif name_u in ("PW", "PROPORTIONAL"):
        return PW(y_true, y_pred, x_test)
    else:
        raise ValueError(f"Unknown strategy: {name}")
