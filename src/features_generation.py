import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

def _rolling_mean(a, w):
    m = swv(a, window_shape=w, axis=1).mean(axis=2)
    return np.concatenate([np.full((a.shape[0], w-1), np.nan), m], axis=1)

def _rolling_std(a, w, eps=1e-12):
    x = swv(a, window_shape=w, axis=1)
    m = x.mean(axis=2, keepdims=True)
    v = np.mean((x - m)**2, axis=2)
    s = np.sqrt(np.maximum(v, 0.0))
    s = np.concatenate([np.full((a.shape[0], w-1), np.nan)], axis=1) if w>1 else s
    return np.concatenate([np.full((a.shape[0], w-1), np.nan), np.squeeze(s, axis=1)], axis=1)

def _rolling_min(a, w):
    mn = swv(a, w, axis=1).min(axis=2)
    return np.concatenate([np.full((a.shape[0], w-1), np.nan), mn], axis=1)

def _rolling_max(a, w):
    mx = swv(a, w, axis=1).max(axis=2)
    return np.concatenate([np.full((a.shape[0], w-1), np.nan), mx], axis=1)

def _ema(a, span):
    alpha = 2.0 / (span + 1.0)
    y = np.empty_like(a, dtype=float)
    y[:, 0] = a[:, 0]
    for t in range(1, a.shape[1]):
        y[:, t] = alpha * a[:, t] + (1.0 - alpha) * y[:, t-1]
    return y

def _rolling_slope_logprice(price, w, eps=1e-12):
    """OLS slope of log(price) on time index over rolling window of length w."""
    lp = np.log(np.clip(price, eps, None))
    X = np.arange(w, dtype=float)
    Xc = X - X.mean()
    denom = np.sum(Xc**2)
    Yw = swv(lp, w, axis=1)                          # (N, T-w+1, w)
    Ym = Yw.mean(axis=2, keepdims=True)
    num = np.sum((Xc * (Yw - Ym)), axis=2)           # (N, T-w+1)
    slope = num / (denom + eps)
    return np.concatenate([np.full((lp.shape[0], w-1), np.nan), slope], axis=1)

def _rolling_ac1(a, w, eps=1e-12):
    """Rolling lag-1 autocorrelation over window w."""
    W = swv(a, w, axis=1)                            # (N, T-w+1, w)
    x = W[:, :, :-1]
    y = W[:, :, 1:]
    xm = x.mean(axis=2, keepdims=True)
    ym = y.mean(axis=2, keepdims=True)
    xc = x - xm
    yc = y - ym
    num = np.sum(xc * yc, axis=2)
    den = np.sqrt(np.sum(xc**2, axis=2) * np.sum(yc**2, axis=2)) + eps
    ac1 = num / den
    return np.concatenate([np.full((a.shape[0], w-1), np.nan), ac1], axis=1)

def build_features_np(X, fillna=False, eps=1e-12):
    """
    X: np.ndarray of shape (N, T, 2) with channels [price, volume]
    Returns: features (N, T, F), feature_names (list[str])
    """
    assert X.ndim == 3 and X.shape[2] == 2, "X must be (N,T,2)"
    N, T, _ = X.shape
    price = np.clip(X[:, :, 0].astype(float), eps, None)
    volume = np.clip(X[:, :, 1].astype(float), eps, None)

    log_p = np.log(price)
    log_v = np.log(volume)

    # 1-step log return (causal, first step NaN)
    r1 = np.empty_like(log_p)
    r1[:, 0] = np.nan
    r1[:, 1:] = log_p[:, 1:] - log_p[:, :-1]

    # EMAs on *price* (not log) for interpretability
    ema3  = _ema(price, 3)
    ema10 = _ema(price, 10)
    ema30 = _ema(price, 30)
    ema3_10 = (ema3 / (ema10 + eps)) - 1.0
    ema10_30 = (ema10 / (ema30 + eps)) - 1.0

    # Momentum on log-price
    mom3  = np.full_like(log_p, np.nan);  mom3[:, 3:]  = log_p[:, 3:]  - log_p[:, :-3]
    mom10 = np.full_like(log_p, np.nan);  mom10[:, 10:] = log_p[:, 10:] - log_p[:, :-10]

    # Z-scores (price & log-volume) over 20
    # mean_lp20 = _rolling_mean(log_p, 20)
    # std_lp20  = _rolling_std(log_p, 20, eps)
    # z_lp20 = (log_p - mean_lp20) / (std_lp20 + eps)
    #
    # mean_lv20 = _rolling_mean(log_v, 20)
    # std_lv20  = _rolling_std(log_v, 20, eps)
    # z_lv20 = (log_v - mean_lv20) / (std_lv20 + eps)

    # Rolling slope of log-price (trend strength)
    slope30 = _rolling_slope_logprice(price, 30, eps)

    # Volatility (std of returns)
    def _roll_std_returns(r, w):
        R = swv(r, w, axis=1)                        # (N, T-w+1, w)
        m = np.nanmean(R, axis=2, keepdims=True)
        v = np.nanmean((R - m)**2, axis=2)
        s = np.sqrt(np.maximum(v, 0.0))
        return np.concatenate([np.full((N, w-1), np.nan), s], axis=1)
    vol10 = _roll_std_returns(r1, 10)
    vol20 = _roll_std_returns(r1, 20)
    vol60 = _roll_std_returns(r1, 60)
    vol_ratio = vol10 / (vol60 + eps)

    # Range position over 20 (on price)
    min20 = _rolling_min(price, 20)
    max20 = _rolling_max(price, 20)
    range_pos20 = (price - min20) / (np.maximum(max20 - min20, eps))

    # Relative volume & volume dynamics
    mean_v20 = _rolling_mean(volume, 20)
    rel_vol20 = volume / (mean_v20 + eps)

    vol_mom5 = np.full_like(log_v, np.nan)
    vol_mom5[:, 5:] = log_v[:, 5:] - log_v[:, :-5]

    signed_vol = np.sign(r1) * volume
    obv = np.cumsum(np.nan_to_num(np.sign(r1)) * volume, axis=1)

    # Interactions
    r1_abs_relvol = np.abs(r1) * rel_vol20
    r1_relvol = r1 * rel_vol20

    # Regime / sentiment
    up_frac20 = _rolling_mean((r1 > 0).astype(float), 20)
    ac1_10 = _rolling_ac1(r1, 10)

    # Stack features (order matters only for naming)
    feats = [price, volume,
        log_p, log_v, r1,
        mom3, mom10,
        ema3_10, ema10_30,
        #z_lp20,
        slope30,
        vol10, vol20, vol60, vol_ratio,
        range_pos20,
        rel_vol20, vol_mom5,
        #z_lv20,
        signed_vol, obv,
        r1_abs_relvol, r1_relvol,
        up_frac20, ac1_10
    ]
    feat_names = ['price', 'volume',
        "log_p","log_v","r1",
        "mom3","mom10",
        "ema3_10","ema10_30",
        #"z_lp20",
        "slope30",
        "vol10","vol20","vol60","vol_ratio",
        "range_pos20",
        "rel_vol20","vol_mom5",
        #"z_lv20",
        "signed_vol","obv",
        "r1_abs_relvol","r1_relvol",
        "up_frac20","ac1_10"
    ]
    F = len(feats)
    Xfeat = np.stack(feats, axis=2)  # (N, T, F)

    if fillna:
        # simple zero-fill after z-scoring downstream; or use forward-fill along time
        Xfeat = np.nan_to_num(Xfeat, nan=0.0, posinf=0.0, neginf=0.0)

    return Xfeat #, feat_names
