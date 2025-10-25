"""
Ridge MIMO forecaster for 60→10 minute crypto price paths.

Inputs (per sample):
  prices60: np.ndarray, shape (n_samples, 60) — last 60 closing prices
  volumes60: np.ndarray, shape (n_samples, 60) — last 60 volumes
  future_prices10: np.ndarray, shape (n_samples, 10) — next 10 closing prices (for training)

Model:
  - Feature engineering that respects the 60-minute window constraint (no leakage)
  - Multi-output Ridge regression trained to predict the next 10 closing prices
  - Convenience methods to return predicted future price paths

Usage example:
  model = RidgeMIMOPriceForecaster(alpha=2.0)
  model.fit(prices60_train, volumes60_train, future_prices10_train)
  yprice_pred = model.predict(prices60_val, volumes60_val)

Notes:
  - No signatures used here (as requested)
  - All features are computed strictly from the 60-minute window
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# =============================
# Utility functions (vectorized)
# =============================

def _safe_std(x: np.ndarray, axis: int) -> np.ndarray:
    eps = 1e-12
    s = x.std(axis=axis, ddof=1)
    return np.where(s < eps, eps, s)


def _ewma_last(x: np.ndarray, alpha: float) -> np.ndarray:
    """Compute EWMA along axis=1 and return the last value per row.
    x: (n, T)
    alpha: smoothing factor in (0, 1]
    """
    n, T = x.shape
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = 0.0
        for t in range(T):
            s = alpha * x[i, t] + (1.0 - alpha) * s
        out[i] = s
    return out


def _ema_last_from_span(x: np.ndarray, span: int) -> np.ndarray:
    """EMA with pandas-like span semantics: alpha = 2/(span+1). Returns last EMA per row."""
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    return _ewma_last(x, alpha)


def _ewma_vol_last(returns: np.ndarray, halflife: float = 12.0) -> np.ndarray:
    """EWMA volatility (sqrt of EWMA of squared returns) with given halflife (in minutes).
    returns: (n, T)
    returns are log-returns per minute.
    """
    # Convert halflife to alpha for EWMA on squares
    lam = 0.5 ** (1.0 / max(1e-6, halflife))  # decay per step
    alpha = 1.0 - lam
    ewma_sq_last = _ewma_last(returns ** 2, alpha)
    vol_last = np.sqrt(np.maximum(ewma_sq_last, 1e-12))
    return vol_last


def _sign(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))


def _alt_sign_count_last_m(returns: np.ndarray, m: int = 10) -> np.ndarray:
    """Count sign alternations over the last m returns for each sample.
    returns: (n, T)
    """
    n, T = returns.shape
    m = min(m, T)
    out = np.zeros(n, dtype=float)
    if m < 2:
        return out
    for i in range(n):
        r = returns[i, T - m : T]
        s = _sign(r)
        # Count alternations between consecutive nonzero signs
        flips = 0
        for k in range(1, len(s)):
            if s[k] != 0 and s[k - 1] != 0 and s[k] != s[k - 1]:
                flips += 1
        out[i] = float(flips)
    return out


# =============================
# Feature engineering
# =============================

@dataclass
class FeatureConfig:
    ema_spans: Tuple[int, int, int] = (5, 15, 30)
    vol_halflife: float = 12.0  # minutes
    alt_sign_m: int = 10
    zvol_max_lookback: int = 5
    use_feature_scaler: bool = True


class WindowFeatureBuilder:
    """Builds features from 60-minute windows of price & volume.

    Features implemented (computed from the window only):
      - Last 10 lagged log-returns: r_t, r_{t-1}, ..., r_{t-9}
      - Multi-scale log-returns over 2, 5, 10 minutes: r^{(2)}, r^{(5)}, r^{(10)}
      - Acceleration: a_t = r_t - r_{t-1}
      - zMA_k for k in {5, 15, 30}: (logp_t - EMA_k(logp)) / EWMA_vol_last
      - Volume z-score at t: zV_t (on log1p(volume) within-window)
      - Max zV over last 5 steps: max_{k<=5} zV_{t-k}
      - Interaction: zV_t * r_t
      - Microstructure bounce: sign alternation count over last m returns
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.cfg = config

    @staticmethod
    def _validate_inputs(prices60: np.ndarray, volumes60: np.ndarray) -> None:
        assert prices60.ndim == 2 and volumes60.ndim == 2, "prices60/volumes60 must be 2D"
        n1, T1 = prices60.shape
        n2, T2 = volumes60.shape
        assert n1 == n2, "prices and volumes must have same number of samples"
        assert T1 == 60 and T2 == 60, "expect 60 minutes for prices and volumes"
        if not np.all(prices60 > 0):
            raise ValueError("All prices must be positive to compute log-prices.")

    def build(self, prices60: np.ndarray, volumes60: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        self._validate_inputs(prices60, volumes60)
        n, T = prices60.shape  # T=60

        logp = np.log(prices60)
        # Minute returns over the 60-minute window → length 59
        r = np.diff(logp, axis=1)
        # Last observed price and last return
        p_last = prices60[:, -1]
        r_last = r[:, -1]

        # EWMA volatility (from returns) at the end of the window
        vol_last = _ewma_vol_last(r, halflife=self.cfg.vol_halflife)
        vol_last = np.where(vol_last < 1e-8, 1e-8, vol_last)

        # EMA-based mean-reversion z-scores (zMA_k)
        zma_feats = []
        zma_names = []
        for k in self.cfg.ema_spans:
            ema_last = _ema_last_from_span(logp, span=k)
            zma = (logp[:, -1] - ema_last) / vol_last
            zma_feats.append(zma)
            zma_names.append(f"zMA_{k}")
        zma_mat = np.stack(zma_feats, axis=1)  # (n, 3)

        # Volume features
        lv = np.log1p(volumes60)
        lv_mean = lv.mean(axis=1, keepdims=True)
        lv_std = _safe_std(lv, axis=1).reshape(-1, 1)
        zV = (lv - lv_mean) / lv_std
        zV_last = zV[:, -1]
        lb = self.cfg.zvol_max_lookback
        zV_max_last5 = zV[:, -lb:].max(axis=1)

        # Interaction
        zV_r_last = zV_last * r_last

        # Lagged returns stack (last 10)
        L = min(10, r.shape[1])
        lags = np.zeros((n, 10), dtype=float)
        for i in range(n):
            lags_i = r[i, -L:][::-1]  # r_t, r_{t-1}, ...
            lags[i, :L] = lags_i
        lag_names = [f"r_lag_{k+1}" for k in range(10)]

        # Multi-scale returns r^{(2)}, r^{(5)}, r^{(10)}
        def multi_ret(k: int) -> np.ndarray:
            if T - k - 1 < 0:
                return np.zeros(n)
            return logp[:, -1] - logp[:, -1 - k]

        r2 = multi_ret(2)
        r5 = multi_ret(5)
        r10 = multi_ret(10)
        ms_mat = np.stack([r2, r5, r10], axis=1)
        ms_names = ["r_win_2", "r_win_5", "r_win_10"]

        # Acceleration: r_t - r_{t-1}
        accel = r[:, -1] - (r[:, -2] if r.shape[1] >= 2 else 0.0)

        # Alternating sign count over last m returns
        altc = _alt_sign_count_last_m(r, m=self.cfg.alt_sign_m)

        # Assemble feature matrix
        X = np.concatenate([
            lags,                # 10
            ms_mat,              # +3
            zma_mat,             # +3
            zV_last[:, None],    # +1
            zV_max_last5[:, None],  # +1
            zV_r_last[:, None],  # +1
            accel[:, None],      # +1
            altc[:, None],       # +1
        ], axis=1)

        feature_names = (
            lag_names
            + ms_names
            + zma_names
            + ["zV_last", "zV_max_last5", "zV_last_x_r_last", "accel", "alt_sign_count_last_m"]
        )

        aux = {
            "p_last": p_last,           # needed to convert returns → prices at inference
            "vol_last": vol_last,       # can be useful for weighting/evaluation
            "r_last": r_last,
        }
        return X, aux, feature_names


# =============================
# Ridge MIMO forecaster
# =============================

class RidgeMIMOPriceForecaster:
    def __init__(self, alpha: float = 1.0, feature_config: FeatureConfig = FeatureConfig()):
        self.alpha = alpha
        self.feature_builder = WindowFeatureBuilder(feature_config)
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Ridge] = None
        self.feature_names_: Optional[List[str]] = None

    @staticmethod
    def _targets_from_prices(prices60: np.ndarray, future_prices10: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return future price levels and current price for training diagnostics."""
        if future_prices10.ndim != 2 or future_prices10.shape[1] != 10:
            raise ValueError("future_prices10 must have shape (n_samples, 10)")
        return future_prices10, prices60[:, -1]

    def fit(self, prices60: np.ndarray, volumes60: np.ndarray, future_prices10: np.ndarray) -> "RidgeMIMOPriceForecaster":
        """Fit the Ridge multi-output model on engineered features.
        - prices60: (n, 60)
        - volumes60: (n, 60)
        - future_prices10: (n, 10)
        """
        X, _, feat_names = self.feature_builder.build(prices60, volumes60)
        self.feature_names_ = feat_names
        y, _ = self._targets_from_prices(prices60, future_prices10)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.model = Ridge(alpha=self.alpha, fit_intercept=True, random_state=42)
        self.model.fit(Xs, y)
        return self

    def predict(self, prices60: np.ndarray, volumes60: np.ndarray) -> np.ndarray:
        """Predict next-10 price levels (shape: n_samples × 10)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X, _, _ = self.feature_builder.build(prices60, volumes60)
        Xs = self.scaler.transform(X)
        yprice = self.model.predict(Xs)
        return yprice

    # ============
    # Diagnostics
    # ============
    def get_feature_names(self) -> List[str]:
        if self.feature_names_ is None:
            raise RuntimeError("Features not built yet. Fit the model first.")
        return list(self.feature_names_)

    @staticmethod
    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    def evaluate(self,
                 prices60_val: np.ndarray,
                 volumes60_val: np.ndarray,
                 future_prices10_val: np.ndarray) -> Dict[str, float]:
        """Compute MAE in return-space and price-space on a validation set."""
        y_true_price, _ = self._targets_from_prices(prices60_val, future_prices10_val)
        base_price = prices60_val[:, -1][:, None]
        y_true_ret = np.diff(np.log(np.concatenate([base_price, y_true_price], axis=1)), axis=1)

        yprice_pred = self.predict(prices60_val, volumes60_val)
        yret_pred = np.diff(np.log(np.concatenate([base_price, yprice_pred], axis=1)), axis=1)

        price_mae = self.mae(yprice_pred, y_true_price)
        ret_mae = self.mae(yret_pred, y_true_ret)
        return {"ret_mae": ret_mae, "price_mae": price_mae}

    def save(self, path: Union[str, Path]) -> Path:
        if self.model is None or self.scaler is None or self.feature_names_ is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        path = Path(path)
        state = {
            "alpha": self.alpha,
            "feature_config": asdict(self.feature_builder.cfg),
            "scaler": self.scaler,
            "model": self.model,
            "feature_names": self.feature_names_,
        }
        joblib.dump(state, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RidgeMIMOPriceForecaster":
        path = Path(path)
        state = joblib.load(path)
        cfg = FeatureConfig(**state["feature_config"])
        inst = cls(alpha=state.get("alpha", 1.0), feature_config=cfg)
        inst.scaler = state["scaler"]
        inst.model = state["model"]
        inst.feature_names_ = state.get("feature_names")
        return inst


DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("model_weights.pkl")


def init_model(weights_path: Optional[Union[str, Path]] = None) -> RidgeMIMOPriceForecaster:
    """Load the trained Ridge model from disk and return a ready-to-use instance."""
    path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found at {path}")
    return RidgeMIMOPriceForecaster.load(path)
