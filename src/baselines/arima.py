"""
Lightweight ARIMA baseline for 60 -> 10 forecasting.

Usage (in quickstart or any script):
------------------------------------
from baselines.arima import ARIMABaseline

model = ARIMABaseline(order=(1,1,0), maxiter=50)
model.fit(x_train_df)  # validates schema & stores config (no global training)
submission = model.predict_x_test(x_test_df)  # -> DataFrame with window_id, time_step, pred_close
submission.to_pickle("sample_submission/submission.pkl")

Notes:
- We fit one ARIMA per window on the 60-step 'close' series, then forecast 10 steps.
- If a window fails to converge, we fall back to a naive-last strategy.
- Requires: statsmodels, pandas, numpy, tqdm.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


@dataclass
class ARIMAConfig:
    order: Tuple[int, int, int] = (1, 1, 0)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    trend: Optional[str] = None
    maxiter: int = 50
    tol: float = 1e-6
    disp: bool = False
    # Fallback behavior
    fallback: str = "naive_last"  # only option implemented
    # Random seed for any stochastic init inside statsmodels (kept for reproducibility)
    seed: int = 1337


class ARIMABaseline:
    """
    Simple ARIMA baseline:
    - 'fit' does schema checks and records configuration (no global parameter training).
    - 'predict_x_test' fits a per-window ARIMA on the last 60 closes and forecasts 10 steps.

    Expected schemas:
    - x_train: columns ['series_id','time_step','close','volume'] (used only for validation)
    - x_test : columns ['window_id','time_step','close','volume'] with time_step=0..59 per window
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 0),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 enforce_stationarity: bool = False,
                 enforce_invertibility: bool = False,
                 trend: Optional[str] = None,
                 maxiter: int = 50,
                 tol: float = 1e-6,
                 disp: bool = False,
                 seed: int = 1337,
                 fallback: str = "naive_last") -> None:
        self.cfg = ARIMAConfig(
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            trend=trend,
            maxiter=maxiter,
            tol=tol,
            disp=disp,
            fallback=fallback,
            seed=seed,
        )
        self._is_fitted = False
        self._train_schema_ok = False

    # -------------------------- public API --------------------------

    def fit(self, x_train: pd.DataFrame) -> "ARIMABaseline":
        """
        Validate x_train schema and mark the model as fitted (no global params for ARIMA).
        """
        self._check_train_schema(x_train)
        self._is_fitted = True
        self._train_schema_ok = True
        return self

    def predict_x_test(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Fit one ARIMA per window on the 60-step close series and forecast 10 steps.

        Returns
        -------
        submission : pd.DataFrame
            Columns: ['window_id','time_step','pred_close']
            dtypes : int32, int8, float32
        """
        if not self._is_fitted:
            warnings.warn("Model not fitted. Calling predict_x_test() without fit() first.")
        self._check_test_schema(x_test)

        out_rows = []
        grouped = x_test.groupby("window_id", sort=False)
        total = x_test["window_id"].nunique()

        for wid, g in tqdm(grouped, total=total, desc="ARIMA infer"):
            g = g.sort_values("time_step")
            # Expect 60 steps per window
            if g["time_step"].nunique() < 60:
                # Skip malformed windows
                continue
            y = g["close"].to_numpy(dtype=np.float64)  # statsmodels expects float64
            # Keep a copy of last value for fallback
            last_close = float(y[-1]) if len(y) > 0 else np.nan

            yhat = self._forecast_10(y, last_close)

            for h in range(len(yhat)):
                out_rows.append({
                    "window_id": np.int32(wid),
                    "time_step": np.int8(h),
                    "pred_close": np.float32(yhat[h]),
                })

        submission = pd.DataFrame(out_rows, columns=["window_id", "time_step", "pred_close"])
        if not submission.empty:
            submission["window_id"] = submission["window_id"].astype("int32")
            submission["time_step"] = submission["time_step"].astype("int8")
            submission["pred_close"] = submission["pred_close"].astype("float32")

            # Basic validation: each window should have 10 steps
            counts = submission.groupby("window_id")["time_step"].nunique()
            if not (counts == 10).all():
                warnings.warn("Some windows did not produce 10 forecast steps.")
        return submission

    # -------------------------- internals --------------------------

    def _forecast_10(self, y: np.ndarray, last_close: float) -> np.ndarray:
        """
        Fit a SARIMAX on y (length 60) and forecast 10 steps.
        On failure, fallback to naive-last.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(
                    endog=y,
                    order=self.cfg.order,
                    seasonal_order=self.cfg.seasonal_order,
                    trend=self.cfg.trend,
                    enforce_stationarity=self.cfg.enforce_stationarity,
                    enforce_invertibility=self.cfg.enforce_invertibility,
                )
                res = mod.fit(
                    disp=self.cfg.disp,
                    maxiter=self.cfg.maxiter,
                    tol=self.cfg.tol,
                )
                fcast = res.forecast(steps=10)
                yhat = np.asarray(fcast, dtype=np.float64)
        except Exception:
            # Fallback strategy
            if self.cfg.fallback == "naive_last" and np.isfinite(last_close):
                yhat = np.full(10, last_close, dtype=np.float64)
            else:
                yhat = np.full(10, np.nan, dtype=np.float64)
        return yhat

    # -------------------------- schema checks --------------------------

    @staticmethod
    def _check_train_schema(df: pd.DataFrame) -> None:
        cols = {"series_id", "time_step", "close", "volume"}
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"x_train missing columns: {sorted(missing)}")
        # dtypes are not strictly enforced here; only presence & basic sanity
        if df.empty:
            raise ValueError("x_train is empty.")

    @staticmethod
    def _check_test_schema(df: pd.DataFrame) -> None:
        cols = {"window_id", "time_step", "close", "volume"}
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"x_test missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("x_test is empty.")