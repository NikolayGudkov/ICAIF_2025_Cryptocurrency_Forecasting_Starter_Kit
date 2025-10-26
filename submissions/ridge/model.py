# regressions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

import os
import math
import numpy as np
import pandas as pd
from time import perf_counter

# Optional dependencies
try:
    import statsmodels.api as sm  # for true (unpenalized) Quantile Regression
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


# ======================================================================================
# Lightweight profiling helpers
# ======================================================================================

_PROFILE_ENABLED = os.getenv("ATHENEA_PROFILE_REGRESSIONS", "0").lower() in {"1", "true", "yes"}


def _profile_log(message: str) -> None:
    if _PROFILE_ENABLED:
        print(f"[regressions profile] {message}", flush=True)




# ======================================================================================
# Public datatypes
# ======================================================================================

@dataclass
class RegressionOutput:
    betas: pd.DataFrame
    alphas: pd.DataFrame
    residuals: pd.DataFrame
    fitted: pd.DataFrame
    info: Dict[str, Union[pd.DataFrame, pd.Series, Dict[str, Union[str, float, int]]]]


# ======================================================================================
# Small, testable utilities
# ======================================================================================

def _align_inputs(
    y: pd.DataFrame, X_list: List[pd.DataFrame], *, how: str = "inner"
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Align y and predictors on a common index & columns.
    """
    if how not in {"inner", "outer"}:
        raise ValueError("align must be 'inner' or 'outer'")
    idx = y.index
    cols = y.columns
    for Xi in X_list:
        idx = idx.intersection(Xi.index) if how == "inner" else idx.union(Xi.index)
        cols = cols.intersection(Xi.columns) if how == "inner" else cols.union(Xi.columns)
    y2 = y.reindex(index=idx, columns=cols)
    X2 = [Xi.reindex(index=idx, columns=cols) for Xi in X_list]
    return y2, X2


def _default_names(X_list: List[pd.DataFrame]) -> List[str]:
    """
    Create unique predictor names. Prefer DataFrame.name if present.
    """
    names = []
    for i, Xi in enumerate(X_list):
        n = getattr(Xi, "name", None) or f"x{i + 1}"
        names.append(str(n))
    if len(set(names)) != len(names):
        names = [f"x{i + 1}" for i in range(len(X_list))]
    return names


def _add_intercept(X: np.ndarray, add: bool) -> np.ndarray:
    if not add:
        return X
    if X.ndim == 2:
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.column_stack([ones, X])
    if X.ndim == 3:
        ones = np.ones((X.shape[0], X.shape[1], 1), dtype=float)
        return np.concatenate([ones, X], axis=2)
    raise ValueError("Unsupported array dimension for intercept augmentation.")


def _finite_mask_yX(y: np.ndarray, X: np.ndarray, skip_col0: bool) -> np.ndarray:
    """
    Mask rows where y or any predictor (optionally skipping the intercept col 0) is NaN/inf.
    """
    m = np.isfinite(y)
    start = 1 if skip_col0 else 0
    if X.shape[1] > start:
        m &= np.all(np.isfinite(X[:, start:]), axis=1)
    return m


def _window_iter(
    index: pd.Index,
    rolling_window: Optional[Union[int, str]],
    stride_window: Optional[int] = None,
) -> Iterable[Tuple[Union[pd.Timestamp, str], np.ndarray]]:
    """
    Yield (label, row_mask) pairs for each estimation window.

    - None  -> a single window covering all rows; label = last index
    - int W -> rolling window of width W; label = each timestamp
    - 'per_halves' -> half-year groups ('YYYYH1'/'YYYYH2'); label = group key
    """
    n = len(index)
    stride = 1 if not isinstance(stride_window, int) or stride_window is None or stride_window < 1 else int(stride_window)
    if rolling_window is None:
        mask = np.ones(n, dtype=bool)
        yield index[-1], mask
        return

    if isinstance(rolling_window, int):
        W = int(rolling_window)
        W = max(1, W)
        for t in range(0, n, stride):
            start = max(0, t - W + 1)
            mask = np.zeros(n, dtype=bool)
            mask[start : t + 1] = True
            yield index[t], mask
        return

    if isinstance(rolling_window, str) and rolling_window == "per_halves":
        idx_dt = pd.DatetimeIndex(index)
        labels = np.array([f"{d.year}H{1 if d.month <= 6 else 2}" for d in idx_dt])
        unique_labels = pd.unique(labels)
        for i in range(0, len(unique_labels), stride):
            lab = unique_labels[i]
            yield lab, labels == lab
        return

    raise ValueError("Unsupported rolling_window. Use None, int, or 'per_halves'.")


def _ess_scale(resid: np.ndarray) -> float:
    """
    Effective sample size scale (Lag-1 autocorr) used to inflate standard errors.
    """
    u = resid - np.nanmean(resid)
    u = u[np.isfinite(u)]
    n = len(u)
    if n < 3:
        return 1.0
    u0 = u[:-1]
    u1 = u[1:]
    denom = float(np.dot(u0, u0))
    rho = float(np.dot(u0, u1) / denom) if denom > 0 else 0.0
    rho = max(min(rho, 0.999), -0.999)
    n_eff = n * (1.0 - rho) / (1.0 + rho)
    n_eff = max(1.0, min(n, n_eff))
    return math.sqrt(n / n_eff)


def _hac_covariance(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """
    Newey–West HAC 'meat' of the sandwich covariance: sum of lagged outer products of x_t u_t.
    """
    n, p = X.shape
    S = (X.T * resid)  # p x n
    S0 = S @ S.T
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u_t = resid[L:]
        u_tL = resid[:-L]
        X_t = X[L:]
        X_tL = X[:-L]
        S1 = (X_t.T * u_t) @ (X_tL * u_tL[:, None])
        S2 = (X_tL.T * u_tL) @ (X_t * u_t[:, None])
        S0 += w * (S1 + S2)
    return S0 / n


def _choose_nw_lags(n: int) -> int:
    """
    Default Newey–West lag heuristic: floor(4 * (n/100)^(2/9)), min 1.
    """
    L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(1, L)


# ======================================================================================
# OLS (with ridge option) core
# ======================================================================================

def _ridge_bread(X: np.ndarray, l2: float, penalize_mask: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    if l2 > 0.0:
        return XtX + l2 * np.diag(penalize_mask)
    return XtX


def _solve_ridge(X: np.ndarray, y: np.ndarray, l2: float, penalize_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return beta, fitted, SSE on training set.
    """
    B = _ridge_bread(X, l2, penalize_mask)
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(B, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(B, Xty, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    sse = float(resid.T @ resid)
    return beta, resid, sse


def _ols_cov_se_t(X: np.ndarray,
                  resid: np.ndarray,
                  beta: np.ndarray,
                  l2: float,
                  penalize_mask: np.ndarray,
                  se_mode: str = "sample",
                  nw_lags: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (stderr, tstats) for beta.
    """
    n, p = X.shape
    B = _ridge_bread(X, l2, penalize_mask)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B)

    if se_mode == "newey_west":
        L = _choose_nw_lags(n) if nw_lags is None else int(nw_lags)
        meat = _hac_covariance(X, resid, L)
        cov = B_inv @ meat @ B_inv
    else:
        # Homoskedastic OLS-like covariance with ridge bread
        XtX = X.T @ X
        df_model = float(np.trace(XtX @ B_inv))
        df_resid = max(1.0, n - df_model)
        s2 = float(resid.T @ resid) / df_resid
        cov = s2 * (B_inv @ XtX @ B_inv)
        if se_mode == "ess":
            cov *= _ess_scale(resid) ** 2

    cov = 0.5 * (cov + cov.T)  # symmetrize
    var = np.clip(np.diag(cov), 0.0, np.inf)
    stderr = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = beta / stderr
    return stderr, tstats


# ======================================================================================
# Quantile Regression core (unpenalized if statsmodels is available)
# ======================================================================================

def _quantile_regression_unpenalized(y: np.ndarray, X: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use statsmodels QuantReg and return (params, bse, tstats).
    X must already include intercept if desired.
    """
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels is required for quantile_regression")
    # statsmodels expects intercept already in X if you want one
    model = sm.QuantReg(y, X)
    # Request robust covariance; handle API changes across statsmodels versions
    try:
        res = model.fit(q=q, vcov="robust")
    except TypeError:
        res = model.fit(q=q, cov="robust")
    params = np.asarray(res.params, dtype=float)
    # Obtain standard errors robustly
    if getattr(res, "bse", None) is not None:
        bse = np.asarray(res.bse, dtype=float)
    else:
        try:
            cov = np.asarray(res.cov_params(), dtype=float)
            cov = 0.5 * (cov + cov.T)
            var = np.clip(np.diag(cov), 0.0, np.inf)
            bse = np.sqrt(var)
        except Exception:
            bse = np.full_like(params, np.nan, dtype=float)
    # t-stats
    if getattr(res, "tvalues", None) is not None:
        tvals = np.asarray(res.tvalues, dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = np.divide(params, bse, out=np.full_like(params, np.nan, dtype=float), where=np.isfinite(bse))
    return params, bse, tvals


def _quantile_fit(y: np.ndarray, X: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (coefficients, std_errors, t_stats) for τ-quantile using statsmodels QuantReg.
    X may include intercept.
    """
    return _quantile_regression_unpenalized(y, X, q)


def _check_quantiles(q_spec: Union[float, Iterable[float]]) -> List[float]:
    if isinstance(q_spec, (list, tuple, np.ndarray, pd.Series)):
        qs = [float(q) for q in q_spec]
    else:
        qs = [float(q_spec)]
    for q in qs:
        if not (0.0 < q < 1.0):
            raise ValueError("Quantiles must satisfy 0 < q < 1")
    # unique + sorted
    return sorted(set(qs))


def _quantile_pseudo_r2(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    """
    Koenker-Machado pseudo-R^2: 1 - ρτ(y - yhat) / ρτ(y - θ_q),
    with θ_q = sample q-th quantile and ρτ(u) = sum u * (τ - I[u < 0]).
    """
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.nan
    yy = y[m]
    yhat_m = yhat[m]
    def rho(u: np.ndarray, tau: float) -> float:
        return float(np.sum(u * (tau - (u < 0).astype(float))))
    loss = rho(yy - yhat_m, q)
    theta = np.quantile(yy, q)
    loss0 = rho(yy - theta, q)
    if loss0 == 0:
        return np.nan
    return 1.0 - (loss / loss0)


# ======================================================================================
# Data shaping helpers
# ======================================================================================

def _precompute_arrays(
    y: pd.DataFrame, X_list: List[pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (y_arr: T x A, X_pred: T x A x K)
    """
    t_start = perf_counter()
    y_arr = y.to_numpy(dtype=float)
    X_pred = np.stack([Xi.to_numpy(dtype=float) for Xi in X_list], axis=2)
    _profile_log(
        "_precompute_arrays: "
        f"y_arr_shape={y_arr.shape}, X_pred_shape={X_pred.shape}, "
        f"elapsed={perf_counter() - t_start:.4f}s"
    )
    return y_arr, X_pred


def _stack_global_window(
    y_arr: np.ndarray,
    X_pred: np.ndarray,
    rows: np.ndarray,
    add_intercept: bool,
    min_obs_per_column: Optional[int],
    *,
    finite_mask_all: Optional[np.ndarray] = None,
    X_full_all: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack across assets for a set of time rows.
    Returns (y_vec, X_mat, t_idx_of_obs, a_idx_of_obs) for mapping predictions.
    Assets with fewer than `min_obs_per_column` valid observations inside the window are dropped.
    """
    t_start = perf_counter()
    _, _, K = X_pred.shape
    p = K + int(add_intercept)
    row_idx = np.flatnonzero(rows)
    row_count = int(row_idx.size)
    if row_count == 0:
        return np.empty((0,), float), np.empty((0, p), dtype=float), np.empty((0,), int), np.empty((0,), int)
    Y_win = y_arr[row_idx, :]  # rows x assets
    X_win = X_pred[row_idx, :, :]  # rows x assets x K
    if finite_mask_all is not None:
        finite_mask = finite_mask_all[row_idx, :]
    else:
        finite_mask = np.isfinite(Y_win) & np.all(np.isfinite(X_win), axis=2)
    if not np.any(finite_mask):
        return np.empty((0,), float), np.empty((0, p), dtype=float), np.empty((0,), int), np.empty((0,), int)

    asset_lookup = np.arange(Y_win.shape[1])
    if min_obs_per_column is not None:
        min_obs = max(int(min_obs_per_column), 1)
        obs_counts = finite_mask.sum(axis=0)
        valid_assets_mask = obs_counts >= min_obs
        if not np.any(valid_assets_mask):
            return (
                np.empty((0,), float),
                np.empty((0, p), dtype=float),
                np.empty((0,), int),
                np.empty((0,), int),
            )
        finite_mask = finite_mask[:, valid_assets_mask]
        Y_win = Y_win[:, valid_assets_mask]
        X_win = X_win[:, valid_assets_mask, :]
        asset_lookup = np.where(valid_assets_mask)[0]
        if X_full_all is not None:
            X_full_win = X_full_all[row_idx][:, asset_lookup, :]
        else:
            X_full_win = _add_intercept(X_win, add_intercept)
    else:
        if X_full_all is not None:
            X_full_win = X_full_all[row_idx]
        else:
            X_full_win = _add_intercept(X_win, add_intercept)

    if not np.any(finite_mask):
        return np.empty((0,), float), np.empty((0, p), dtype=float), np.empty((0,), int), np.empty((0,), int)

    mask_flat = finite_mask.reshape(-1)
    if not np.any(mask_flat):
        return np.empty((0,), float), np.empty((0, p), dtype=float), np.empty((0,), int), np.empty((0,), int)

    y_vec = Y_win.reshape(-1)[mask_flat]
    X_flat = X_full_win.reshape(-1, p)
    X_mat = X_flat[mask_flat, :]

    counts = finite_mask.sum(axis=1).astype(int)
    t_idx = np.repeat(row_idx, counts)
    asset_indices = np.broadcast_to(asset_lookup, finite_mask.shape)
    a_idx = asset_indices[finite_mask].astype(int)
    _profile_log(
        "_stack_global_window: "
        f"rows={row_count}, obs={y_vec.shape[0]}, add_intercept={add_intercept}, "
        f"elapsed={perf_counter() - t_start:.4f}s"
    )
    return y_vec, X_mat, t_idx, a_idx


# ======================================================================================
# Shared configuration helpers
# ======================================================================================

def _prepare_linear_inputs(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    *,
    align: str,
    add_intercept: bool,
    rolling_window: Optional[Union[int, str]],
    min_periods: Optional[int],
    min_obs: Optional[int],
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str], np.ndarray, int, Optional[int]]:
    """
    Align inputs, determine predictor names, ridge penalization mask, and the minimum
    number of observations required for estimation.
    """
    t_start = perf_counter()
    y_aligned, X_list = _align_inputs(y, predictors, how=align)
    xnames = _default_names(X_list)

    K = len(X_list)
    p = K + (1 if add_intercept else 0)

    if min_periods is not None:
        min_required = int(min_periods)
    elif isinstance(rolling_window, int):
        min_required = max(1, int(rolling_window) // 2)
    elif isinstance(rolling_window, str) and rolling_window == "per_halves":
        min_required = max(10, p + 5)
    elif min_obs is not None:
        min_required = int(min_obs)
    else:
        min_required = max(10, p + 5)
    min_required = max(min_required, p + 1)

    if add_intercept:
        penalize_mask = np.concatenate([[0.0], np.ones(K, dtype=float)])
    else:
        penalize_mask = np.ones(K, dtype=float)

    effective_min_periods = (
        int(rolling_window) // 2 if (min_periods is None and isinstance(rolling_window, int)) else min_periods
    )

    _profile_log(
        "_prepare_linear_inputs: "
        f"align={align}, add_intercept={add_intercept}, K={K}, p={p}, "
        f"y_shape={y_aligned.shape}, min_required={min_required}, window={rolling_window}, "
        f"elapsed={perf_counter() - t_start:.4f}s"
    )

    return y_aligned, X_list, xnames, penalize_mask, min_required, effective_min_periods


# ======================================================================================
# Main engines: OLS and Quantile, per-asset and global
# ======================================================================================


def _prefix_sum(arr: np.ndarray) -> np.ndarray:
    """
    Return prefix sums with a zero row prepended for constant-time window queries.
    """
    return np.concatenate(
        [np.zeros((1, arr.shape[1]), dtype=np.float64), np.cumsum(arr, axis=0, dtype=np.float64)],
        axis=0,
    )


def _prefix_sum_1d(arr: np.ndarray) -> np.ndarray:
    """
    Return prefix sum for 1-D array with leading zero.
    """
    return np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])


def _ols_per_asset_single_factor_fast(
    y: pd.DataFrame,
    X_list: List[pd.DataFrame],
    xnames: List[str],
    *,
    rolling_window: int,
    stride_window: Optional[int],
    min_required: int,
) -> RegressionOutput:
    """
    Fast path for per-asset rolling OLS with a single predictor and sample SEs.
    """
    T, A = y.shape
    stride = 1 if stride_window is None or stride_window < 1 else int(stride_window)
    W = int(rolling_window)

    idx_labels = y.index
    predictor_name = xnames[0] if xnames else "x1"
    beta_cols = pd.MultiIndex.from_product([y.columns, [predictor_name]], names=["asset", "predictor"])

    y_arr = y.to_numpy(dtype=float)
    x_arr = X_list[0].to_numpy(dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)

    # Prefix sums for efficient window statistics
    mask_f = mask.astype(np.float64)
    c_mask = _prefix_sum(mask_f)
    x_valid = np.where(mask, x_arr, 0.0)
    y_valid = np.where(mask, y_arr, 0.0)
    xy_valid = np.where(mask, x_arr * y_arr, 0.0)
    x2_valid = np.where(mask, x_arr * x_arr, 0.0)
    y2_valid = np.where(mask, y_arr * y_arr, 0.0)

    c_sum_x = _prefix_sum(x_valid)
    c_sum_y = _prefix_sum(y_valid)
    c_sum_xy = _prefix_sum(xy_valid)
    c_sum_x2 = _prefix_sum(x2_valid)
    c_sum_y2 = _prefix_sum(y2_valid)

    fitted_arr = np.full((T, A), np.nan, dtype=float)
    resid_arr = np.full((T, A), np.nan, dtype=float)

    labels_out: List[pd.Timestamp] = []
    r2_rows: List[np.ndarray] = []
    adjr2_rows: List[np.ndarray] = []
    nobs_rows: List[np.ndarray] = []
    df_model_rows: List[np.ndarray] = []
    df_resid_rows: List[np.ndarray] = []
    corr_fit_rows: List[np.ndarray] = []
    alphas_rows: List[np.ndarray] = []
    stderrs_alpha_rows: List[np.ndarray] = []
    tstats_alpha_rows: List[np.ndarray] = []
    betas_rows: List[np.ndarray] = []
    stderrs_beta_rows: List[np.ndarray] = []
    tstats_beta_rows: List[np.ndarray] = []
    corr_pred_rows: List[np.ndarray] = []

    evaluated = 0
    computed = 0
    skipped_empty = 0
    skipped_insufficient = 0

    tol = 1e-12

    for t in range(0, T, stride):
        evaluated += 1
        start = 0 if t - W + 1 < 0 else t - W + 1

        n = c_mask[t + 1] - c_mask[start]
        if not np.any(n > 0.0):
            skipped_empty += 1
            continue

        sum_x = c_sum_x[t + 1] - c_sum_x[start]
        sum_y = c_sum_y[t + 1] - c_sum_y[start]
        sum_xy = c_sum_xy[t + 1] - c_sum_xy[start]
        sum_x2 = c_sum_x2[t + 1] - c_sum_x2[start]
        sum_y2 = c_sum_y2[t + 1] - c_sum_y2[start]

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_x = np.divide(sum_x, n, out=np.zeros_like(sum_x), where=n > 0.0)
            mean_y = np.divide(sum_y, n, out=np.zeros_like(sum_y), where=n > 0.0)

        cov_xy = sum_xy - mean_x * sum_y
        var_x = sum_x2 - sum_x * mean_x
        var_y = sum_y2 - sum_y * mean_y

        valid = (n >= min_required) & (var_x > tol)
        valid &= n >= 2.0
        if not np.any(valid):
            skipped_insufficient += 1
            continue

        beta = np.full(A, np.nan, dtype=float)
        alpha = np.full(A, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta[valid] = cov_xy[valid] / var_x[valid]
            alpha[valid] = mean_y[valid] - beta[valid] * mean_x[valid]

        SSE = np.full(A, np.nan, dtype=float)
        SSE_valid = (
            sum_y2[valid]
            - 2.0 * beta[valid] * sum_xy[valid]
            + beta[valid] ** 2 * sum_x2[valid]
            - 2.0 * alpha[valid] * sum_y[valid]
            + 2.0 * alpha[valid] * beta[valid] * sum_x[valid]
            + alpha[valid] ** 2 * n[valid]
        )
        SSE[valid] = np.clip(SSE_valid, 0.0, np.inf)

        TSS = np.full(A, np.nan, dtype=float)
        TSS_valid = var_y[valid]
        TSS[valid] = np.clip(TSS_valid, 0.0, np.inf)

        r2 = np.full(A, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2[valid] = 1.0 - np.divide(SSE[valid], TSS[valid], out=np.zeros_like(SSE[valid]), where=TSS[valid] > 0.0)

        df_resid = np.full(A, np.nan, dtype=float)
        df_resid[valid] = n[valid] - 2.0
        positive_df = valid & (df_resid > 0.0)

        s2 = np.full(A, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            s2[positive_df] = SSE[positive_df] / df_resid[positive_df]

        stderr_beta = np.full(A, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            stderr_beta[positive_df] = np.sqrt(s2[positive_df] / var_x[positive_df])

        stderr_alpha = np.full(A, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            stderr_alpha[positive_df] = np.sqrt(
                s2[positive_df] * (1.0 / n[positive_df] + (mean_x[positive_df] ** 2) / var_x[positive_df])
            )

        t_beta = np.full(A, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_beta[positive_df] = beta[positive_df] / stderr_beta[positive_df]

        t_alpha = np.full(A, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_alpha[positive_df] = alpha[positive_df] / stderr_alpha[positive_df]

        adj_r2 = np.full(A, np.nan, dtype=float)
        denom_adj = (n - 1.0)
        mask_adj = positive_df & (TSS > 0.0) & (denom_adj > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            adj_r2[mask_adj] = 1.0 - (SSE[mask_adj] / df_resid[mask_adj]) / (TSS[mask_adj] / denom_adj[mask_adj])

        corr_pred = np.full(A, np.nan, dtype=float)
        denom_corr = np.sqrt(var_x * var_y)
        mask_corr = valid & (denom_corr > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_pred[mask_corr] = cov_xy[mask_corr] / denom_corr[mask_corr]

        corr_fit = np.full(A, np.nan, dtype=float)
        corr_fit[valid] = np.sqrt(np.clip(r2[valid], 0.0, 1.0))

        # Update fitted/residual arrays at the window boundary
        mask_last = mask[t]
        valid_last = valid & mask_last
        if np.any(valid_last):
            x_last = x_arr[t, valid_last]
            y_last = y_arr[t, valid_last]
            yhat_last = alpha[valid_last] + beta[valid_last] * x_last
            fitted_arr[t, valid_last] = yhat_last
            mask_y_last = np.isfinite(y_last)
            if np.any(mask_y_last):
                idx_valid = np.where(valid_last)[0]
                idx_keep = idx_valid[mask_y_last]
                resid_arr[t, idx_keep] = y_last[mask_y_last] - yhat_last[mask_y_last]

        # Only keep windows with at least one valid asset
        if not np.any(valid):
            skipped_insufficient += 1
            continue

        labels_out.append(idx_labels[t])
        computed += 1

        def _row(values: np.ndarray) -> np.ndarray:
            row = np.full(A, np.nan, dtype=float)
            row[valid] = values[valid]
            return row

        r2_rows.append(_row(r2))
        adjr2_rows.append(_row(adj_r2))
        nobs_rows.append(_row(n))

        df_model_row = np.full(A, np.nan, dtype=float)
        df_model_row[valid] = 2.0
        df_model_rows.append(df_model_row)
        df_resid_rows.append(_row(df_resid))
        corr_fit_rows.append(_row(corr_fit))
        alphas_rows.append(_row(alpha))
        stderrs_alpha_rows.append(_row(stderr_alpha))
        tstats_alpha_rows.append(_row(t_alpha))

        betas_matrix = np.full((A, 1), np.nan, dtype=float)
        stderr_beta_matrix = np.full((A, 1), np.nan, dtype=float)
        t_beta_matrix = np.full((A, 1), np.nan, dtype=float)
        corr_pred_matrix = np.full((A, 1), np.nan, dtype=float)

        betas_matrix[valid, 0] = beta[valid]
        stderr_beta_matrix[valid, 0] = stderr_beta[valid]
        t_beta_matrix[valid, 0] = t_beta[valid]
        corr_pred_matrix[valid, 0] = corr_pred[valid]

        betas_rows.append(betas_matrix)
        stderrs_beta_rows.append(stderr_beta_matrix)
        tstats_beta_rows.append(t_beta_matrix)
        corr_pred_rows.append(corr_pred_matrix)

    idx_out = pd.Index(labels_out, name=y.index.name)

    betas_out = (
        pd.DataFrame(
            np.vstack([row.reshape(1, -1) for row in betas_rows]),
            index=idx_out,
            columns=beta_cols,
            dtype=float,
        )
        if betas_rows
        else pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    )

    stderrs_b = (
        pd.DataFrame(
            np.vstack([row.reshape(1, -1) for row in stderrs_beta_rows]),
            index=idx_out,
            columns=beta_cols,
            dtype=float,
        )
        if stderrs_beta_rows
        else pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    )
    tstats_b = (
        pd.DataFrame(
            np.vstack([row.reshape(1, -1) for row in tstats_beta_rows]),
            index=idx_out,
            columns=beta_cols,
            dtype=float,
        )
        if tstats_beta_rows
        else pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    )
    corr_pred_df = (
        pd.DataFrame(
            np.vstack([row.reshape(1, -1) for row in corr_pred_rows]),
            index=idx_out,
            columns=beta_cols,
            dtype=float,
        )
        if corr_pred_rows
        else pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    )

    r2_df = pd.DataFrame(r2_rows, index=idx_out, columns=y.columns, dtype=float) if r2_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    adjr2_df = pd.DataFrame(adjr2_rows, index=idx_out, columns=y.columns, dtype=float) if adjr2_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    nobs_df = pd.DataFrame(nobs_rows, index=idx_out, columns=y.columns, dtype=float) if nobs_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    df_model_df = pd.DataFrame(df_model_rows, index=idx_out, columns=y.columns, dtype=float) if df_model_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    df_resid_df = pd.DataFrame(df_resid_rows, index=idx_out, columns=y.columns, dtype=float) if df_resid_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    corr_fit_df = pd.DataFrame(corr_fit_rows, index=idx_out, columns=y.columns, dtype=float) if corr_fit_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)

    alphas_out = pd.DataFrame(alphas_rows, index=idx_out, columns=y.columns, dtype=float) if alphas_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    stderrs_a = pd.DataFrame(stderrs_alpha_rows, index=idx_out, columns=y.columns, dtype=float) if stderrs_alpha_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    tstats_a = pd.DataFrame(tstats_alpha_rows, index=idx_out, columns=y.columns, dtype=float) if tstats_alpha_rows else pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)

    window_summary = pd.Series(
        {
            "evaluated": float(evaluated),
            "computed": float(computed),
            "skipped_empty": float(skipped_empty),
            "skipped_insufficient": float(skipped_insufficient),
        },
        dtype=float,
        name="rolling_ols",
    )

    info = {
        "r2": r2_df,
        "adj_r2": adjr2_df,
        "n_obs": nobs_df,
        "df_model": df_model_df,
        "df_resid": df_resid_df,
        "tstats_beta": tstats_b,
        "stderr_beta": stderrs_b,
        "tstats_alpha": tstats_a,
        "stderr_alpha": stderrs_a,
        "corr": corr_pred_df,
        "corr_fitted": corr_fit_df,
        "window_summary": window_summary,
    }

    fitted_df = pd.DataFrame(fitted_arr, index=y.index, columns=y.columns, dtype=float)
    resid_df = pd.DataFrame(resid_arr, index=y.index, columns=y.columns, dtype=float)

    return RegressionOutput(betas=betas_out, alphas=alphas_out, residuals=resid_df, fitted=fitted_df, info=info)


def _ols_global_single_factor_fast(
    y: pd.DataFrame,
    X_list: List[pd.DataFrame],
    xnames: List[str],
    *,
    rolling_window: int,
    stride_window: Optional[int],
    min_required: int,
) -> RegressionOutput:
    """
    Fast path for global rolling OLS with a single predictor and sample SEs.
    """
    T, A = y.shape
    stride = 1 if stride_window is None or stride_window < 1 else int(stride_window)
    W = int(rolling_window)

    predictor_name = xnames[0] if xnames else "x1"
    beta_cols = pd.Index([predictor_name], name="predictor")

    y_arr = y.to_numpy(dtype=float)
    x_arr = X_list[0].to_numpy(dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)

    mask_float = mask.astype(np.float64)
    prefix_counts = _prefix_sum(mask_float)
    prefix_sum_x = _prefix_sum(np.where(mask, x_arr, 0.0))
    prefix_sum_y = _prefix_sum(np.where(mask, y_arr, 0.0))
    prefix_sum_xy = _prefix_sum(np.where(mask, x_arr * y_arr, 0.0))
    prefix_sum_x2 = _prefix_sum(np.where(mask, x_arr * x_arr, 0.0))
    prefix_sum_y2 = _prefix_sum(np.where(mask, y_arr * y_arr, 0.0))

    fitted_arr = np.full((T, A), np.nan, dtype=float)
    resid_arr = np.full((T, A), np.nan, dtype=float)

    labels_out: List[pd.Timestamp] = []
    beta_values: List[float] = []
    alpha_values: List[float] = []
    stderr_beta_values: List[float] = []
    stderr_alpha_values: List[float] = []
    t_beta_values: List[float] = []
    t_alpha_values: List[float] = []
    r2_values: List[float] = []
    adjr2_values: List[float] = []
    nobs_values: List[float] = []
    df_model_values: List[float] = []
    df_resid_values: List[float] = []
    corr_fit_values: List[float] = []
    corr_pred_values: List[float] = []

    tol = 1e-12
    evaluated = 0
    computed = 0
    skipped_empty = 0
    skipped_insufficient = 0

    index_vals = y.index

    for t in range(0, T, stride):
        evaluated += 1
        start = 0 if t - W + 1 < 0 else t - W + 1

        counts = prefix_counts[t + 1] - prefix_counts[start]
        valid_assets = counts >= min_required
        if not np.any(valid_assets):
            skipped_empty += 1
            continue

        counts_valid = counts[valid_assets]
        n = float(np.sum(counts_valid))
        if n <= 0.0:
            skipped_empty += 1
            continue

        min_obs_total = max(min_required, 3.0)
        if n < min_obs_total:
            skipped_insufficient += 1
            continue

        sum_x_all = prefix_sum_x[t + 1] - prefix_sum_x[start]
        sum_y_all = prefix_sum_y[t + 1] - prefix_sum_y[start]
        sum_xy_all = prefix_sum_xy[t + 1] - prefix_sum_xy[start]
        sum_x2_all = prefix_sum_x2[t + 1] - prefix_sum_x2[start]
        sum_y2_all = prefix_sum_y2[t + 1] - prefix_sum_y2[start]

        sum_x = float(np.sum(sum_x_all[valid_assets]))
        sum_y = float(np.sum(sum_y_all[valid_assets]))
        sum_xy = float(np.sum(sum_xy_all[valid_assets]))
        sum_x2 = float(np.sum(sum_x2_all[valid_assets]))
        sum_y2 = float(np.sum(sum_y2_all[valid_assets]))

        mean_x = sum_x / n
        mean_y = sum_y / n

        cov_xy = sum_xy - sum_x * mean_y
        var_x = sum_x2 - sum_x * mean_x
        var_y = sum_y2 - sum_y * mean_y

        if var_x <= tol or n < 2.0:
            skipped_insufficient += 1
            continue

        beta = cov_xy / var_x
        alpha = mean_y - beta * mean_x

        # Residual sum of squares
        SSE = (
            sum_y2
            - 2.0 * alpha * sum_y
            - 2.0 * beta * sum_xy
            + 2.0 * alpha * beta * sum_x
            + beta * beta * sum_x2
            + alpha * alpha * n
        )
        SSE = float(np.clip(SSE, 0.0, np.inf))

        TSS = float(np.clip(var_y, 0.0, np.inf))
        if TSS <= tol:
            r2 = np.nan
        else:
            r2 = 1.0 - (SSE / TSS)

        df_model = 2.0
        df_resid = max(1.0, n - df_model)
        s2 = SSE / df_resid

        stderr_beta = math.sqrt(s2 / var_x)
        stderr_alpha = math.sqrt(s2 * (1.0 / n + (mean_x * mean_x) / var_x))

        t_beta = beta / stderr_beta if stderr_beta > 0.0 else np.nan
        t_alpha = alpha / stderr_alpha if stderr_alpha > 0.0 else np.nan

        denom_adj = n - 1.0
        if TSS > tol and denom_adj > 0.0:
            adj_r2 = 1.0 - (SSE / df_resid) / (TSS / denom_adj)
        else:
            adj_r2 = np.nan

        denom_corr = math.sqrt(max(var_x * var_y, 0.0))
        if denom_corr > 0.0:
            corr_pred = cov_xy / denom_corr
        else:
            corr_pred = np.nan

        corr_fit = math.sqrt(max(r2, 0.0)) if r2 == r2 else np.nan

        labels_out.append(index_vals[t])
        beta_values.append(beta)
        alpha_values.append(alpha)
        stderr_beta_values.append(stderr_beta)
        stderr_alpha_values.append(stderr_alpha)
        t_beta_values.append(t_beta)
        t_alpha_values.append(t_alpha)
        r2_values.append(r2)
        adjr2_values.append(adj_r2)
        nobs_values.append(float(n))
        df_model_values.append(df_model)
        df_resid_values.append(df_resid)
        corr_fit_values.append(corr_fit)
        corr_pred_values.append(corr_pred)
        computed += 1

        # Fitted/residuals at the window label (last row)
        mask_last = mask[t] & valid_assets
        if np.any(mask_last):
            idx_last = np.where(mask_last)[0]
            x_last = x_arr[t, idx_last]
            yhat_last = alpha + beta * x_last
            fitted_arr[t, idx_last] = yhat_last
            y_last = y_arr[t, idx_last]
            mask_y_last = np.isfinite(y_last)
            if np.any(mask_y_last):
                resid_arr[t, idx_last[mask_y_last]] = y_last[mask_y_last] - yhat_last[mask_y_last]

    idx_out = pd.Index(labels_out, name=y.index.name)

    betas_out = pd.DataFrame(beta_values, index=idx_out, columns=beta_cols, dtype=float)
    stderrs_b = pd.DataFrame(stderr_beta_values, index=idx_out, columns=beta_cols, dtype=float)
    tstats_b = pd.DataFrame(t_beta_values, index=idx_out, columns=beta_cols, dtype=float)
    corr_df = pd.DataFrame(corr_pred_values, index=idx_out, columns=beta_cols, dtype=float)

    alpha_df = pd.DataFrame(alpha_values, index=idx_out, columns=["alpha"], dtype=float)
    stderrs_a = pd.DataFrame(stderr_alpha_values, index=idx_out, columns=["alpha"], dtype=float)
    tstats_a = pd.DataFrame(t_alpha_values, index=idx_out, columns=["alpha"], dtype=float)

    def _series(data: List[float]) -> pd.DataFrame:
        return pd.DataFrame(data, index=idx_out, columns=["global"], dtype=float)

    r2_df = _series(r2_values)
    adjr2_df = _series(adjr2_values)
    nobs_df = _series(nobs_values)
    df_model_df = _series(df_model_values)
    df_resid_df = _series(df_resid_values)
    corr_fit_df = _series(corr_fit_values)

    window_summary = pd.Series(
        {
            "evaluated": float(evaluated),
            "computed": float(computed),
            "skipped_empty": float(skipped_empty),
            "skipped_insufficient": float(skipped_insufficient),
        },
        dtype=float,
        name="rolling_ols",
    )

    info = {
        "r2": r2_df,
        "adj_r2": adjr2_df,
        "n_obs": nobs_df,
        "df_model": df_model_df,
        "df_resid": df_resid_df,
        "tstats_beta": tstats_b,
        "stderr_beta": stderrs_b,
        "tstats_alpha": tstats_a,
        "stderr_alpha": stderrs_a,
        "corr": corr_df,
        "corr_fitted": corr_fit_df,
        "window_summary": window_summary,
    }

    fitted_df = pd.DataFrame(fitted_arr, index=y.index, columns=y.columns, dtype=float)
    resid_df = pd.DataFrame(resid_arr, index=y.index, columns=y.columns, dtype=float)

    return RegressionOutput(betas=betas_out, alphas=alpha_df, residuals=resid_df, fitted=fitted_df, info=info)


def _ols_per_asset(
    y: pd.DataFrame,
    X_list: List[pd.DataFrame],
    xnames: List[str],
    *,
    rolling_window: Optional[Union[int, str]],
    stride_window: Optional[int],
    add_intercept: bool,
    l2: float,
    se_mode: str,
    nw_lags: Optional[int],
    min_required: int,
    penalize_mask: np.ndarray,
) -> RegressionOutput:
    T, A = y.shape
    K = len(X_list)
    t_total = perf_counter()
    _profile_log(
        "_ols_per_asset: "
        f"start T={T}, A={A}, K={K}, rolling_window={rolling_window}, stride_window={stride_window}"
    )

    if (
        isinstance(rolling_window, int)
        and K == 1
        and add_intercept
        and l2 == 0.0
        and se_mode == "sample"
        and nw_lags is None
    ):
        return _ols_per_asset_single_factor_fast(
            y,
            X_list,
            xnames,
            rolling_window=int(rolling_window),
            stride_window=stride_window,
            min_required=min_required,
        )

    if rolling_window is None:
        idx_out = pd.Index([y.index[-1]], name=y.index.name)
    elif isinstance(rolling_window, int):
        idx_out = pd.Index(y.index, name=y.index.name)
    else:
        idx_out = pd.Index([], dtype=object)

    if isinstance(rolling_window, str) and rolling_window == "per_halves":
        idx_dt = pd.DatetimeIndex(y.index)
        halves = pd.unique([f"{d.year}H{1 if d.month <= 6 else 2}" for d in idx_dt])
        idx_out = pd.Index(halves, name="half")

    beta_cols = pd.MultiIndex.from_product([y.columns, xnames], names=["asset", "predictor"])

    n_labels = len(idx_out)
    n_pred = len(xnames)

    if n_labels:
        if idx_out.has_duplicates:
            label_to_pos: Dict[Union[pd.Timestamp, str], List[int]] = {}
            for pos, key in enumerate(idx_out):
                label_to_pos.setdefault(key, []).append(pos)
        else:
            label_to_pos = {key: [pos] for pos, key in enumerate(idx_out)}
    else:
        label_to_pos = {}

    fitted_arr = np.full((T, A), np.nan, dtype=float)
    resid_arr = np.full((T, A), np.nan, dtype=float)

    r2_arr = np.full((n_labels, A), np.nan, dtype=float)
    adjr2_arr = np.full((n_labels, A), np.nan, dtype=float)
    nobs_arr = np.full((n_labels, A), np.nan, dtype=float)
    df_model_arr = np.full((n_labels, A), np.nan, dtype=float)
    df_resid_arr = np.full((n_labels, A), np.nan, dtype=float)
    corr_fit_arr = np.full((n_labels, A), np.nan, dtype=float)

    if add_intercept:
        alphas_arr = np.full((n_labels, A), np.nan, dtype=float)
        stderrs_alpha_arr = np.full((n_labels, A), np.nan, dtype=float)
        tstats_alpha_arr = np.full((n_labels, A), np.nan, dtype=float)
    else:
        alphas_arr = stderrs_alpha_arr = tstats_alpha_arr = None

    if n_pred:
        betas_arr = np.full((n_labels, A, n_pred), np.nan, dtype=float)
        stderrs_beta_arr = np.full((n_labels, A, n_pred), np.nan, dtype=float)
        tstats_beta_arr = np.full((n_labels, A, n_pred), np.nan, dtype=float)
        corr_pred_arr = np.full((n_labels, A, n_pred), np.nan, dtype=float)
    else:
        betas_arr = stderrs_beta_arr = tstats_beta_arr = corr_pred_arr = None

    y_arr = y.to_numpy(dtype=float)
    if K:
        X_pred = np.stack([Xi.to_numpy(dtype=float) for Xi in X_list], axis=2)
    else:
        X_pred = np.empty((T, A, 0), dtype=float)

    finite_y_all = np.isfinite(y_arr)
    if K:
        finite_X_all = np.all(np.isfinite(X_pred), axis=2)
    else:
        finite_X_all = np.ones((T, A), dtype=bool)
    finite_joint_all = finite_y_all & finite_X_all

    is_half = isinstance(rolling_window, str) and rolling_window == "per_halves"
    start_col = 1 if add_intercept else 0
    min_obs_required = max(min_required, len(penalize_mask) + 1)

    positions_used = set()
    window_count = 0
    for label, rows in _window_iter(y.index, rolling_window, stride_window=stride_window):
        win_start = perf_counter()
        window_count += 1
        row_idx = np.flatnonzero(rows)
        row_count = int(row_idx.size)
        if row_count == 0:
            _profile_log(
                "_ols_per_asset window: "
                f"label={label}, rows={row_count}, processed_assets=0, obs=0, "
                f"status=no_rows, elapsed={perf_counter() - win_start:.4f}s"
            )
            continue

        positions = label_to_pos.get(label, [])
        window_assets = 0
        window_obs = 0.0

        for asset_idx in range(A):
            mask_valid = finite_joint_all[row_idx, asset_idx]
            obs_valid = int(mask_valid.sum())
            if obs_valid < min_obs_required:
                continue

            y_asset = y_arr[row_idx[mask_valid], asset_idx]
            if K:
                X_asset_raw = X_pred[row_idx[mask_valid], asset_idx, :]
            else:
                X_asset_raw = np.empty((obs_valid, 0), dtype=float)
            X_asset = _add_intercept(X_asset_raw, add_intercept)

            beta, resid_tr, sse = _solve_ridge(X_asset, y_asset, l2, penalize_mask)
            yhat = X_asset @ beta

            y_centered = y_asset - np.mean(y_asset)
            tss = float(np.dot(y_centered, y_centered))
            r2_val = 1.0 - (sse / tss) if tss > 0.0 else np.nan

            B = _ridge_bread(X_asset, l2, penalize_mask)
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                B_inv = np.linalg.pinv(B)
            XtX = X_asset.T @ X_asset
            df_model = float(np.trace(XtX @ B_inv))
            df_resid = max(1.0, obs_valid - df_model)
            adj_r2_val = (
                1.0 - (sse / df_resid) / (tss / max(1.0, obs_valid - 1.0)) if tss > 0.0 else np.nan
            )

            se, t = _ols_cov_se_t(X_asset, resid_tr, beta, l2, penalize_mask, se_mode, nw_lags)

            if obs_valid >= 2:
                yhat_centered = yhat - np.mean(yhat)
                var_yhat = float(np.dot(yhat_centered, yhat_centered))
                if tss > 0.0 and var_yhat > 0.0:
                    corr_fit_val = float(np.dot(y_centered, yhat_centered) / math.sqrt(tss * var_yhat))
                else:
                    corr_fit_val = np.nan
                corr_pred_vals = np.full(n_pred, np.nan, dtype=float)
                if n_pred and tss > 0.0:
                    X_pred_mat = X_asset[:, 1:] if add_intercept else X_asset
                    for j in range(n_pred):
                        x_col = X_pred_mat[:, j]
                        x_centered = x_col - np.mean(x_col)
                        var_x = float(np.dot(x_centered, x_centered))
                        if var_x > 0.0:
                            corr_pred_vals[j] = float(np.dot(y_centered, x_centered) / math.sqrt(tss * var_x))
            else:
                corr_fit_val = np.nan
                corr_pred_vals = np.full(n_pred, np.nan, dtype=float)

            if positions:
                positions_used.update(positions)
                for pos in positions:
                    nobs_arr[pos, asset_idx] = float(obs_valid)
                    r2_arr[pos, asset_idx] = r2_val
                    adjr2_arr[pos, asset_idx] = adj_r2_val
                    df_model_arr[pos, asset_idx] = df_model
                    df_resid_arr[pos, asset_idx] = df_resid
                    corr_fit_arr[pos, asset_idx] = corr_fit_val
                    if add_intercept and alphas_arr is not None:
                        alphas_arr[pos, asset_idx] = beta[0]
                        stderrs_alpha_arr[pos, asset_idx] = se[0]
                        tstats_alpha_arr[pos, asset_idx] = t[0]
                    if n_pred and betas_arr is not None:
                        betas_arr[pos, asset_idx, :] = beta[start_col:].copy()
                        stderrs_beta_arr[pos, asset_idx, :] = se[start_col:].copy()
                        tstats_beta_arr[pos, asset_idx, :] = t[start_col:].copy()
                        corr_pred_arr[pos, asset_idx, :] = corr_pred_vals.copy()

            idx_valid_rows = row_idx[mask_valid]
            if rolling_window is None or is_half:
                fitted_arr[idx_valid_rows, asset_idx] = yhat
                resid_arr[idx_valid_rows, asset_idx] = resid_tr
            else:
                t_last = row_idx[-1]
                if finite_joint_all[t_last, asset_idx]:
                    if K:
                        X_last_raw = X_pred[t_last, asset_idx, :][None, :]
                    else:
                        X_last_raw = np.empty((1, 0), dtype=float)
                    X_last = _add_intercept(X_last_raw, add_intercept)
                    yhat_last = float((X_last @ beta).reshape(-1)[0])
                    fitted_arr[t_last, asset_idx] = yhat_last
                    y_val_last = y_arr[t_last, asset_idx]
                    if np.isfinite(y_val_last):
                        resid_arr[t_last, asset_idx] = y_val_last - yhat_last

            window_obs += float(obs_valid)
            window_assets += 1

        _profile_log(
            "_ols_per_asset window: "
            f"label={label}, rows={row_count}, processed_assets={window_assets}, "
            f"obs={window_obs}, elapsed={perf_counter() - win_start:.4f}s"
        )

    _profile_log(
        "_ols_per_asset: "
        f"completed windows={window_count}, total_elapsed={perf_counter() - t_total:.4f}s"
    )

    if n_labels:
        if positions_used:
            keep_positions = sorted(positions_used)
            idx_out = idx_out.take(keep_positions)
            r2_arr = r2_arr[keep_positions, :]
            adjr2_arr = adjr2_arr[keep_positions, :]
            nobs_arr = nobs_arr[keep_positions, :]
            df_model_arr = df_model_arr[keep_positions, :]
            df_resid_arr = df_resid_arr[keep_positions, :]
            corr_fit_arr = corr_fit_arr[keep_positions, :]
            if alphas_arr is not None:
                alphas_arr = alphas_arr[keep_positions, :]
                stderrs_alpha_arr = stderrs_alpha_arr[keep_positions, :]
                tstats_alpha_arr = tstats_alpha_arr[keep_positions, :]
            if betas_arr is not None:
                betas_arr = betas_arr[keep_positions, :, :]
                stderrs_beta_arr = stderrs_beta_arr[keep_positions, :, :]
                tstats_beta_arr = tstats_beta_arr[keep_positions, :, :]
                corr_pred_arr = corr_pred_arr[keep_positions, :, :]
            n_labels = len(idx_out)
        else:
            idx_out = idx_out[:0]
            r2_arr = r2_arr[:0]
            adjr2_arr = adjr2_arr[:0]
            nobs_arr = nobs_arr[:0]
            df_model_arr = df_model_arr[:0]
            df_resid_arr = df_resid_arr[:0]
            corr_fit_arr = corr_fit_arr[:0]
            if alphas_arr is not None:
                alphas_arr = alphas_arr[:0]
                stderrs_alpha_arr = stderrs_alpha_arr[:0]
                tstats_alpha_arr = tstats_alpha_arr[:0]
            if betas_arr is not None:
                betas_arr = betas_arr[:0]
                stderrs_beta_arr = stderrs_beta_arr[:0]
                tstats_beta_arr = tstats_beta_arr[:0]
                corr_pred_arr = corr_pred_arr[:0]
            n_labels = 0

    fitted = pd.DataFrame(fitted_arr, index=y.index, columns=y.columns, dtype=float)
    resid = pd.DataFrame(resid_arr, index=y.index, columns=y.columns, dtype=float)

    r2 = pd.DataFrame(r2_arr, index=idx_out, columns=y.columns, dtype=float)
    adjr2 = pd.DataFrame(adjr2_arr, index=idx_out, columns=y.columns, dtype=float)
    nobs = pd.DataFrame(nobs_arr, index=idx_out, columns=y.columns, dtype=float)
    df_model_df = pd.DataFrame(df_model_arr, index=idx_out, columns=y.columns, dtype=float)
    df_resid_df = pd.DataFrame(df_resid_arr, index=idx_out, columns=y.columns, dtype=float)
    corr_fit_df = pd.DataFrame(corr_fit_arr, index=idx_out, columns=y.columns, dtype=float)

    if add_intercept and alphas_arr is not None:
        alphas_out = pd.DataFrame(alphas_arr, index=idx_out, columns=y.columns, dtype=float)
        stderrs_a = pd.DataFrame(stderrs_alpha_arr, index=idx_out, columns=y.columns, dtype=float)
        tstats_a = pd.DataFrame(tstats_alpha_arr, index=idx_out, columns=y.columns, dtype=float)
    else:
        alphas_out = pd.DataFrame(index=idx_out, dtype=float)
        stderrs_a = pd.DataFrame(index=idx_out, dtype=float)
        tstats_a = pd.DataFrame(index=idx_out, dtype=float)

    if n_pred and betas_arr is not None:
        reshaped = betas_arr.reshape(n_labels, -1)
        betas_out = pd.DataFrame(reshaped, index=idx_out, columns=beta_cols, dtype=float)
        stderrs_b = pd.DataFrame(
            stderrs_beta_arr.reshape(n_labels, -1), index=idx_out, columns=beta_cols, dtype=float
        )
        tstats_b = pd.DataFrame(
            tstats_beta_arr.reshape(n_labels, -1), index=idx_out, columns=beta_cols, dtype=float
        )
        corr_pred_df = pd.DataFrame(
            corr_pred_arr.reshape(n_labels, -1), index=idx_out, columns=beta_cols, dtype=float
        )
    else:
        betas_out = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
        stderrs_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
        tstats_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
        corr_pred_df = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)

    info = {
        "r2": r2,
        "adj_r2": adjr2,
        "n_obs": nobs,
        "df_model": df_model_df,
        "df_resid": df_resid_df,
        "tstats_beta": tstats_b,
        "stderr_beta": stderrs_b,
        "tstats_alpha": tstats_a,
        "stderr_alpha": stderrs_a,
        "corr": corr_pred_df,
        "corr_fitted": corr_fit_df,
    }
    return RegressionOutput(betas=betas_out, alphas=alphas_out, residuals=resid, fitted=fitted, info=info)



def _ols_global(
    y: pd.DataFrame,
    X_list: List[pd.DataFrame],
    xnames: List[str],
    *,
    rolling_window: Optional[Union[int, str]],
    stride_window: Optional[int],
    add_intercept: bool,
    l2: float,
    se_mode: str,
    nw_lags: Optional[int],
    min_required: int,
    penalize_mask: np.ndarray,
) -> RegressionOutput:
    T, A = y.shape
    K = len(X_list)
    t_total = perf_counter()
    _profile_log(
        "_ols_global: "
        f"start T={T}, A={A}, K={K}, rolling_window={rolling_window}, stride_window={stride_window}"
    )

    if (
        isinstance(rolling_window, int)
        and K == 1
        and add_intercept
        and l2 == 0.0
        and se_mode == "sample"
        and nw_lags is None
    ):
        return _ols_global_single_factor_fast(
            y,
            X_list,
            xnames,
            rolling_window=int(rolling_window),
            stride_window=stride_window,
            min_required=min_required,
        )

    beta_cols = pd.Index(xnames, name="predictor")
    n_pred = len(xnames)

    fitted_arr = np.full((T, A), np.nan, dtype=float)
    resid_arr = np.full((T, A), np.nan, dtype=float)

    labels_out: List[Union[pd.Timestamp, str]] = []
    r2_out: List[float] = []
    adjr2_out: List[float] = []
    nobs_out: List[float] = []
    df_model_out: List[float] = []
    df_resid_out: List[float] = []
    corr_fit_out: List[float] = []

    alpha_out: List[float] = []
    stderr_alpha_out: List[float] = []
    tstat_alpha_out: List[float] = []

    betas_out_rows: List[np.ndarray] = []
    stderr_beta_rows: List[np.ndarray] = []
    tstat_beta_rows: List[np.ndarray] = []
    corr_pred_rows: List[np.ndarray] = []

    y_arr = y.to_numpy(dtype=float)
    if K:
        X_pred = np.stack([Xi.to_numpy(dtype=float) for Xi in X_list], axis=2)
    else:
        X_pred = np.empty((T, A, 0), dtype=float)

    finite_y_all = np.isfinite(y_arr)
    if K:
        finite_X_all = np.all(np.isfinite(X_pred), axis=2)
    else:
        finite_X_all = np.ones((T, A), dtype=bool)
    finite_joint_all = finite_y_all & finite_X_all

    window_count = 0
    windows_computed = 0
    skipped_empty = 0
    skipped_insufficient = 0

    for label, rows in _window_iter(y.index, rolling_window, stride_window=stride_window):
        win_start = perf_counter()
        window_count += 1
        row_idx = np.flatnonzero(rows)
        row_count = int(row_idx.size)
        if row_count == 0:
            skipped_empty += 1
            feature_count = len(penalize_mask)
            _profile_log(
                "_ols_global window: "
                f"label={label}, rows={row_count}, obs=0, features={feature_count}, "
                f"status=no_rows, elapsed={perf_counter() - win_start:.4f}s"
            )
            continue

        y_segments: List[np.ndarray] = []
        X_segments: List[np.ndarray] = []
        t_segments: List[np.ndarray] = []
        a_segments: List[np.ndarray] = []

        for asset_idx in range(A):
            mask_valid = finite_joint_all[row_idx, asset_idx]
            if not np.any(mask_valid):
                continue
            obs_valid = int(mask_valid.sum())
            if obs_valid < min_required:
                continue
            y_vals_asset = y_arr[row_idx[mask_valid], asset_idx]
            if K:
                X_vals_asset = X_pred[row_idx[mask_valid], asset_idx, :]
            else:
                X_vals_asset = np.empty((obs_valid, 0), dtype=float)
            y_segments.append(y_vals_asset)
            X_segments.append(X_vals_asset)
            t_segments.append(row_idx[mask_valid])
            a_segments.append(np.full(obs_valid, asset_idx, dtype=int))

        if not y_segments:
            skipped_empty += 1
            feature_count = len(penalize_mask)
            _profile_log(
                "_ols_global window: "
                f"label={label}, rows={row_count}, obs=0, features={feature_count}, "
                f"status=empty, elapsed={perf_counter() - win_start:.4f}s"
            )
            continue

        y_vec = np.concatenate(y_segments)
        if X_segments:
            X_mat_raw = np.vstack(X_segments)
        else:
            X_mat_raw = np.empty((0, K), dtype=float)
        t_map = np.concatenate(t_segments)
        a_map = np.concatenate(a_segments)

        X_mat = _add_intercept(X_mat_raw, add_intercept)
        feature_count = X_mat.shape[1]
        obs_count = y_vec.shape[0]
        min_obs_total = max(min_required, feature_count + 1)
        if obs_count < min_obs_total:
            skipped_insufficient += 1
            _profile_log(
                "_ols_global window: "
                f"label={label}, rows={row_count}, obs={obs_count}, features={feature_count}, "
                f"status=insufficient_obs, elapsed={perf_counter() - win_start:.4f}s"
            )
            continue

        beta, resid_tr, sse = _solve_ridge(X_mat, y_vec, l2, penalize_mask)
        yhat = X_mat @ beta

        y_bar = float(np.mean(y_vec))
        y_centered = y_vec - y_bar
        tss = float(np.dot(y_centered, y_centered))
        r2_val = 1.0 - (sse / tss) if tss > 0.0 else np.nan

        B = _ridge_bread(X_mat, l2, penalize_mask)
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            B_inv = np.linalg.pinv(B)
        XtX = X_mat.T @ X_mat
        df_model = float(np.trace(XtX @ B_inv))
        df_resid = max(1.0, obs_count - df_model)
        adj_r2_val = (
            1.0 - (sse / df_resid) / (tss / max(1.0, obs_count - 1.0)) if tss > 0.0 else np.nan
        )

        se, t = _ols_cov_se_t(X_mat, resid_tr, beta, l2, penalize_mask, se_mode, nw_lags)

        if obs_count >= 2:
            yhat_centered = yhat - np.mean(yhat)
            var_yhat = float(np.dot(yhat_centered, yhat_centered))
            if tss > 0.0 and var_yhat > 0.0:
                corr_fit_val = float(np.dot(y_centered, yhat_centered) / math.sqrt(tss * var_yhat))
            else:
                corr_fit_val = np.nan
            corr_pred_vals = np.full(n_pred, np.nan, dtype=float)
            if n_pred and tss > 0.0:
                X_pred_mat = X_mat[:, 1:] if add_intercept else X_mat
                for j in range(n_pred):
                    x_col = X_pred_mat[:, j]
                    x_centered = x_col - np.mean(x_col)
                    var_x = float(np.dot(x_centered, x_centered))
                    if var_x > 0.0:
                        corr_pred_vals[j] = float(np.dot(y_centered, x_centered) / math.sqrt(tss * var_x))
        else:
            corr_fit_val = np.nan
            corr_pred_vals = np.full(n_pred, np.nan, dtype=float)

        labels_out.append(label)
        r2_out.append(r2_val)
        adjr2_out.append(adj_r2_val)
        nobs_out.append(float(obs_count))
        df_model_out.append(df_model)
        df_resid_out.append(df_resid)
        corr_fit_out.append(corr_fit_val)

        if add_intercept:
            alpha_out.append(beta[0])
            stderr_alpha_out.append(se[0])
            tstat_alpha_out.append(t[0])

        if n_pred:
            start_col = 1 if add_intercept else 0
            betas_out_rows.append(beta[start_col:].copy())
            stderr_beta_rows.append(se[start_col:].copy())
            tstat_beta_rows.append(t[start_col:].copy())
            corr_pred_rows.append(corr_pred_vals.copy())

        if rolling_window is None or (isinstance(rolling_window, str) and rolling_window == "per_halves"):
            fitted_arr[t_map, a_map] = yhat
            resid_arr[t_map, a_map] = resid_tr
        else:
            t_last = row_idx[-1]
            candidate_assets = np.where(finite_joint_all[t_last, :])[0]
            if candidate_assets.size:
                assets_in_training = np.unique(a_map)
                valid_assets = np.intersect1d(candidate_assets, assets_in_training, assume_unique=True)
                if valid_assets.size:
                    if K:
                        X_last_raw = X_pred[t_last, valid_assets, :]
                    else:
                        X_last_raw = np.empty((valid_assets.size, 0), dtype=float)
                    X_last = _add_intercept(X_last_raw, add_intercept)
                    yhat_last = X_last @ beta
                    fitted_arr[t_last, valid_assets] = yhat_last
                    y_vals_last = y_arr[t_last, valid_assets]
                    mask_finite_last = np.isfinite(y_vals_last)
                    if np.any(mask_finite_last):
                        resid_arr[t_last, valid_assets[mask_finite_last]] = (
                            y_vals_last[mask_finite_last] - yhat_last[mask_finite_last]
                        )

        windows_computed += 1
        _profile_log(
            "_ols_global window: "
            f"label={label}, rows={row_count}, obs={obs_count}, features={feature_count}, "
            f"status=processed, elapsed={perf_counter() - win_start:.4f}s"
        )

    _profile_log(
        "_ols_global: "
        f"completed windows={window_count}, total_elapsed={perf_counter() - t_total:.4f}s"
    )

    is_half = isinstance(rolling_window, str) and rolling_window == "per_halves"
    if labels_out:
        idx_out = pd.Index(labels_out)
    else:
        idx_out = (pd.Index([], dtype=object) if is_half else y.index[:0])
    idx_out = idx_out.rename("half" if is_half else y.index.name)

    fitted = pd.DataFrame(fitted_arr, index=y.index, columns=y.columns, dtype=float)
    resid = pd.DataFrame(resid_arr, index=y.index, columns=y.columns, dtype=float)

    def _build_series(data: List[float]) -> pd.DataFrame:
        return pd.DataFrame(data, index=idx_out, columns=["global"], dtype=float)

    r2 = _build_series(r2_out)
    adjr2 = _build_series(adjr2_out)
    nobs = _build_series(nobs_out)
    df_model_df = _build_series(df_model_out)
    df_resid_df = _build_series(df_resid_out)
    corr_fit_df = _build_series(corr_fit_out)

    if add_intercept:
        alphas_out = pd.DataFrame(alpha_out, index=idx_out, columns=["alpha"], dtype=float)
        stderrs_a = pd.DataFrame(stderr_alpha_out, index=idx_out, columns=["alpha"], dtype=float)
        tstats_a = pd.DataFrame(tstat_alpha_out, index=idx_out, columns=["alpha"], dtype=float)
    else:
        alphas_out = pd.DataFrame(index=idx_out, dtype=float)
        stderrs_a = pd.DataFrame(index=idx_out, dtype=float)
        tstats_a = pd.DataFrame(index=idx_out, dtype=float)

    if n_pred:
        betas_out = pd.DataFrame(betas_out_rows, index=idx_out, columns=beta_cols, dtype=float)
        stderrs_b = pd.DataFrame(stderr_beta_rows, index=idx_out, columns=beta_cols, dtype=float)
        tstats_b = pd.DataFrame(tstat_beta_rows, index=idx_out, columns=beta_cols, dtype=float)
        corr_df = pd.DataFrame(corr_pred_rows, index=idx_out, columns=beta_cols, dtype=float)
    else:
        betas_out = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
        stderrs_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
        tstats_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
        corr_df = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)

    window_summary = pd.Series(
        {
            "evaluated": window_count,
            "computed": windows_computed,
            "skipped_empty": skipped_empty,
            "skipped_insufficient": skipped_insufficient,
        },
        dtype=float,
        name="rolling_ols",
    )

    info = {
        "r2": r2,
        "adj_r2": adjr2,
        "n_obs": nobs,
        "df_model": df_model_df,
        "df_resid": df_resid_df,
        "tstats_beta": tstats_b,
        "stderr_beta": stderrs_b,
        "tstats_alpha": tstats_a,
        "stderr_alpha": stderrs_a,
        "corr": corr_df,
        "corr_fitted": corr_fit_df,
        "window_summary": window_summary,
    }
    return RegressionOutput(betas=betas_out, alphas=alphas_out, residuals=resid, fitted=fitted, info=info)


def _linear_regression_interface(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    *,
    mode: str,
    rolling_window: Optional[Union[int, str]],
    stride_window: Optional[int],
    add_intercept: bool,
    std_errors: str,
    nw_lags: Optional[int],
    l2: float,
    align: str,
    min_periods: Optional[int],
    min_obs: Optional[int],
    n_jobs: Optional[int],
    regression_label: str,
) -> RegressionOutput:
    t_start = perf_counter()
    _profile_log(
        "_linear_regression_interface: "
        f"start mode={mode}, regression={regression_label}, rolling_window={rolling_window}, "
        f"stride_window={stride_window}, n_jobs={n_jobs}"
    )

    if mode not in {"per_asset", "global"}:
        raise ValueError("mode must be 'per_asset' or 'global'")

    valid_std_errors = {"sample", "newey_west", "ess"}
    if std_errors not in valid_std_errors:
        raise ValueError(f"std_errors must be one of {sorted(valid_std_errors)}")

    y_aligned, X_list, xnames, penalize_mask, min_required, effective_min_periods = _prepare_linear_inputs(
        y,
        predictors,
        align=align,
        add_intercept=add_intercept,
        rolling_window=rolling_window,
        min_periods=min_periods,
        min_obs=min_obs,
    )

    engine = _ols_per_asset if mode == "per_asset" else _ols_global
    out = engine(
        y_aligned,
        X_list,
        xnames,
        rolling_window=rolling_window,
        stride_window=stride_window,
        add_intercept=add_intercept,
        l2=float(l2),
        se_mode=std_errors,
        nw_lags=nw_lags,
        min_required=min_required,
        penalize_mask=penalize_mask,
    )

    meta = {
        "mode": mode,
        "std_errors": std_errors,
        "nw_lags": nw_lags,
        "l2": float(l2),
        "add_intercept": bool(add_intercept),
        "rolling_window": rolling_window,
        "stride_window": stride_window,
        "min_periods": effective_min_periods,
        "min_obs": min_required,
        "n_jobs": n_jobs,
        "regression_type": regression_label,
        "align": align,
    }
    window_summary = out.info.get("window_summary") if isinstance(out.info, dict) else None
    if isinstance(window_summary, pd.Series):
        meta["windows_evaluated"] = float(window_summary.get("computed", np.nan))
        meta["windows_total"] = float(window_summary.get("evaluated", np.nan))
    out.info["meta"] = meta
    _profile_log(
        "_linear_regression_interface: "
        f"complete mode={mode}, regression={regression_label}, elapsed={perf_counter() - t_start:.4f}s"
    )
    return out


def ols(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    *,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    stride_window: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Global (stacked) OLS regression over the full sample.
    """
    return _linear_regression_interface(
        y,
        predictors,
        mode="global",
        rolling_window=None,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=0.0,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="OLS",
    )


def ols_per_asset(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    *,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    stride_window: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Per-asset OLS regression estimated independently for each column in y.
    """
    return _linear_regression_interface(
        y,
        predictors,
        mode="per_asset",
        rolling_window=None,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=0.0,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="OLS",
    )


def rolling_ols(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    window: Union[int, str],
    *,
    stride_window: Optional[int] = None,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Global OLS regression evaluated on rolling or grouped windows.
    """
    return _linear_regression_interface(
        y,
        predictors,
        mode="global",
        rolling_window=window,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=0.0,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="OLS",
    )


def rolling_ols_per_asset(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    window: Union[int, str],
    *,
    stride_window: Optional[int] = None,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Per-asset OLS regression evaluated on rolling or grouped windows.
    """
    return _linear_regression_interface(
        y,
        predictors,
        mode="per_asset",
        rolling_window=window,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=0.0,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="OLS",
    )


def ridge(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    *,
    l2: float = 1.0,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    stride_window: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Global ridge regression (ℓ2-penalised OLS) over the full sample.
    """
    if l2 <= 0.0:
        raise ValueError("l2 must be positive for ridge regression")
    return _linear_regression_interface(
        y,
        predictors,
        mode="global",
        rolling_window=None,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=l2,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="ridge",
    )


def ridge_per_asset(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    *,
    l2: float = 1.0,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    stride_window: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Per-asset ridge regression (ℓ2-penalised OLS) over the full sample.
    """
    if l2 <= 0.0:
        raise ValueError("l2 must be positive for ridge regression")
    return _linear_regression_interface(
        y,
        predictors,
        mode="per_asset",
        rolling_window=None,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=l2,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="ridge",
    )


def rolling_ridge(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    window: Union[int, str],
    *,
    l2: float = 1.0,
    stride_window: Optional[int] = None,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Global ridge regression evaluated on rolling or grouped windows.
    """
    if l2 <= 0.0:
        raise ValueError("l2 must be positive for ridge regression")
    return _linear_regression_interface(
        y,
        predictors,
        mode="global",
        rolling_window=window,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=l2,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="ridge",
    )


def rolling_ridge_per_asset(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    window: Union[int, str],
    *,
    l2: float = 1.0,
    stride_window: Optional[int] = None,
    add_intercept: bool = True,
    std_errors: str = "sample",
    nw_lags: Optional[int] = None,
    align: str = "inner",
    min_periods: Optional[int] = None,
    min_obs: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> RegressionOutput:
    """
    Per-asset ridge regression evaluated on rolling or grouped windows.
    """
    if l2 <= 0.0:
        raise ValueError("l2 must be positive for ridge regression")
    return _linear_regression_interface(
        y,
        predictors,
        mode="per_asset",
        rolling_window=window,
        stride_window=stride_window,
        add_intercept=add_intercept,
        std_errors=std_errors,
        nw_lags=nw_lags,
        l2=l2,
        align=align,
        min_periods=min_periods,
        min_obs=min_obs,
        n_jobs=n_jobs,
        regression_label="ridge",
    )


class _LinearRegressorBase:
    """
    Thin OO wrapper around the functional regression helpers exposed in this module.
    """

    def __init__(
        self,
        *,
        per_asset: bool = False,
        rolling_window: Optional[Union[int, str]] = None,
        stride_window: Optional[int] = None,
        add_intercept: bool = True,
        std_errors: str = "sample",
        nw_lags: Optional[int] = None,
        align: str = "inner",
        min_periods: Optional[int] = None,
        min_obs: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.per_asset = bool(per_asset)
        self.rolling_window = rolling_window
        self.stride_window = stride_window
        self.add_intercept = bool(add_intercept)
        self.std_errors = std_errors
        self.nw_lags = nw_lags
        self.align = align
        self.min_periods = min_periods
        self.min_obs = min_obs
        self.n_jobs = n_jobs

        self._result: Optional[RegressionOutput] = None
        self._predictor_count: int = 0
        self._input_predictor_names: List[str] = []
        self._predictor_names: List[str] = []
        self._train_columns: pd.Index = pd.Index([])
        self._train_index: pd.Index = pd.Index([])

    def fit(self, y: pd.DataFrame, predictors: Union[pd.DataFrame, Iterable[pd.DataFrame], None]) -> RegressionOutput:
        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pandas DataFrame")
        predictor_frames = self._ensure_predictor_list(predictors)
        self._predictor_count = len(predictor_frames)
        self._input_predictor_names = _default_names(predictor_frames) if predictor_frames else []

        result = self._fit_impl(y, predictor_frames)
        self._result = result
        self._train_columns = result.fitted.columns
        self._train_index = result.fitted.index

        inferred = self._extract_predictor_names(result.betas.columns)
        if inferred and len(inferred) == self._predictor_count:
            self._predictor_names = inferred
        else:
            self._predictor_names = list(self._input_predictor_names)
        return result

    def predict(
        self,
        predictors: Union[pd.DataFrame, Iterable[pd.DataFrame], None],
        *,
        label: Optional[Union[int, pd.Timestamp, str]] = None,
        index: Optional[Iterable] = None,
    ) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("call fit() before predict()")

        frames = self._ensure_predictor_list(predictors)
        if len(frames) != self._predictor_count:
            raise ValueError(f"expected {self._predictor_count} predictor(s), received {len(frames)}")

        aligned_frames, out_index = self._prepare_predictors(frames, index=index)
        beta_series = self._select_series(self._result.betas, label)
        alpha_series = self._select_series(self._result.alphas, label) if self.add_intercept else pd.Series(dtype=float)
        return self._assemble_prediction(aligned_frames, out_index, beta_series, alpha_series)

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------

    def _fit_impl(self, y: pd.DataFrame, predictors: List[pd.DataFrame]) -> RegressionOutput:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _ensure_predictor_list(
        predictors: Union[pd.DataFrame, Iterable[pd.DataFrame], None]
    ) -> List[pd.DataFrame]:
        if predictors is None:
            return []
        if isinstance(predictors, pd.DataFrame):
            frames = [predictors]
        else:
            try:
                frames = list(predictors)
            except TypeError as exc:
                raise TypeError("predictors must be a DataFrame, an iterable of DataFrames, or None") from exc
        for frame in frames:
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("each predictor must be a pandas DataFrame")
        return frames

    def _prepare_predictors(
        self,
        frames: List[pd.DataFrame],
        *,
        index: Optional[Iterable],
    ) -> Tuple[List[pd.DataFrame], pd.Index]:
        columns = self._train_columns
        aligned: List[pd.DataFrame] = []
        idx: Optional[pd.Index] = None

        for frame in frames:
            if idx is None:
                idx = frame.index
            elif not frame.index.equals(idx):
                raise ValueError("all predictor DataFrames must share the same index")

            missing = columns.difference(frame.columns)
            if missing.size:
                missing_cols = ", ".join(str(c) for c in missing)
                raise ValueError(f"predictor DataFrame missing required columns: {missing_cols}")
            aligned.append(frame.reindex(columns=columns))

        if not aligned:
            if index is None:
                idx = self._train_index
            else:
                idx = pd.Index(index)
        else:
            if index is not None:
                index_candidate = pd.Index(index)
                if not index_candidate.equals(idx):
                    raise ValueError("supplied index does not match predictor indices")
            idx = idx if idx is not None else self._train_index

        return aligned, pd.Index(idx)

    def _available_labels(self) -> pd.Index:
        if self._result is None:
            return pd.Index([])
        if not self._result.betas.index.empty:
            return self._result.betas.index
        return self._result.alphas.index

    def _select_series(self, df: pd.DataFrame, label: Optional[Union[int, pd.Timestamp, str]]) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        if label is None:
            series = df.iloc[-1]
        elif isinstance(label, (int, np.integer)):
            if 0 <= label < len(df):
                series = df.iloc[label]
            else:
                available = ", ".join(str(v) for v in self._available_labels().unique())
                raise ValueError(f"label {label!r} not found in fitted coefficients; available: {available}")
        else:
            if label not in df.index:
                available = ", ".join(str(v) for v in self._available_labels().unique())
                raise ValueError(f"label {label!r} not found in fitted coefficients; available: {available}")
            series = df.loc[label]

        if isinstance(series, pd.DataFrame):
            series = series.iloc[-1]
        return series

    @staticmethod
    def _extract_predictor_names(columns: Union[pd.Index, pd.MultiIndex]) -> List[Any]:
        if isinstance(columns, pd.MultiIndex):
            names: List[Any] = []
            if columns.names and "predictor" in columns.names:
                predictor_level = columns.names.index("predictor")
            else:
                predictor_level = columns.nlevels - 1
            for col in columns:
                candidate = col[predictor_level]
                if candidate not in names:
                    names.append(candidate)
            return names
        return list(columns)

    def _assemble_prediction(
        self,
        predictors: List[pd.DataFrame],
        index: pd.Index,
        beta_series: pd.Series,
        alpha_series: pd.Series,
    ) -> pd.DataFrame:
        if self.per_asset:
            return self._predict_per_asset(predictors, index, beta_series, alpha_series)
        return self._predict_global(predictors, index, beta_series, alpha_series)

    def _predict_global(
        self,
        predictors: List[pd.DataFrame],
        index: pd.Index,
        beta_series: pd.Series,
        alpha_series: pd.Series,
    ) -> pd.DataFrame:
        columns = self._train_columns
        fitted = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)

        if predictors:
            beta_vector = beta_series.astype(float) if not beta_series.empty else pd.Series(dtype=float)
            for frame, name in zip(predictors, self._predictor_names):
                coeff = beta_vector.get(name, np.nan)
                fitted = fitted.add(frame * coeff)

        if self.add_intercept and not alpha_series.empty:
            intercept = alpha_series.iloc[0] if alpha_series.size == 1 else alpha_series.reindex(columns, copy=False)
            if np.isscalar(intercept):
                fitted = fitted + intercept
            else:
                fitted = fitted.add(intercept, axis=1)
        return fitted

    def _predict_per_asset(
        self,
        predictors: List[pd.DataFrame],
        index: pd.Index,
        beta_series: pd.Series,
        alpha_series: pd.Series,
    ) -> pd.DataFrame:
        columns = self._train_columns
        fitted = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)

        if predictors and not beta_series.empty:
            if isinstance(beta_series.index, pd.MultiIndex):
                betas_matrix = beta_series.unstack(level=-1)
            else:
                betas_matrix = pd.DataFrame(index=columns, columns=self._predictor_names, dtype=float)
            betas_matrix = betas_matrix.reindex(index=columns)
            betas_matrix = betas_matrix.reindex(columns=self._predictor_names, copy=False)

            for frame, name in zip(predictors, self._predictor_names):
                coeffs = betas_matrix[name] if name in betas_matrix else pd.Series(dtype=float)
                fitted = fitted.add(frame.mul(coeffs, axis=1), fill_value=0.0)

        if self.add_intercept and not alpha_series.empty:
            intercept = alpha_series.reindex(columns, copy=False)
            fitted = fitted.add(intercept, axis=1)
        return fitted


class OLS(_LinearRegressorBase):
    """
    Object-oriented wrapper for the OLS helpers in this module.
    """

    def _fit_impl(self, y: pd.DataFrame, predictors: List[pd.DataFrame]) -> RegressionOutput:
        kwargs = dict(
            add_intercept=self.add_intercept,
            std_errors=self.std_errors,
            nw_lags=self.nw_lags,
            align=self.align,
            min_periods=self.min_periods,
            min_obs=self.min_obs,
            stride_window=self.stride_window,
            n_jobs=self.n_jobs,
        )
        if self.rolling_window is None:
            func = ols_per_asset if self.per_asset else ols
            return func(y, predictors, **kwargs)
        func = rolling_ols_per_asset if self.per_asset else rolling_ols
        return func(y, predictors, window=self.rolling_window, **kwargs)


class Ridge(_LinearRegressorBase):
    """
    Object-oriented wrapper for ridge regression helpers in this module.
    """

    def __init__(self, *, l2: float = 1.0, **kwargs: object) -> None:
        if float(l2) <= 0.0:
            raise ValueError("l2 must be positive for ridge regression")
        super().__init__(**kwargs)
        self.l2 = float(l2)

    def _fit_impl(self, y: pd.DataFrame, predictors: List[pd.DataFrame]) -> RegressionOutput:
        kwargs = dict(
            l2=self.l2,
            add_intercept=self.add_intercept,
            std_errors=self.std_errors,
            nw_lags=self.nw_lags,
            align=self.align,
            min_periods=self.min_periods,
            min_obs=self.min_obs,
            stride_window=self.stride_window,
            n_jobs=self.n_jobs,
        )
        if self.rolling_window is None:
            func = ridge_per_asset if self.per_asset else ridge
            return func(y, predictors, **kwargs)
        func = rolling_ridge_per_asset if self.per_asset else rolling_ridge
        return func(y, predictors, window=self.rolling_window, **kwargs)


def _qr_per_asset(
    y: pd.DataFrame,
    X_list: List[pd.DataFrame],
    xnames: List[str],
    *,
    rolling_window: Optional[Union[int, str]],
    stride_window: Optional[int],
    add_intercept: bool,
    quantiles: List[float],
    primary_q: float,
    min_required: int,
) -> RegressionOutput:
    """
    Per-asset quantile regression (Koenker–Bassett) using statsmodels.
    Populates standard errors and t-stats for parameters.
    """
    T, A = y.shape
    K = len(X_list)

    # Output axes (MultiIndex time x quantile)
    base_idx = (
        pd.Index([y.index[-1]], name=y.index.name)
        if rolling_window is None
        else (pd.Index(y.index) if isinstance(rolling_window, int) else pd.Index([], dtype=object))
    )
    if isinstance(rolling_window, str) and rolling_window == "per_halves":
        idx_dt = pd.DatetimeIndex(y.index)
        halves = pd.unique([f"{d.year}H{1 if d.month <= 6 else 2}" for d in idx_dt])
        base_idx = pd.Index(halves, name="half")
    idx_out = pd.MultiIndex.from_product([base_idx, quantiles], names=[base_idx.name or "time", "quantile"])

    beta_cols = pd.MultiIndex.from_product([y.columns, xnames], names=["asset", "predictor"])
    betas_out = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    alphas_out = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float) if add_intercept else pd.DataFrame(
        index=idx_out, dtype=float
    )

    fitted = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    resid = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)

    r2 = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    adjr2 = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)  # stays NaN
    nobs = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)
    df_model_df = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)  # NaN
    df_resid_df = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)  # NaN
    tstats_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    stderrs_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    tstats_a = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float) if add_intercept else pd.DataFrame(
        index=idx_out, dtype=float
    )
    stderrs_a = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float) if add_intercept else pd.DataFrame(
        index=idx_out, dtype=float
    )
    corr_df = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    corr_fit_df = pd.DataFrame(index=idx_out, columns=y.columns, dtype=float)

    y_arr, X_pred = _precompute_arrays(y, X_list)

    for label, rows in _window_iter(y.index, rolling_window, stride_window=stride_window):
        for a_idx, a in enumerate(y.columns):
            y_win = y_arr[rows, a_idx]
            X_win = X_pred[rows, a_idx, :]  # (#rows, K)
            X_mat = _add_intercept(X_win, add_intercept)

            m = _finite_mask_yX(y_win, X_mat, skip_col0=add_intercept)
            if m.sum() < max(min_required, X_mat.shape[1] + 1):
                continue

            y_tr = y_win[m]
            X_tr = X_mat[m, :]

            for q in quantiles:
                beta, se_all, t_all = _quantile_fit(y_tr, X_tr, q)

                # Write params
                for j, name in enumerate(xnames):
                    betas_out.loc[(label, q), (a, name)] = beta[j + 1] if add_intercept else beta[j]
                    stderrs_b.loc[(label, q), (a, name)] = se_all[j + 1] if add_intercept else se_all[j]
                    tstats_b.loc[(label, q), (a, name)] = t_all[j + 1] if add_intercept else t_all[j]
                if add_intercept:
                    alphas_out.loc[(label, q), a] = beta[0]
                    stderrs_a.loc[(label, q), a] = se_all[0]
                    tstats_a.loc[(label, q), a] = t_all[0]

                # Metrics
                # Use pseudo-R^2 on the TRAIN rows only
                yhat_tr = X_tr @ beta
                r2_val = _quantile_pseudo_r2(y_tr, yhat_tr, q)
                r2.loc[(label, q), a] = r2_val
                nobs.loc[(label, q), a] = int(m.sum())
                if m.sum() >= 2:
                    y_centered = y_tr - np.mean(y_tr)
                    var_y = float(np.dot(y_centered, y_centered))
                    yhat_centered = yhat_tr - np.mean(yhat_tr)
                    var_yhat = float(np.dot(yhat_centered, yhat_centered))
                    if var_y > 0.0 and var_yhat > 0.0:
                        cov_y_yhat = float(np.dot(y_centered, yhat_centered))
                        corr_fit_val = cov_y_yhat / math.sqrt(var_y * var_yhat)
                    else:
                        corr_fit_val = np.nan
                    X_pred_tr = X_tr[:, 1:] if add_intercept else X_tr
                    corr_pred_vals = np.full(len(xnames), np.nan, dtype=float)
                    if X_pred_tr.size and var_y > 0.0:
                        X_centered = X_pred_tr - np.mean(X_pred_tr, axis=0)
                        var_x = np.sum(X_centered ** 2, axis=0)
                        std_y = math.sqrt(var_y)
                        std_x = np.sqrt(var_x, out=np.zeros_like(var_x), where=var_x > 0.0)
                        denom = std_y * std_x
                        cov_yx = y_centered @ X_centered
                        valid = denom > 0.0
                        if np.any(valid):
                            corr_pred_vals[valid] = cov_yx[valid] / denom[valid]
                else:
                    corr_fit_val = np.nan
                    corr_pred_vals = np.full(len(xnames), np.nan, dtype=float)
                corr_fit_df.loc[(label, q), a] = corr_fit_val
                for j, name in enumerate(xnames):
                    corr_df.loc[(label, q), (a, name)] = corr_pred_vals[j]

                # Fitted/residuals
                if (rolling_window is None) or (isinstance(rolling_window, str) and rolling_window == "per_halves"):
                    # Fill window
                    yhat_win = X_mat @ beta
                    if q == primary_q:
                        fitted.loc[pd.Index(y.index)[rows], a] = yhat_win
                        yy = y_win
                        r = np.where(np.isfinite(yy), yy - yhat_win, np.nan)
                        resid.loc[pd.Index(y.index)[rows], a] = r
                else:
                    # Rolling int: only at the label row
                    t_idx = np.where(rows)[0][-1]
                    x_row = X_mat[-1, :]
                    if np.all(np.isfinite(x_row[1:] if add_intercept else x_row)):
                        yhat_t = float(x_row @ beta)
                        if q == primary_q:
                            fitted.iat[t_idx, a_idx] = yhat_t
                            y_val = y_arr[t_idx, a_idx]
                            if np.isfinite(y_val):
                                resid.iat[t_idx, a_idx] = y_val - yhat_t

    info = {
        "r2": r2,
        "adj_r2": adjr2,
        "n_obs": nobs,
        "df_model": df_model_df,
        "df_resid": df_resid_df,
        "tstats_beta": tstats_b,
        "stderr_beta": stderrs_b,
        "tstats_alpha": tstats_a,
        "stderr_alpha": stderrs_a,
        "meta": {"primary_quantile": primary_q},
        "corr": corr_df,
        "corr_fitted": corr_fit_df,
    }
    return RegressionOutput(betas=betas_out, alphas=alphas_out, residuals=resid, fitted=fitted, info=info)


def _qr_global(
    y: pd.DataFrame,
    X_list: List[pd.DataFrame],
    xnames: List[str],
    *,
    rolling_window: Optional[Union[int, str]],
    stride_window: Optional[int],
    add_intercept: bool,
    quantiles: List[float],
    primary_q: float,
    min_required: int,
) -> RegressionOutput:
    """
    Global (stacked) quantile regression using statsmodels.
    Populates standard errors and t-stats for parameters.
    """
    T, A = y.shape
    K = len(X_list)

    base_idx = (
        pd.Index([y.index[-1]], name=y.index.name)
        if rolling_window is None
        else (pd.Index(y.index) if isinstance(rolling_window, int) else pd.Index([], dtype=object))
    )
    if isinstance(rolling_window, str) and rolling_window == "per_halves":
        idx_dt = pd.DatetimeIndex(y.index)
        halves = pd.unique([f"{d.year}H{1 if d.month <= 6 else 2}" for d in idx_dt])
        base_idx = pd.Index(halves, name="half")
    idx_out = pd.MultiIndex.from_product([base_idx, quantiles], names=[base_idx.name or "time", "quantile"])

    beta_cols = pd.Index(xnames, name="predictor")
    betas_out = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    alphas_out = pd.DataFrame(index=idx_out, columns=["alpha"], dtype=float) if add_intercept else pd.DataFrame(
        index=idx_out, dtype=float
    )

    fitted = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    resid = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)

    r2 = pd.DataFrame(index=idx_out, columns=["global"], dtype=float)
    adjr2 = pd.DataFrame(index=idx_out, columns=["global"], dtype=float)
    nobs = pd.DataFrame(index=idx_out, columns=["global"], dtype=float)
    df_model_df = pd.DataFrame(index=idx_out, columns=["global"], dtype=float)
    df_resid_df = pd.DataFrame(index=idx_out, columns=["global"], dtype=float)
    tstats_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    stderrs_b = pd.DataFrame(index=idx_out, columns=beta_cols, dtype=float)
    tstats_a = pd.DataFrame(index=idx_out, columns=["alpha"], dtype=float) if add_intercept else pd.DataFrame(
        index=idx_out, dtype=float
    )
    stderrs_a = pd.DataFrame(index=idx_out, columns=["alpha"], dtype=float) if add_intercept else pd.DataFrame(
        index=idx_out, dtype=float
    )

    y_arr, X_pred = _precompute_arrays(y, X_list)
    finite_y_all = np.isfinite(y_arr)
    finite_X_all = np.all(np.isfinite(X_pred), axis=2)
    finite_joint_all = finite_y_all & finite_X_all
    X_full_all = _add_intercept(X_pred, add_intercept)

    for label, rows in _window_iter(y.index, rolling_window, stride_window=stride_window):
        row_idx = np.flatnonzero(rows)
        y_vec, X_mat, t_map, a_map = _stack_global_window(
            y_arr,
            X_pred,
            rows,
            add_intercept,
            min_required,
            finite_mask_all=finite_joint_all,
            X_full_all=X_full_all,
        )
        if y_vec.shape[0] < max(min_required, X_mat.shape[1] + 1):
            continue

        assets_in_training = np.unique(a_map) if a_map.size else np.empty((0,), dtype=int)
        for q in quantiles:
            beta, se_all, t_all = _quantile_fit(y_vec, X_mat, q)
            for j, name in enumerate(xnames):
                betas_out.loc[(label, q), name] = beta[j + 1] if add_intercept else beta[j]
                stderrs_b.loc[(label, q), name] = se_all[j + 1] if add_intercept else se_all[j]
                tstats_b.loc[(label, q), name] = t_all[j + 1] if add_intercept else t_all[j]
            if add_intercept:
                alphas_out.loc[(label, q), "alpha"] = beta[0]
                stderrs_a.loc[(label, q), "alpha"] = se_all[0]
                tstats_a.loc[(label, q), "alpha"] = t_all[0]

            # Pseudo-R^2 on train sample
            yhat = X_mat @ beta
            r2_val = _quantile_pseudo_r2(y_vec, yhat, q)
            r2.loc[(label, q), "global"] = r2_val
            nobs.loc[(label, q), "global"] = int(X_mat.shape[0])
            if X_mat.shape[0] >= 2:
                y_centered = y_vec - np.mean(y_vec)
                var_y = float(np.dot(y_centered, y_centered))
                yhat_centered = yhat - np.mean(yhat)
                var_yhat = float(np.dot(yhat_centered, yhat_centered))
                if var_y > 0.0 and var_yhat > 0.0:
                    cov_y_yhat = float(np.dot(y_centered, yhat_centered))
                    corr_fit_val = cov_y_yhat / math.sqrt(var_y * var_yhat)
                else:
                    corr_fit_val = np.nan
                X_pred_mat = X_mat[:, 1:] if add_intercept else X_mat
                corr_pred_vals = np.full(len(xnames), np.nan, dtype=float)
                if X_pred_mat.size and var_y > 0.0:
                    X_centered = X_pred_mat - np.mean(X_pred_mat, axis=0)
                    var_x = np.sum(X_centered ** 2, axis=0)
                    std_y = math.sqrt(var_y)
                    std_x = np.sqrt(var_x, out=np.zeros_like(var_x), where=var_x > 0.0)
                    denom = std_y * std_x
                    cov_yx = y_centered @ X_centered
                    valid = denom > 0.0
                    if np.any(valid):
                        corr_pred_vals[valid] = cov_yx[valid] / denom[valid]
            else:
                corr_fit_val = np.nan
                corr_pred_vals = np.full(len(xnames), np.nan, dtype=float)
            corr_fit_df.loc[(label, q), "global"] = corr_fit_val
            for j, name in enumerate(xnames):
                corr_df.loc[(label, q), name] = corr_pred_vals[j]

            # Fitted/resid
            if rolling_window is None or (isinstance(rolling_window, str) and rolling_window == "per_halves"):
                if q == primary_q:
                    for i in range(yhat.shape[0]):
                        t_i = t_map[i]; a_i = a_map[i]
                        fitted.iat[t_i, a_i] = yhat[i]
                        y_val = y_arr[t_i, a_i]
                        if np.isfinite(y_val):
                            resid.iat[t_i, a_i] = y_val - yhat[i]
            else:
                # Rolling int: only label row
                if q == primary_q:
                    if row_idx.size == 0:
                        continue
                    t_last = row_idx[-1]
                    m_last = finite_X_all[t_last]
                    if np.any(m_last) and assets_in_training.size:
                        candidate_assets = np.where(m_last)[0]
                        valid_last_assets = (
                            np.intersect1d(candidate_assets, assets_in_training, assume_unique=True)
                            if candidate_assets.size
                            else candidate_assets
                        )
                        if valid_last_assets.size:
                            X_last_full = X_full_all[t_last, valid_last_assets, :]
                            yhat_last = X_last_full @ beta
                            for j, a_i in enumerate(valid_last_assets):
                                fitted.iat[t_last, a_i] = yhat_last[j]
                                y_val = y_arr[t_last, a_i]
                                if np.isfinite(y_val):
                                    resid.iat[t_last, a_i] = y_val - yhat_last[j]

    info = {
        "r2": r2,
        "adj_r2": adjr2,
        "n_obs": nobs,
        "df_model": df_model_df,
        "df_resid": df_resid_df,
        "tstats_beta": tstats_b,
        "stderr_beta": stderrs_b,
        "tstats_alpha": tstats_a,
        "stderr_alpha": stderrs_a,
        "corr": corr_df,
        "corr_fitted": corr_fit_df,
        "meta": {"primary_quantile": primary_q},
    }
    return RegressionOutput(betas=betas_out, alphas=alphas_out, residuals=resid, fitted=fitted, info=info)


# ======================================================================================
# Public API
# ======================================================================================

def time_series_regression(
    y: pd.DataFrame,
    predictors: List[pd.DataFrame],
    rolling_window: Optional[Union[int, str]] = None,
    stride_window: Optional[int] = None,
    min_periods: Optional[int] = None,
    mode: str = "per_asset",             # {'per_asset', 'global'}
    std_errors: str = "sample",          # {'sample', 'newey_west', 'ess'} (OLS only)
    nw_lags: Optional[int] = None,       # Newey-West lags (OLS only)
    l2: float = 0.0,                     # Ridge penalty on betas (0 = OLS)
    add_intercept: bool = True,
    align: str = "inner",                # {'inner','outer'}
    min_obs: Optional[int] = None,       # alias; see min_periods behavior below
    n_jobs: Optional[int] = None,        # reserved (parallelization omitted for clarity)
    regression_type: str = "OLS",        # {'OLS', 'quantile_regression'}
    regression_kwargs: Optional[Dict] = None,
) -> RegressionOutput:
    """
    Unified interface for OLS and Quantile Regression over time-indexed panels.

    Parameters mirror the original function for drop-in compatibility but the internal
    implementation is modular and easier to test/extend.
    """
    if mode not in {"per_asset", "global"}:
        raise ValueError("mode must be 'per_asset' or 'global'")
    if regression_type not in {"OLS", "quantile_regression"}:
        raise ValueError("regression_type must be 'OLS' or 'quantile_regression'")

    regression_kwargs = {} if regression_kwargs is None else dict(regression_kwargs)

    # OLS path
    if regression_type == "OLS":
        std_errors = regression_kwargs.get("std_errors", std_errors)
        nw_lags = regression_kwargs.get("nw_lags", nw_lags)
        l2 = float(regression_kwargs.get("l2", l2))
        add_intercept = bool(regression_kwargs.get("add_intercept", add_intercept))
        return _linear_regression_interface(
            y,
            predictors,
            mode=mode,
            rolling_window=rolling_window,
            stride_window=stride_window,
            add_intercept=add_intercept,
            std_errors=std_errors,
            nw_lags=nw_lags,
            l2=l2,
            align=align,
            min_periods=min_periods,
            min_obs=min_obs,
            n_jobs=n_jobs,
            regression_label="OLS",
        )

    # Quantile Regression path
    y_aligned, X_list, xnames, _, min_required, effective_min_periods = _prepare_linear_inputs(
        y,
        predictors,
        align=align,
        add_intercept=add_intercept,
        rolling_window=rolling_window,
        min_periods=min_periods,
        min_obs=min_obs,
    )

    qs = _check_quantiles(regression_kwargs.get("quantiles", regression_kwargs.get("quantile", 0.5)))
    primary_q = regression_kwargs.get("primary_quantile", (0.5 if 0.5 in qs else qs[0]))
    if primary_q not in qs:
        qs = [primary_q] + qs

    engine = _qr_per_asset if mode == "per_asset" else _qr_global
    out = engine(
        y_aligned,
        X_list,
        xnames,
        rolling_window=rolling_window,
        stride_window=stride_window,
        add_intercept=add_intercept,
        quantiles=qs,
        primary_q=float(primary_q),
        min_required=min_required,
    )

    meta = {
        "mode": mode,
        "std_errors": None,
        "nw_lags": None,
        "l2": None,
        "add_intercept": add_intercept,
        "rolling_window": rolling_window,
        "stride_window": stride_window,
        "min_periods": effective_min_periods,
        "min_obs": min_required,
        "n_jobs": n_jobs,
        "regression_type": "quantile_regression",
        "regression_kwargs": regression_kwargs,
        "primary_quantile": float(primary_q),
        "statsmodels_available": bool(_HAS_STATSMODELS),
        "align": align,
    }
    out.info["meta"] = meta
    return out

import pickle
def init_model():
    with open('./model_weights.pkl', 'rb') as f:
        models = pickle.load(f)
    return models
