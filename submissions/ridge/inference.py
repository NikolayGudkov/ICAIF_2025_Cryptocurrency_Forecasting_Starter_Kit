"""Inference utilities for the Ridge submission.

This module exposes `generate_forecast(x_test_path)` which loads the trained
ridge model (via `init_model`) and produces `submission.pkl` containing the
required forecast DataFrame.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Ensure local imports work whether run as module or script
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model import init_model  # type: ignore  # noqa: E402


SubmissionPath = Union[str, Path]


def _pivot_feature_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot(index="window_id", columns="time_step", values=value_col)
    if pivot.isna().any().any():
        raise ValueError(f"Missing values detected after pivoting {value_col}")
    return pivot


def _extract_inputs(x_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_cols = {"window_id", "time_step", "close", "volume"}
    if not required_cols.issubset(x_test.columns):
        missing = required_cols.difference(x_test.columns)
        raise ValueError(f"x_test is missing required columns: {sorted(missing)}")

    x_sorted = x_test.sort_values(["window_id", "time_step"]).copy()
    close_pivot = _pivot_feature_matrix(x_sorted, "close")
    volume_pivot = _pivot_feature_matrix(x_sorted, "volume")

    prices60 = close_pivot.to_numpy(dtype=np.float64)
    volumes60 = volume_pivot.to_numpy(dtype=np.float64)
    window_ids = close_pivot.index.to_numpy()

    if prices60.shape[1] != 60 or volumes60.shape[1] != 60:
        raise ValueError(
            "Expected 60 historical steps per window. "
            f"Got close shape {prices60.shape}, volume shape {volumes60.shape}."
        )

    return window_ids, prices60, volumes60


def _build_submission_df(window_ids: Iterable[int], predictions: np.ndarray) -> pd.DataFrame:
    n_samples, horizon = predictions.shape
    window_ids = np.asarray(list(window_ids))
    if window_ids.shape[0] != n_samples:
        raise ValueError("Window ID count does not match prediction rows")

    time_steps = np.tile(np.arange(horizon, dtype=np.int8), n_samples)
    window_col = np.repeat(window_ids.astype(np.int32), horizon)
    pred_flat = predictions.astype(np.float32).reshape(-1)

    submission = pd.DataFrame({
        "window_id": window_col,
        "time_step": time_steps,
        "pred_close": pred_flat,
    })

    return submission


def generate_forecast(x_test_path: SubmissionPath, output_path: Optional[SubmissionPath] = None) -> Path:
    """Generate forecasts for `x_test` and persist them to `submission.pkl`.

    Parameters
    ----------
    x_test_path: str or Path
        Location of the pickled x_test DataFrame with columns
        [window_id, time_step, close, volume].
    output_path: Optional[str or Path]
        Custom output location for submission.pkl. Defaults to the same
        directory as this file.

    Returns
    -------
    Path to the saved `submission.pkl` file.
    """

    x_test_path = Path(x_test_path)
    if not x_test_path.exists():
        raise FileNotFoundError(f"x_test file not found at {x_test_path}")

    x_test = pd.read_pickle(x_test_path)
    window_ids, prices60, volumes60 = _extract_inputs(x_test)

    model = init_model()
    price_pred = model.predict(prices60, volumes60)

    submission_df = _build_submission_df(window_ids, price_pred)

    out_path = Path(output_path) if output_path is not None else THIS_DIR / "submission.pkl"
    submission_df.to_pickle(out_path)
    return out_path


if __name__ == "__main__":
    default_x_test = THIS_DIR.parent.parent / "data" / "x_test.pkl"
    output = generate_forecast(default_x_test)
    print(f"Saved predictions to {output}")



