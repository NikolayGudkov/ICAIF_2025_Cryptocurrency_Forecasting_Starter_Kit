"""
This module implements preprocessing utilities.

It includes a column-wise transformer that applies different transformations
to specified columns of a DataFrame, and a feature pipeline that chains multiple
transformations together with support for inverse transformations.
"""
from pathlib import Path
from typing import Optional, Iterator, Callable

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

from src.dataset import TrainWindowSampler


# -------------------------------
# Base Transform Interface
# -------------------------------
class Transform:
    """Base interface for all feature transforms."""
    def __call__(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Optional inverse operation."""
        return X


# -------------------------------
#  Column-wise transformer
# -------------------------------
class ColumnWiseTransformer(Transform):
    """
    Name-based column transformer.
    NOTE: For the moment, we refir everytime when we apply the transform, since the transformers are
    fit on every single sample.
    """
    def __init__(self, transformers: dict):
        self.transformers = transformers
        self.fitted = False
        self.feature_names_ = None

    # def fit(self, X: np.ndarray | pd.DataFrame):
    #     self.feature_names_ = list(X.columns)
    #     for col, t in self.transformers.items():
    #         if col not in X.columns:
    #             raise KeyError(f"Column `{col}` not found in data.")
    #         t.fit(X[[col]])
    #     self.fitted = True
    #     return self

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names_ = list(X.columns)
        X_t = X.copy()
        for col, t in self.transformers.items():
            X_t[col] = t.fit_transform(X[[col]]).ravel()
        return X_t.to_numpy(np.float32)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("Transformer not fitted --> cannot perform inverse_transform")
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        for col, t in self.transformers.items():
            if hasattr(t, "inverse_transform"):
                X_df[col] = t.inverse_transform(X_df[[col]]).ravel()
        return X_df


# -------------------------------
#  FeaturePipeline
# -------------------------------
class FeaturePipeline:
    """Composable transform pipeline with inverse functions."""
    def __init__(self, steps: list[Transform]):
        self.steps = steps

    def __call__(self, X):
        for step in self.steps:
            X = step(X)
        return X

    def inverse_transform(self, X):
        for step in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                X = step.inverse_transform(X)
        return X


class FeatureDataset(Dataset):
    """
    Dataset that wraps the competition's TrainWindowSampler and applies a
    transform or pipeline to each window.

    Note: Works only with materialize=True: preloads all samples into memory.

    Parameters
    ----------
    data_path : str
        Path to either `train.pkl` or `x_test.pkl`.
    pipeline : Optional[callable]
        A composed pipeline (e.g., FeaturePipeline).
    materialize : bool, default=True
        Whether to load all samples into memory.
    """
    def __init__(
        self,
        data_path: str | Path,
        pipeline: Optional[callable] = None,
        materialize: bool = True
    ):
        if not materialize:
            raise ValueError("For map-style Dataset with random access, set materialize=True or use FeatureIterableDataset.")
        self.pipeline = pipeline
        self.samples = list(TrainWindowSampler(data_path, rolling=True).iter_windows())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, Y = self.samples[idx]
        if self.pipeline:
            X = self.pipeline(X)
        if not isinstance(X, np.ndarray):
            X = X.to_numpy(np.float32)
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


class FeatureIterableDataset(IterableDataset):
    """
    True streaming dataset: yields (X, y) from TrainWindowSampler and
    applies an optional pipeline callable on the fly.
    """
    def __init__(self, data_path: str | Path, pipeline: Callable | None = None):
        super().__init__()
        self.data_path = data_path
        self.pipeline = pipeline

    def _iter_all(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        sampler = TrainWindowSampler(self.data_path, rolling=True)
        for X, Y in sampler.iter_windows():
            if self.pipeline is not None:
                X = self.pipeline(X)
            yield torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Optional: shard iterator across workers
        info = get_worker_info()
        if info is None:
            # Single-process
            yield from self._iter_all()
        else:
            # Multi-worker: shard by round-robin
            worker_id, num_workers = info.id, info.num_workers
            for i, item in enumerate(self._iter_all()):
                if i % num_workers == worker_id:
                    yield item


# ---------------------------------------------------------------------
# Feature Engineering Transforms
# ---------------------------------------------------------------------
class BasicFeatureGenerator(Transform):
    """
    Generates additional time-series features from [close, volume]:
      - log return
      - rolling volatility
      - volume delta
    """
    def __init__(self, vol_window: int = 5) -> None:
        self.vol_window = vol_window
        self.columns_ = ["close", "volume", "log_ret", "vol_delta", "rolling_vol"]

    def __call__(self, X: np.ndarray) -> pd.DataFrame:
        close, vol = X[:, 0], X[:, 1]
        log_ret = np.diff(np.log(close + 1e-6), prepend=np.log(close[0] + 1e-6))
        vol_delta = np.diff(vol, prepend=vol[0])
        rolling_vol = np.array([
            np.std(log_ret[max(0, i - self.vol_window): i + 1])
            for i in range(len(log_ret))
        ], dtype=np.float32)
        features = np.stack([close, vol, log_ret, vol_delta, rolling_vol], axis=1)
        features_df = pd.DataFrame(features, columns=self.columns_)
        return features_df


# class ZScoreNormalize(Transform):
#     """Per-window z-score normalization (for small features)."""
#     def __init__(self, eps: float = 1e-6):
#         self.eps = eps
#         self.mean_: Optional[np.ndarray] = None
#         self.std_: Optional[np.ndarray] = None
#
#     def __call__(self, X: np.ndarray) -> np.ndarray:
#         self.mean_ = X.mean(0)
#         self.std_ = np.clip(X.std(0), self.eps, None)
#         return (X - self.mean_) / self.std_
#
#     def inverse_transform(self, X: np.ndarray) -> np.ndarray:
#         if self.mean_ is None or self.std_ is None:
#             raise RuntimeError("ZScoreNormalize must be fitted first (call __call__)")
#         return X * self.std_ + self.mean_
