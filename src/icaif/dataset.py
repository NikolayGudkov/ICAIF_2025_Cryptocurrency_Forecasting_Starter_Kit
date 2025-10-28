import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

class TrainWindowSampler:
    """
    Build (X, y) samples from train by slicing non-overlapping or rolling windows of length 70,
    where X is the first 60 steps (close, volume) and y is the next 10 steps (close).
    """
    def __init__(
        self,
        df : pd.DataFrame,
        train_path: str = None,
        window: int = 70,
        input_len: int = 60,
        horizon_len: int = 10,
        rolling: bool = True,
        step_size: int = 1,
        seed: int = 42,
    ) -> None:
        assert input_len + horizon_len == window, "window must equal input_len + horizon_len"
        self.input_len = input_len
        self.horizon_len = horizon_len
        self.window = window
        self.rolling = rolling
        self.step_size = 1 if rolling else window
        if step_size is not None:
            self.step_size = step_size

        if train_path is not None:
            self.df = pd.read_pickle(train_path)
        else:
            self.df = df

        # Expect columns: ['series_id','time_step','close','volume']
        required = {'series_id','time_step','close','volume'}
        if not required.issubset(self.df.columns):
            raise ValueError(f"train missing columns, found {self.df.columns}")

        self.rng = np.random.default_rng(seed)

        # group index for sampling
        self.groups = {sid: g.sort_values('time_step').reset_index(drop=True)
                       for sid, g in self.df.groupby('series_id')}

    def __len__(self) -> int:
        total = 0
        for g in self.groups.values():
            n = len(g)
            if n < self.window:
                continue
            if self.rolling:
                total += (n - self.window) // self.step_size + 1
            else:
                total += n // self.window
        return total

    def iter_windows(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (X, y) where:
        X: (60, 2) float32 -> columns [close, volume]
        y: (10,)  float32 -> close only
        """
        for g in self.groups.values():
            n = len(g)
            if n < self.window:
                continue

            arr = g[['close', 'volume']].to_numpy(np.float32)  

            for s in range(0, n - self.window + 1, self.step_size):

                chunk = arr[s:s + self.window]               
                x = chunk[:self.input_len]                   
                y = chunk[self.input_len:, 0]                 
                yield x, y



class TrainWindowSamplerVect:
    """
    Holds configuration and grouped data for the vectorized dataset.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        train_path: str = None,
        window: int = 70,
        input_len: int = 60,
        horizon_len: int = 10,
        rolling: bool = True,
        step_size: int = 1,
        seed: int = 42,
    ) -> None:
        assert input_len + horizon_len == window, "window must equal input_len + horizon_len"
        self.input_len = input_len
        self.horizon_len = horizon_len
        self.window = window
        self.rolling = rolling
        self.step_size = 1 if rolling else window
        if step_size is not None:
            self.step_size = step_size

        if train_path is not None:
            self.df = pd.read_pickle(train_path)
        else:
            self.df = df

        required = {"series_id", "time_step", "close", "volume", "event_datetime"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"train missing columns {required - set(self.df.columns)}, found {self.df.columns}")

        self.groups = {
            sid: g.sort_values("time_step").reset_index(drop=True)
            for sid, g in self.df.groupby("series_id")
        }