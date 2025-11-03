from typing import Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.dataset import WindowsDataset
import numpy as np

class Seq2FuturePriceDataset(Dataset):
    """
    - Handles features with leading NaNs only (warm-up), no internal gaps.
    - Fills the leading block per feature with a configurable 'prefix_fill':
        'first_valid' (default): left-pad with the first observed value in the window;
        'mean': pad with dataset mean (NaN-aware);
        'zero': pad with 0.0;
        'linear_to_first_mean': linear ramp from dataset mean -> first valid across warm-up.
      Optionally use (long_warmup_linear=True, long_warmup_threshold=k) to apply the ramp for long prefixes.
    - Optionally appends missingness indicators for each feature as extra channels.
    """
    def __init__(
        self,
        X: torch.Tensor,                  # (N, 60, d)
        Y_future_levels: Optional[torch.Tensor] = None,    # (N, 10) future price levels
        p0_tensor: Optional[torch.Tensor] = None,    # (N,1) last observed price if not inside X
        standardize_X: bool = True,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        allow_nans: bool = True,
        add_missing_indicators: bool = True,
        impute_strategy: str = "mean",           # no internal gaps -> 'mean' is fine
        prefix_fill: str = "first_valid",        # 'first_valid' | 'mean' | 'zero' | 'linear_to_first_mean'
        long_warmup_linear: bool = True,         # enable ramp for long prefixes
        long_warmup_threshold: int = 20          # length threshold for ramp
    ):
        assert X.ndim == 3 and X.shape[1] == 60, "X must be (N, 60, d)"
        self.X_raw = X.float().clone()

        if Y_future_levels is not None:
            assert Y_future_levels.ndim == 2 and Y_future_levels.shape[1] == 10, "Y must be (N, 10)"
            self.Y = Y_future_levels.float().clone().unsqueeze(-1)  # (N, 10, 1)
        else:
            self.Y = None

        self.N, _, self.d = self.X_raw.shape

        self.add_missing_indicators = add_missing_indicators
        self.impute_strategy = impute_strategy
        self.prefix_fill = prefix_fill
        self.long_warmup_linear = long_warmup_linear
        self.long_warmup_threshold = long_warmup_threshold

        obs_mask = ~torch.isnan(self.X_raw)  # True where observed

        # Compute dataset μ/σ from *observed* values only (NaN-aware)
        flat = self.X_raw.reshape(-1, self.d)
        if mean is None:
            mask = ~torch.isnan(flat)
            mu = torch.sum(torch.where(mask, flat, torch.zeros_like(flat)), dim=0) / mask.sum(dim=0).clamp_min(1)
        else:
            mu = mean

        if std is None:
            mask = ~torch.isnan(flat)
            diffsq = torch.where(mask, (flat - mu) ** 2, torch.zeros_like(flat))
            sigma = torch.sqrt(diffsq.sum(dim=0) / mask.sum(dim=0).clamp_min(1))
            sigma = torch.clamp(sigma, min=1e-6)
        else:
            sigma = std

        if allow_nans:
            def fill_prefix(x_samp: torch.Tensor, m_samp: torch.Tensor) -> torch.Tensor:
                out = x_samp.clone()
                for j in range(self.d):
                    mj = m_samp[:, j]
                    valid_idx = torch.nonzero(mj, as_tuple=False)
                    if valid_idx.numel() == 0:
                        continue  # all NaN for this feature
                    first_idx = int(valid_idx[0].item())
                    if first_idx == 0:
                        continue  # no leading NaNs
                    if self.prefix_fill == "first_valid":
                        out[:first_idx, j] = x_samp[first_idx, j]
                    elif self.prefix_fill == "mean":
                        out[:first_idx, j] = mu[j]
                    elif self.prefix_fill == "zero":
                        out[:first_idx, j] = 0.0
                    elif self.prefix_fill == "linear_to_first_mean" or (
                            self.long_warmup_linear and first_idx >= self.long_warmup_threshold
                    ):
                        start_val = mu[j]
                        end_val = x_samp[first_idx, j]
                        ramp = torch.linspace(0.0, 1.0, steps=int(first_idx), device=x_samp.device, dtype=x_samp.dtype)
                        out[:first_idx, j] = (1 - ramp) * start_val + ramp * end_val
                    else:
                        out[:first_idx, j] = x_samp[first_idx, j]
                return out

            # 1) Prefix-only fill (your case; no internal gaps expected)
            X_prefilled = torch.stack([fill_prefix(self.X_raw[i], obs_mask[i]) for i in range(self.N)], dim=0)

            # 2) Fill any remaining NaNs (e.g., entire feature missing in window) by mean/zero
            rem_mask = ~torch.isnan(X_prefilled)
            if self.impute_strategy in ("mean",):
                X_imp = torch.where(rem_mask, X_prefilled, mu.view(1, 1, self.d))
            elif self.impute_strategy == "zero":
                X_imp = torch.where(rem_mask, X_prefilled, torch.zeros_like(X_prefilled))
            else:
                X_imp = X_prefilled  # no-op for this scenario

            # 3) Standardize (optional)
            X_std = (X_imp - mu.view(1, 1, self.d)) / sigma.view(1, 1, self.d) if standardize_X else X_imp

            # 4) Optional mask channels (useful if features have different warm-ups)
            if self.add_missing_indicators:
                self.X = torch.cat([X_std, obs_mask.float()], dim=-1)  # (N,60, d+d)
                self.d_out = self.d * 2
            else:
                self.X = X_std
                self.d_out = self.d

            self.mu, self.sigma = mu, sigma
            self.mask = obs_mask
        else:
            # No NaNs allowed: standard path
            self.mu, self.sigma = mu, sigma
            self.X = (self.X_raw - mu.view(1, 1, self.d)) / sigma.view(1, 1, self.d) if standardize_X else self.X_raw
            self.d_out = self.d

        # p0 (last observed price level)
        if p0_tensor is not None:
            assert p0_tensor.shape == (self.N, 1)
            self.p0 = p0_tensor.float().clone()
        else:
            raise ValueError("Provide p0_tensor explicitly.")
        self.X = self.X[:,:,:20]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx], self.p0[idx]
        else:
            return self.X[idx], self.p0[idx]

def make_loaders(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    p0_train: Optional[torch.Tensor] = None,
    p0_val: Optional[torch.Tensor] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    standardize_X: bool = True
):
    ds_tr = Seq2FuturePriceDataset(X_train, Y_train, p0_tensor=p0_train, standardize_X=standardize_X)
    ds_va = Seq2FuturePriceDataset(X_val, Y_val, p0_tensor=p0_val, standardize_X=standardize_X, mean=ds_tr.mu, std=ds_tr.sigma)

    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr, va, ds_tr.mu, ds_tr.sigma


def data_split(df: pd.DataFrame, step_size: int, max_samples: int, feature_generator: Callable = None):
    samples = WindowsDataset(rolling=True, step_size=step_size, max_samples=max_samples, df=df)
    X, y = samples.X, samples.y
    LLP = torch.log(torch.from_numpy(X[:, -1, 0])).unsqueeze(dim=1)
    if feature_generator is not None:
        X = feature_generator(X)

    X, y = torch.from_numpy(X), torch.from_numpy(y)
    log_y = torch.log(y)

    return X, log_y, LLP
