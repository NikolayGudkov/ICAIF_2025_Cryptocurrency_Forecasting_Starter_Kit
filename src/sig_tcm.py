from dataclasses import dataclass
from typing import Tuple, Union
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import pickle


# ==========================
# Path augmentations for (log-)signature
# ==========================
def time_augment(path: torch.Tensor) -> torch.Tensor:
    """Append normalized time channel to a path: (B,L,k)->(B,L,k+1)."""
    B, L, _ = path.shape
    t = torch.linspace(0.0, 1.0, L, device=path.device, dtype=path.dtype).view(1, L, 1).expand(B, L, 1)
    return torch.cat([path, t], dim=-1)

def lead_lag(path: torch.Tensor) -> torch.Tensor:
    """Lead–lag augmentation on channels: (B,L,c)->(B,L,2c)."""
    x_rep = torch.repeat_interleave(path, repeats=2, dim=1)
    return torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)[:,:,:-1]

def add_basepoint(path: torch.Tensor, base_value: float = 0.0) -> torch.Tensor:
    """Prepend a basepoint row: (B,L,c)->(B,L+1,c)."""
    B, _, c = path.shape
    base = torch.full((B, 1, c), base_value, device=path.device, dtype=path.dtype)
    return torch.cat([base, path], dim=1)


@dataclass
class Config:
    dim_in: int = 20
    steps: int = 10
    T_in: int = 60
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 5
    sig_depth: int = 3
    use_logsig: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    head_hidden_dim: int = 256
    encoder_channels: Tuple[int, ...] = (64, 128, 128, 256)
    encoder_k: int = 3
    encoder_pdrop: float = 0.1
    sig_loss: bool = False


# ==========================
# TCN encoder → 10 returns → reconstruct levels
# ==========================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, k: int = 3, pdrop: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)               # (B, C_out, T) after chomps
        y = y + self.skip(x)          # residual OK
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return y

class EncoderTCN(nn.Module):
    def __init__(self, d_in: int, channels: Tuple[int, ...], k: int, pdrop: float):
        super().__init__()
        blocks = []
        c_prev = d_in
        for i, c_out in enumerate(channels):
            blocks.append(TCNBlock(c_prev, c_out, dilation=2 ** i, k=k, pdrop=pdrop))
            c_prev = c_out
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c_prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,60,d_in)
        x = x.transpose(1, 2)         # (B,d_in,60)
        h = self.tcn(x)               # (B,C,60)
        z = self.pool(h).squeeze(-1)  # (B,C)
        return z

class FutureHead(nn.Module):
    def __init__(self, dim_in: int, hidden_dim: int, _steps: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, _steps))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        log_r_pred = self.mlp(z)                          # (B, steps)
        return torch.cumsum(log_r_pred, dim=1).unsqueeze(-1)  # (B, steps)         # (B,steps,1)


# ==========================
# Loss & Training
# ==========================

import torch.nn.functional as F


def autocorr(x, demean=True, norm='coeff'):
    """
    x: (B, T, C) tensor (CPU or CUDA)
    norm: 'none' | 'biased' | 'unbiased' | 'coeff'
    returns: (B, T, C) autocorr over non-negative lags
    """
    assert x.dim() == 3, "x must be (B, T, C)"
    B, T, C = x.shape
    x = x.to(torch.float32)

    # to (B, C, T) for conv1d, then optional de-mean per (b,c)
    x_bcT = x.transpose(1, 2)  # (B, C, T)
    if demean:
        x_bcT = x_bcT - x_bcT.mean(dim=2, keepdim=True)

    # Treat each (b,c) as a separate channel via reshape
    x_chT = x_bcT.reshape(1, B * C, T)                  # input:  (N=1, channels=B*C, T)
    w_chT = x_bcT.flip(dims=[2]).reshape(B * C, 1, T)   # weight: (out=B*C, in=1, T)

    r_full = F.conv1d(x_chT, w_chT, padding=T - 1, groups=B * C)  # (1, B*C, 2T-1)
    r_full = r_full.reshape(B, C, -1)                              # (B, C, 2T-1)
    r = r_full[:, :, T - 1:]                                       # keep lags 0..T-1  -> (B, C, T)

    # normalizations
    if norm == 'biased':
        r = r / T
    elif norm == 'unbiased':
        denom = torch.arange(T, 0, -1, device=r.device, dtype=r.dtype).view(1, 1, T)
        r = r / denom
    elif norm == 'coeff':
        r0 = r[:, :, :1].abs().clamp_min(1e-12)  # r(0)
        r = r / r0

    return r.transpose(1, 2)

class SigPathLoss(nn.Module):
    def __init__(self, lam_path: float = 0.1, lam_corr: float = 0.1):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.lam_path = lam_path
        self.lam_corr = lam_corr

    def forward(self, outputs: dict) -> torch.Tensor:
        L_path = self.l2(outputs["log_y_pred_levels"], outputs["log_y_true_levels"])
        L_corr = self.l2(outputs['auto_corr_pred'], outputs['auto_corr_true'])

        if outputs["S_pred"] is not None:
            L_sig = self.l2(outputs["S_pred"], outputs["S_true"])
            return L_sig + self.lam_path * L_path + self.lam_corr * L_corr
        else:
            return self.lam_path * L_path + self.lam_corr * L_corr


class SigLossTCN(nn.Module):
    def __init__(self, cnf: Config):
        super().__init__()
        self.encoder = EncoderTCN(d_in = cnf.dim_in, channels=cnf.encoder_channels, k = cnf.encoder_k, pdrop=cnf.encoder_pdrop)
        self.head = FutureHead(dim_in = self.encoder.out_dim, hidden_dim=cnf.head_hidden_dim, _steps=cnf.steps)
        self.depth = cnf.sig_depth
        self.use_logsig = cnf.use_logsig
        self.mu = None
        self.sig = None
        self.cnf = cnf


    def signature(self, path_levels: torch.Tensor) -> torch.Tensor:
        path = time_augment(path_levels)
        path = lead_lag(path)
        path = add_basepoint(path)

        if self.cnf.sig_loss:
            import signatory
            return signatory.logsignature(path, self.depth) if self.use_logsig else signatory.signature(path, self.depth)
        else:
            return None

    def forward(self, x: torch.Tensor, log_last_price: torch.Tensor):
        z = self.encoder(x)
        log_y_pred_levels = log_last_price.unsqueeze(-1) + torch.cumsum(self.head(z), dim=1)
        return log_y_pred_levels


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


# -------- Required by platform --------
def init_model(weights_path: Union[Path, str] = "model_weights.pkl") -> SigLossTCN:
    """Factory function required for submission."""
    if weights_path and Path(weights_path).exists():
        cfg = Config()
        model = SigLossTCN(cfg).to('cpu')
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict['best_state'])
        model.mu, model.sig = state_dict['ds_train_mean'], state_dict['ds_train_std']
        model.eval()
        return model
    else:
        raise FileNotFoundError('Weights are not found.')