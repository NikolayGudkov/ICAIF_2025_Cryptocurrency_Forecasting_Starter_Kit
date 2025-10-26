"""
TCN → 10‑step price path trained with Log‑Signature loss

Your setup:
- Input: past 60 minutes with d features → X: (N, 60, d)
- Target: 10 future price *levels* (1D) → Y: (N, 10)
- Objective: compare predicted vs true future *paths* in log‑signature space (with time + lead‑lag + basepoint augmentation),
  plus small auxiliaries on levels.

Requirements:
    pip install torch signatory numpy

How to use (outline):
1) Prepare tensors: X_train, Y_train, X_val, Y_val. Optionally provide `price_feature_index` if the last observed price is inside X.
2) Set config in `Config` (especially d_in = X.shape[-1]).
3) Run `train()`.

Notes:
- The model predicts 10 future *returns* and reconstructs future *levels* from the last observed price p0.
- If your price is in X (e.g., first feature), set `price_feature_index=0` in the dataset to extract p0 automatically.
- If not, provide p0 explicitly via the dataset argument `p0_tensor` with shape (N, 1).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import signatory
from src.dataset import WindowsDataset
from pathlib import Path
import sys
import pickle


# ==========================
# Utilities: path augmentation
# ==========================

def time_augment(path: torch.Tensor) -> torch.Tensor:
    """Append normalized time channel to a path.
    path: (B, L, k)
    returns: (B, L, k+1)
    """
    B, L, _ = path.shape
    t = torch.linspace(0.0, 1.0, L, device=path.device).view(1, L, 1).expand(B, L, 1)
    return torch.cat([path, t], dim=-1)


def lead_lag(path: torch.Tensor) -> torch.Tensor:
    """Lead–lag augmentation on the channel dimension.
    Input: (B, L, c) → Output: (B, L, 2c)
    """
    lead = path
    lag = torch.cat([path[:, :1, :], path[:, :-1, :]], dim=1)
    return torch.cat([lead, lag], dim=-1)


def add_basepoint(path: torch.Tensor, base_value: float = 0.0) -> torch.Tensor:
    """Prepend a basepoint row of `base_value`.
    (B, L, c) → (B, L+1, c)
    """
    B, _, c = path.shape
    base = torch.full((B, 1, c), base_value, device=path.device, dtype=path.dtype)
    return torch.cat([base, path], dim=1)


# ==========================
# Dataset
# ==========================
class Seq2FuturePriceDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,             # (N, 60, d)
        Y_future_levels: torch.Tensor,  # (N, 10)
        price_feature_index: Optional[int] = None,  # index in X for price feature to get p0
        p0_tensor: Optional[torch.Tensor] = None,   # (N, 1) last observed price, if not extractable from X
        standardize_X: bool = True,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        assert X.ndim == 3 and X.shape[1] == 60, "X must be (N, 60, d)"
        assert Y_future_levels.ndim == 2 and Y_future_levels.shape[1] == 10, "Y must be (N, 10)"
        self.X = X.float().clone()
        self.Y = Y_future_levels.float().clone().unsqueeze(-1)  # (N, 10, 1)
        self.N, _, self.d = self.X.shape

        # Determine p0 (last observed level at t)
        if p0_tensor is not None:
            assert p0_tensor.shape == (self.N, 1)
            self.p0 = p0_tensor.float().clone()
        elif price_feature_index is not None:
            self.p0 = self.X[:, -1, price_feature_index].unsqueeze(-1)
        else:
            raise ValueError("Provide either price_feature_index to extract p0 from X, or p0_tensor explicitly.")

        # Standardize X using provided mean/std or compute from this dataset
        self.mu = mean
        self.sigma = std
        if standardize_X:
            if self.mu is None:
                self.mu = self.X.reshape(self.N * 60, self.d).mean(dim=0)
            if self.sigma is None:
                self.sigma = self.X.reshape(self.N * 60, self.d).std(dim=0).clamp_min(1e-6)
            self.X = (self.X - self.mu.view(1, 1, self.d)) / self.sigma.view(1, 1, self.d)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.p0[idx]


# ==========================
# Model: TCN encoder → 10-step returns → reconstruct levels
# ==========================
class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, k: int = 3, pdrop: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            nn.ReLU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        y = self.net(x)
        #y = y + self.skip(x)
        # layernorm over channels: transpose to (B, T, C)
        y_ln = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return y_ln


class EncoderTCN(nn.Module):
    def __init__(self, d_in: int, channels=(60, 128, 128, 256, 256), k: int = 3, pdrop: float = 0.1):
        super().__init__()
        blocks = []
        c_prev = d_in
        for i, c_out in enumerate(channels):
            blocks.append(TCNBlock(c_prev, c_out, dilation=2 ** i, k=k, pdrop=pdrop))
            c_prev = c_out
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c_prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 60, d)
        x = x.transpose(1, 2)  # (B, d, 60)
        h = self.tcn(x)        # (B, C, 60)
        z = self.pool(h).squeeze(-1)  # (B, C)
        return z


class FutureHead(nn.Module):
    def __init__(self, in_dim: int, steps: int = 10):
        super().__init__()
        self.steps = steps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, steps)
        )

    def forward(self, z: torch.Tensor, p0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return levels and returns.
        z: (B, C); p0: (B, 1)
        returns:
            p_pred: (B, steps, 1) price levels
            r_pred: (B, steps)    returns
        """
        r_pred = self.mlp(z)  # (B, steps)
        levels = p0 + torch.cumsum(r_pred, dim=1)  # (B, steps)
        return levels.unsqueeze(-1), r_pred


class SigLossTCN(nn.Module):
    def __init__(self, d_in: int, steps: int = 10, logsig_depth: int = 3, use_logsig: bool = True):
        super().__init__()
        self.encoder = EncoderTCN(d_in)
        self.head = FutureHead(self.encoder.out_dim, steps=steps)
        self.depth = logsig_depth
        self.use_logsig = use_logsig

    def signature(self, path_levels: torch.Tensor) -> torch.Tensor:
        """Compute (log-)signature of augmented path.
        path_levels: (B, L, 1)
        returns: (B, D)
        """
        path = path_levels
        path = time_augment(path)    # (B, L, 2)
        path = lead_lag(path)        # (B, L, 4)
        path = add_basepoint(path)   # (B, L+1, 4)
        if self.use_logsig:
            return signatory.logsignature(path, self.depth)
        else:
            return signatory.signature(path, self.depth)

    def forward(self, x: torch.Tensor, y_true_levels: torch.Tensor, p0: torch.Tensor):
        """
        x: (B, 60, d)
        y_true_levels: (B, 10, 1)
        p0: (B, 1)
        returns: dict with predictions and signatures
        """
        z = self.encoder(x)
        y_pred_levels, y_pred_returns = self.head(z, p0)
        S_pred = self.signature(y_pred_levels)
        S_true = self.signature(y_true_levels)
        return {
            "y_pred_levels": y_pred_levels,
            "y_pred_returns": y_pred_returns,
            "S_pred": S_pred,
            "S_true": S_true,
        }


# ==========================
# Loss & Training
# ==========================
class SigPathLoss(nn.Module):
    def __init__(self, lam_path: float = 0.1, lam_end: float = 0.1):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.lam_path = lam_path
        self.lam_end = lam_end

    def forward(self, outputs: dict, y_true_levels: torch.Tensor) -> torch.Tensor:
        L_sig = self.l2(outputs["S_pred"], outputs["S_true"])  # primary
        L_path = self.l2(outputs["y_pred_levels"], y_true_levels)
        L_end = self.l2(outputs["y_pred_levels"][:, -1], y_true_levels[:, -1])
        return L_sig + self.lam_path * L_path + self.lam_end * L_end


@dataclass
class Config:
    d_in: int = 8            # set this to X.shape[-1]
    steps: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 3
    logsig_depth: int = 4
    use_logsig: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_loaders(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    price_feature_index: Optional[int] = 0,
    p0_train: Optional[torch.Tensor] = None,
    p0_val: Optional[torch.Tensor] = None,
    batch_size: int = 128,
    num_workers: int = 0,
):
    ds_tr = Seq2FuturePriceDataset(X_train, Y_train, price_feature_index=price_feature_index, p0_tensor=p0_train)
    ds_va = Seq2FuturePriceDataset(X_val, Y_val, price_feature_index=price_feature_index, p0_tensor=p0_val,
                                   standardize_X=True, mean=ds_tr.mu, std=ds_tr.sigma)
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr, va


def train(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    price_feature_index: Optional[int] = 0,
    p0_train: Optional[torch.Tensor] = None,
    p0_val: Optional[torch.Tensor] = None,
    cfg: Config = Config(),
):
    device = cfg.device
    train_loader, val_loader = make_loaders(
        X_train, Y_train, X_val, Y_val,
        price_feature_index=price_feature_index,
        p0_train=p0_train, p0_val=p0_val,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    model = SigLossTCN(d_in=cfg.d_in, steps=cfg.steps, logsig_depth=cfg.logsig_depth, use_logsig=cfg.use_logsig).to(device)
    criterion = SigPathLoss(lam_path=0.1, lam_end=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    def run_epoch(loader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()
        total_loss, n = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for Xb, Yb, P0b in loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                P0b = P0b.to(device)
                out = model(Xb, Yb, P0b)
                loss = criterion(out, Yb)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_loss += loss.item() * Xb.size(0)
                n += Xb.size(0)
        return total_loss / max(n, 1)

    best_val, best_state = float('inf'), None
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = run_epoch(train_loader, True)
        va_loss = run_epoch(val_loader, False)
        sched.step()
        print(f"Epoch {epoch:3d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ==========================
# Example synthetic usage (delete or adapt)
# ==========================
if __name__ == "__main__":
    torch.manual_seed(0)
    N_train, N_val, T_in, d = 2048, 512, 60, 2
    steps = 10
    # Paths (adjust if your layout differs)
    ROOT = Path.cwd().parent if (Path.cwd().name == 'src') else Path.cwd()
    DATA = ROOT / "data"
    SRC = ROOT / "src"
    SUBM = ROOT / "sample_submission"

    train_path = DATA / "train.parquet"
    weights_path = SUBM / "lstm_weights.pkl"

    # Ensure src is importable
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    # Create sample_submission dir if missing
    SUBM.mkdir(parents=True, exist_ok=True)

    SEED = 1337
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MAX_SAMPLES = 50000
    train_size = int(0.8 * MAX_SAMPLES)# set to None to use all windows
    train_ds = WindowsDataset(str(train_path), rolling=True, step_size=1, max_samples=MAX_SAMPLES)

    X_tr, Y_tr, X_va, Y_va = torch.from_numpy(train_ds.X[:train_size]), torch.from_numpy(train_ds.y[:train_size]), torch.from_numpy(train_ds.X[train_size:]), torch.from_numpy(train_ds.y[train_size:])

    cfg = Config(d_in=d, steps=steps, epochs=1)
    model = train(X_tr, Y_tr, X_va, Y_va, price_feature_index=0, cfg=cfg)

    torch.save(model.state_dict(), weights_path)

    model = SigLossTCN(d_in=cfg.d_in, steps=cfg.steps, logsig_depth=cfg.logsig_depth, use_logsig=cfg.use_logsig).to(DEVICE)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        ds_va = Seq2FuturePriceDataset(X_va, Y_va, price_feature_index=0)
        xb, yb, p0b = next(iter(DataLoader(ds_va, batch_size=32)))
        out = model(xb, yb, p0b)



        print("Predicted path shape:", out["y_pred_levels"].shape)
        print("LogSig dim:", out["S_pred"].shape[-1])


# ==========================
# MULTI-COIN SUPPORT (conditioning on coin ID + per-coin normalization)
# ==========================
from collections import defaultdict

class MultiCoinDataset(Dataset):
    """Like Seq2FuturePriceDataset, but:
    - accepts coin_ids: (N,) int64
    - performs *per-coin* standardization using training stats (pass stats_map from train to val)
    - returns (X, Y, p0, coin_id)
    """
    def __init__(
        self,
        X: torch.Tensor,             # (N, 60, d)
        Y_future_levels: torch.Tensor,  # (N, 10)
        coin_ids: torch.Tensor,      # (N,) int64
        price_feature_index: Optional[int] = None,
        p0_tensor: Optional[torch.Tensor] = None,   # (N, 1)
        standardize_X: bool = True,
        stats_map: Optional[dict] = None,           # {coin_id: (mean[d], std[d])} from TRAIN SPLIT ONLY
        compute_stats: bool = False,                 # True on train split to compute stats_map
    ):
        assert X.ndim == 3 and X.shape[1] == 60
        assert Y_future_levels.ndim == 2 and Y_future_levels.shape[1] == 10
        assert coin_ids.shape[0] == X.shape[0]
        self.X = X.float().clone()
        self.Y = Y_future_levels.float().clone().unsqueeze(-1)
        self.coin_ids = coin_ids.long().clone()
        self.N, _, self.d = self.X.shape

        # p0
        if p0_tensor is not None:
            assert p0_tensor.shape == (self.N, 1)
            self.p0 = p0_tensor.float().clone()
        elif price_feature_index is not None:
            self.p0 = self.X[:, -1, price_feature_index].unsqueeze(-1)
        else:
            raise ValueError("Provide either price_feature_index or p0_tensor.")

        # Per-coin standardization
        self.stats_map = {} if stats_map is None else {int(k): (v[0].clone(), v[1].clone()) for k, v in stats_map.items()}
        if compute_stats:
            # compute from this dataset per coin
            tmp = defaultdict(list)
            for cid in torch.unique(self.coin_ids):
                mask = (self.coin_ids == cid)
                Xc = self.X[mask].reshape(-1, self.d)
                mu = Xc.mean(dim=0)
                sd = Xc.std(dim=0).clamp_min(1e-6)
                self.stats_map[int(cid.item())] = (mu, sd)
        if standardize_X:
            X_std = torch.empty_like(self.X)
            for i in range(self.N):
                cid = int(self.coin_ids[i].item())
                mu, sd = self.stats_map[cid]
                X_std[i] = (self.X[i] - mu.view(1, self.d)) / sd.view(1, self.d)
            self.X = X_std

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.p0[idx], self.coin_ids[idx]


class CoinConditionedHead(nn.Module):
    """Head that conditions on coin embedding by concatenation (simple & effective)."""
    def __init__(self, in_dim: int, coin_dim: int, steps: int = 10):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(in_dim + coin_dim, 256), nn.ReLU(), nn.Linear(256, steps)
        )
    def forward(self, z: torch.Tensor, coin_emb: torch.Tensor, p0: torch.Tensor):
        r_pred = self.net(torch.cat([z, coin_emb], dim=-1))  # (B, steps)
        levels = p0 + torch.cumsum(r_pred, dim=1)
        return levels.unsqueeze(-1), r_pred


class SigLossTCNMultiCoin(nn.Module):
    """TCN encoder shared across coins + coin embedding conditioning for the head."""
    def __init__(self, d_in: int, num_coins: int, coin_dim: int = 16, steps: int = 10, logsig_depth: int = 3, use_logsig: bool = True):
        super().__init__()
        self.encoder = EncoderTCN(d_in)
        self.coin_emb = nn.Embedding(num_coins, coin_dim)
        nn.init.normal_(self.coin_emb.weight, std=0.02)
        self.head = CoinConditionedHead(self.encoder.out_dim, coin_dim, steps=steps)
        self.depth = logsig_depth
        self.use_logsig = use_logsig

    def signature(self, path_levels: torch.Tensor) -> torch.Tensor:
        path = time_augment(path_levels)
        path = lead_lag(path)
        path = add_basepoint(path)
        return signatory.logsignature(path, self.depth) if self.use_logsig else signatory.signature(path, self.depth)

    def forward(self, x: torch.Tensor, y_true_levels: torch.Tensor, p0: torch.Tensor, coin_ids: torch.Tensor):
        z = self.encoder(x)
        coin_e = self.coin_emb(coin_ids)  # (B, coin_dim)
        y_pred_levels, y_pred_returns = self.head(z, coin_e, p0)
        S_pred = self.signature(y_pred_levels)
        S_true = self.signature(y_true_levels)
        return {"y_pred_levels": y_pred_levels, "y_pred_returns": y_pred_returns, "S_pred": S_pred, "S_true": S_true}


@dataclass
class MultiCoinConfig(Config):
    num_coins: int = 10
    coin_dim: int = 16


def make_loaders_multicoin(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    coin_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    coin_val: torch.Tensor,
    price_feature_index: Optional[int] = 0,
    p0_train: Optional[torch.Tensor] = None,
    p0_val: Optional[torch.Tensor] = None,
    batch_size: int = 128,
    num_workers: int = 0,
):
    ds_tr = MultiCoinDataset(X_train, Y_train, coin_train, price_feature_index=price_feature_index, p0_tensor=p0_train,
                             standardize_X=True, stats_map=None, compute_stats=True)
    ds_va = MultiCoinDataset(X_val, Y_val, coin_val, price_feature_index=price_feature_index, p0_tensor=p0_val,
                             standardize_X=True, stats_map=ds_tr.stats_map, compute_stats=False)
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr, va, ds_tr.stats_map


def train_multicoin(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    coin_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    coin_val: torch.Tensor,
    price_feature_index: Optional[int] = 0,
    p0_train: Optional[torch.Tensor] = None,
    p0_val: Optional[torch.Tensor] = None,
    cfg: MultiCoinConfig = MultiCoinConfig(),
):
    device = cfg.device
    train_loader, val_loader, stats_map = make_loaders_multicoin(
        X_train, Y_train, coin_train,
        X_val, Y_val, coin_val,
        price_feature_index=price_feature_index,
        p0_train=p0_train, p0_val=p0_val,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )
    model = SigLossTCNMultiCoin(d_in=cfg.d_in, num_coins=cfg.num_coins, coin_dim=cfg.coin_dim,
                                steps=cfg.steps, logsig_depth=cfg.logsig_depth, use_logsig=cfg.use_logsig).to(device)
    criterion = SigPathLoss(lam_path=0.1, lam_end=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    def run_epoch(loader, train_mode: bool):
        model.train(mode=train_mode)
        total_loss, n = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for Xb, Yb, P0b, Cb in loader:
                Xb, Yb, P0b, Cb = Xb.to(device), Yb.to(device), P0b.to(device), Cb.to(device)
                out = model(Xb, Yb, P0b, Cb)
                loss = criterion(out, Yb)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_loss += loss.item() * Xb.size(0)
                n += Xb.size(0)
        return total_loss / max(n, 1)

    best_val, best_state = float('inf'), None
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = run_epoch(train_loader, True)
        va_loss = run_epoch(val_loader, False)
        sched.step()
        print(f"[MultiCoin] Epoch {epoch:3d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, stats_map


# ==========================
# COIN-AGNOSTIC CONDITIONING (no coin IDs available)
# Learn a per-sample "style" embedding from the past 60 min and condition the head.
# This lets the model adapt to coin-specific scale/vol/regime without explicit IDs.
# ==========================

class StyleExtractor(nn.Module):
    """Extracts a per-sample style vector from X: (B,60,d).
    Features include per-channel mean/std over time, last-first delta, and simple autocorr proxies.
    """
    def __init__(self, d_in: int, out_dim: int = 32):
        super().__init__()
        self.d = d_in
        # MLP to compress handcrafted stats → style embedding
        self.mlp = nn.Sequential(
            nn.Linear(d_in * 5, 60), nn.ReLU(), nn.Linear(60, out_dim)
        )

    @staticmethod
    def _autocorr1(x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,d) → crude lag-1 autocorr per channel
        x0 = x[:, :-1, :]
        x1 = x[:, 1:, :]
        num = (x0 * x1).mean(dim=1)
        den = (x0.pow(2).mean(dim=1) * x1.pow(2).mean(dim=1)).sqrt().clamp_min(1e-6)
        return num / den

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, d = X.shape
        mu = X.mean(dim=1)                 # (B,d)
        sd = X.std(dim=1).clamp_min(1e-6)  # (B,d)
        delta = (X[:, -1, :] - X[:, 0, :]) # (B,d)
        ac1 = self._autocorr1(X)           # (B,d)
        # realized vol proxy on first feature returns if available
        ret = X[:, 1:, 0] - X[:, :-1, 0]   # (B,T-1)
        rvol = ret.std(dim=1, keepdim=True)  # (B,1)
        feats = torch.cat([mu, sd, delta, ac1, rvol.repeat(1, d)], dim=-1)  # (B, d*5)
        return self.mlp(feats)             # (B,out_dim)


class SigLossTCNStyleCond(nn.Module):
    """TCN encoder + self-conditioned head using a style vector inferred from X.
    No coin IDs required.
    """
    def __init__(self, d_in: int, steps: int = 10, logsig_depth: int = 3, use_logsig: bool = True, style_dim: int = 32):
        super().__init__()
        self.encoder = EncoderTCN(d_in)
        self.style = StyleExtractor(d_in, out_dim=style_dim)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim + style_dim, 256), nn.ReLU(), nn.Linear(256, steps)
        )
        self.depth = logsig_depth
        self.use_logsig = use_logsig

    def signature(self, path_levels: torch.Tensor) -> torch.Tensor:
        path = time_augment(path_levels)
        path = lead_lag(path)
        path = add_basepoint(path)
        return signatory.logsignature(path, self.depth) if self.use_logsig else signatory.signature(path, self.depth)

    def forward(self, x: torch.Tensor, y_true_levels: torch.Tensor, p0: torch.Tensor):
        z = self.encoder(x)                 # (B,C)
        s = self.style(x)                   # (B,S)
        r_pred = self.head(torch.cat([z, s], dim=-1))  # (B, steps)
        y_pred_levels = (p0 + torch.cumsum(r_pred, dim=1)).unsqueeze(-1)
        S_pred = self.signature(y_pred_levels)
        S_true = self.signature(y_true_levels)
        return {"y_pred_levels": y_pred_levels, "y_pred_returns": r_pred, "S_pred": S_pred, "S_true": S_true}

# Usage example (swap model):
# model = SigLossTCNStyleCond(d_in=cfg.d_in, steps=cfg.steps, logsig_depth=cfg.logsig_depth).to(cfg.device)
