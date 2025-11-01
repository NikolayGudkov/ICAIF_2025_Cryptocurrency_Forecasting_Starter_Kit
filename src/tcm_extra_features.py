"""
Standalone: TCN → 10-step price path with Log-Signature loss
- Input:  X (N, 60, d)  past 60 minutes with d features (may have leading NaNs per feature)
- Target: Y (N, 10)     future price levels (1D path)
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
import sys, warnings
import pandas as pd
from tqdm import tqdm
import pyarrow

# ==========================
# Dataset (prefix-only NaNs with mixed warm-up lengths)
# ==========================
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
        Y_future_levels: torch.Tensor,    # (N, 10) future price levels
        price_feature_index: Optional[int] = None,
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
        assert Y_future_levels.ndim == 2 and Y_future_levels.shape[1] == 10, "Y must be (N, 10)"
        self.X_raw = X.float().clone()
        self.Y = Y_future_levels.float().clone().unsqueeze(-1)  # (N, 10, 1)
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

        if allow_nans:
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
        elif price_feature_index is not None:
            price_col = self.X_raw[:, :, price_feature_index]
            price_mask = ~torch.isnan(price_col)
            p0_list = []
            idxs = torch.arange(60)
            for i in range(self.N):
                if price_mask[i].any():
                    last_idx = idxs[price_mask[i]].max()
                    p0_list.append(price_col[i, last_idx])
                else:
                    raise ValueError("All NaN for price feature in a sample; provide p0_tensor explicitly.")
            self.p0 = torch.stack(p0_list).unsqueeze(-1)
        else:
            raise ValueError("Provide either price_feature_index to extract p0 from X, or p0_tensor explicitly.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.p0[idx]



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
    ds_tr = Seq2FuturePriceDataset(
        X_train, Y_train,
        price_feature_index=price_feature_index, p0_tensor=p0_train,
        allow_nans=True, add_missing_indicators=True,
        prefix_fill="first_valid", long_warmup_linear=True, long_warmup_threshold=20,
        impute_strategy="mean"
    )
    ds_va = Seq2FuturePriceDataset(
        X_val, Y_val,
        price_feature_index=price_feature_index, p0_tensor=p0_val,
        standardize_X=True, mean=ds_tr.mu, std=ds_tr.sigma,  # use train stats
        allow_nans=True, add_missing_indicators=True,
        prefix_fill="first_valid", long_warmup_linear=True, long_warmup_threshold=20,
        impute_strategy="mean"
    )
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr, va, ds_tr.d_out

def train(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    LLP_train: Optional[torch.Tensor] = None,
    LLP_val: Optional[torch.Tensor] = None,
    cnf: Optional[Config] = None,
):
    train_loader, val_loader, _ = make_loaders(
        X_train, Y_train, X_val, Y_val,
        p0_train=LLP_train, p0_val=LLP_val,
        batch_size=cnf.batch_size,
        num_workers=cnf.num_workers
    )

    device = cnf.device
    model = SigLossTCN(cnf = cnf).to(device)
    criterion = SigPathLoss(lam_path=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=cnf.lr, weight_decay=cnf.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cnf.epochs)

    def run_epoch(loader, train_mode: bool):
        model.train(train_mode)
        total_loss, n = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for Xb, log_Yb, log_P0b in loader:
                Xb, log_Yb, log_P0b = Xb.to(device), log_Yb.to(device), log_P0b.to(device)
                out = model(Xb, log_Yb, log_P0b)
                loss = criterion(out)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_loss += loss.item() * Xb.size(0)
                n += Xb.size(0)
        return total_loss / max(n, 1)

    best_val, best_state = float("inf"), None
    for epoch in tqdm(range(1, cnf.epochs + 1)):
        tr_loss = run_epoch(train_loader, True)
        va_loss = run_epoch(val_loader, False)
        sched.step()

        print('\n')
        print(f"Epoch {epoch:3d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best_val:
            best_val, best_state = va_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ==========================
# Example synthetic usage
# ==========================
if __name__ == "__main__":
    T_in, forward_steps = 60, 10
    offset = T_in + forward_steps

    # Paths (adjust if your layout differs)
    ROOT = Path.cwd().parent if (Path.cwd().name == 'src') else Path.cwd()
    DATA = ROOT / "data"
    SRC = ROOT / "src"
    SUBM = ROOT / "sample_submission"

    train_path = DATA / "train.parquet"
    weights_path = SUBM / "lstm_weights_0.pkl"

    # Ensure src is importable
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    # Create sample_submission dir if missing
    SUBM.mkdir(parents=True, exist_ok=True)

    SEED = 1337
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        raw_data = pd.read_parquet('../data/train.parquet')
    except pyarrow.lib.ArrowInvalid:
        raw_data = pd.read_pickle('../data/train.pkl')

    train_data = raw_data[raw_data['series_id']<40]
    test_data = raw_data[raw_data['series_id']>=40]

    MAX_SAMPLES_tr = 10000000
    MAX_SAMPLES_val = 2000

    val_prc = 0.2

    train_groups = {sid: g.sort_values('time_step').reset_index(drop=True)
                       for sid, g in train_data.groupby('series_id')}

    tr_df, val_df = [], []
    for g in train_groups.values():
        #g['event_timestamp'] = pd.date_range(start="2025-01-01", periods=g.shape[0], freq='T')
        df_size = g.shape[0] - offset + 1
        val_size = int(val_prc * df_size)
        train_size = df_size - val_size - offset
        tr_df.append(g.iloc[:train_size])
        val_df.append(g.iloc[train_size + offset:])

    tr_df = pd.concat(tr_df, axis=0)
    val_df = pd.concat(val_df, axis=0)
    val_df = test_data.copy()


    train_samples = WindowsDataset(rolling=True, step_size=10, max_samples=MAX_SAMPLES_tr, df=tr_df)
    val_samples = WindowsDataset(rolling=True, step_size=500, max_samples=MAX_SAMPLES_val, df=val_df)

    #from src.features_compute import build_features_np
    X_tr, Y_tr = train_samples.X, train_samples.y
    X_tr, Y_tr = torch.from_numpy(X_tr), torch.from_numpy(Y_tr)

    X_va, Y_va = val_samples.X, val_samples.y
    X_va, Y_va = torch.from_numpy(X_va), torch.from_numpy(Y_va)

    Y_tr, Y_va = torch.log(Y_tr), torch.log(Y_va)
    LLP_tr, LLP_va = torch.log(torch.from_numpy(train_samples.X[:,-1, 0])).unsqueeze(dim=1), torch.log(torch.from_numpy(val_samples.X[:,-1, 0])).unsqueeze(dim=1)


    cnf = Config(dim_in=2*X_tr.shape[2])  # quick demo
    # model = train(X_train = X_tr, Y_train=Y_tr, X_val = X_va, Y_val = Y_va, LLP_train=LLP_tr, LLP_val=LLP_va, cnf=cnf)
    # torch.save(model.state_dict(), weights_path)
    #
    model = SigLossTCN(cnf).to(DEVICE)
    state_dict = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        train_loader, val_loader, d_in = make_loaders(
            X_tr, Y_tr, X_va, Y_va,
            p0_train=LLP_tr, p0_val=LLP_va,
            batch_size=X_va.shape[0],
            num_workers=cnf.num_workers
        )

        out = []
        for xb, yb, p0b in val_loader:
            out = model(xb, yb, p0b)


    steps_X = np.arange(T_in)
    window_ids = np.arange(X_va.shape[0])
    x_test = pd.concat([pd.DataFrame(data={'window_id': window_ids[i],
                                           'time_step': steps_X,
                                           'close': X_va[i][:, 0],
                                           'volume': X_va[i][:, 1]}) for i in range(X_va.shape[0])], axis=0)


    window_shifts = np.random.randint(50, size=X_va.shape[0])
    steps_y = np.arange(forward_steps)
    event_datetime = pd.date_range(start="2025-01-01", periods=10, freq='T')
    y_local = pd.concat([pd.DataFrame(data={'window_id': window_ids[i],
                                            'time_step': steps_y,
                                            'close': np.exp(yb[i][:, 0]),
                                            'event_datetime': event_datetime + pd.Timedelta(minutes=11 * window_shifts[i]),
                                            'token': 'NAN'}) for i in range(yb.shape[0])], axis=0)

    FIRST_N_WINDOWS = X_va.shape[0]
    all_wids = x_test['window_id'].drop_duplicates().astype('int32').to_numpy()
    # base_sel = all_wids[:int(
    #     FIRST_N_WINDOWS)] if FIRST_N_WINDOWS is not None else all_wids  # you need to run on all windows for official submission

    # must_wids = np.array([1, 2], dtype=np.int32)
    # exist_mask = np.isin(must_wids, all_wids)
    # if not exist_mask.all():
    #     missing = must_wids[~exist_mask].tolist()
    #     warnings.warn(f"[Preview] Required window_id(s) not in x_test: {missing}")
    # sel_wids = np.unique(np.concatenate([base_sel, must_wids[exist_mask]]))
    # print(f"Infer on {len(sel_wids)} / {len(all_wids)} windows "
    #       f"(forced include: {must_wids[exist_mask].tolist()})")

    # # Build a subset view (optional when running preview)
    # x_test_view = x_test[x_test['window_id'].isin(sel_wids)] if FIRST_N_WINDOWS is not None else x_test

    # predict -> submission-like DataFrame # columns: window_id, time_step, pred_close
    submission_df = pd.concat([pd.DataFrame(data={'window_id': window_ids[i],
                                                  'time_step': steps_y,
                                                  'pred_close': np.exp(out['log_y_pred_levels'][i][:, 0]),
                                                  'event_datetime': event_datetime + pd.Timedelta(minutes=11 * window_shifts[i]),
                                                  'token': 'NAN'}) for i in range(yb.shape[0])], axis=0)


    # # validate shape for selected windows
    # if not submission_df.empty:
    #     counts = submission_df.groupby('window_id')['time_step'].nunique()
    #     assert (counts == 10).all(), "Each selected window_id must have exactly 10 rows (0..9)."

    # Save preview (NOT for official submission)
    # For official submission, run inference on ALL windows and save to sample_submission/submission.pkl
    # out_path = SUBM / "submission.pkl"
    # out_path = SUBM / "submission_example.pkl"
    # submission_df.to_pickle(out_path)
    # print(f"Saved preview to {out_path}  rows={len(submission_df)}  "
    #       f"windows={submission_df['window_id'].nunique()}")
    # # display(submission_df.head(12))
    #
    # print("NOTE: This is a PREVIEW subset. For official submission, you must run full inference on ALL windows.")

    from src.metrics import evaluate_all_metrics

    target_wids = np.arange(0, FIRST_N_WINDOWS + 1)  # [1, 2]
    # y_local = pd.read_pickle(y_local_path)     # ground truth: ['window_id','time_step','close']
    pred_local = submission_df[submission_df["window_id"].isin(target_wids)].copy()

    # # Integrity check: each selected window must have exactly 10 prediction steps
    # if not pred_local.empty:
    #     _c = pred_local.groupby("window_id")["time_step"].nunique()
    #     assert (_c == 10).all(), f"Incomplete prediction steps: {_c.to_dict()}"

    # Build x_like from x_test: use time_step == 59 as base_close reference
    x_like_local = (
        x_test[(x_test["window_id"].isin(target_wids)) & (x_test["time_step"] == 59)]
        [["window_id", "time_step", "close"]]
        .copy()
    )

    # Normalize dtypes for consistency
    for df in (y_local, pred_local, x_like_local):
        if "window_id" in df: df["window_id"] = df["window_id"].astype("int32")
        if "time_step" in df: df["time_step"] = df["time_step"].astype("int8")
        if "close" in df: df["close"] = df["close"].astype("float32")
        if "pred_close" in df: df["pred_close"] = df["pred_close"].astype("float32")

    # Keep only ground truth for {1,2}
    y_true_local = y_local[y_local["window_id"].isin(target_wids)].copy()

    # Merge base_close into y_true for trading-based metrics
    base_close_map = x_like_local.set_index("window_id")["close"].astype("float32")
    y_true_with_base = y_true_local.copy()
    y_true_with_base["base_close"] = y_true_with_base["window_id"].map(base_close_map).astype("float32")

    # # Sanity: ensure no missing base_close
    # if y_true_with_base["base_close"].isna().any():
    #     missing_ids = y_true_with_base.loc[y_true_with_base["base_close"].isna(), "window_id"].unique().tolist()
    #     raise ValueError(f"Missing base_close for window_id(s): {missing_ids}")

    # Compute metrics: error metrics + strategy-based (CSM/LOTQ/PW) Sharpe, MDD, VaR, ES
    off_stats = evaluate_all_metrics(
        y_true=y_true_local,
        y_pred=pred_local,
        x_test=x_like_local
    )

    print("\nLocal Eval on window_id 1 & 2")
    print(pd.DataFrame([off_stats]).T.rename(columns={0: "value"}))
    #
    # print("Predicted path shape:", out["log_y_pred_levels"].shape)
    # # print("LogSig dim:", out["S_pred"].shape[-1])
    #
    # # # Inference sanity check
    # # model.eval()
    # # with torch.no_grad():
    # #     ds_va = Seq2FuturePriceDataset(
    # #         X_va, Y_va, price_feature_index=0,
    # #         allow_nans=True, add_missing_indicators=True,
    # #         prefix_fill="first_valid", long_warmup_linear=True, long_warmup_threshold=20,
    # #         impute_strategy="mean"
    # #     )
    # #     xb, yb, p0b = next(iter(DataLoader(ds_va, batch_size=32)))
    # #     out = model(xb, yb, p0b)
    # #     print("Predicted path:", out["y_pred_levels"].shape, "LogSig dim:", out["S_pred"].shape[-1])
