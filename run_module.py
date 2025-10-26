#%% md
# Quickstart

# This notebook will:
#
# 1. Load the datasets: `data/train.pkl`, `data/x_test.pkl` and `data/y_test_local.pkl`.
# 2. Fit per-window ARIMA models on 60-minute input sequences to forecast the following 10 minutes.
# 3. Perform evaluation:
#    - **Local validation** on `y_test_local.pkl` (window_id 1–2)
#    - **Official metrics:** MSE, MAE, IC, IR, Sharpe Ratio, MDD, VaR, ES
#    - **Trading snapshots:** CSM and LOTQ
# 4. Run inference on `x_test` and generate a PICKLE submission file at `sample_submission/submission.pkl`.
# 5. Save dummy `model_weights.pkl` into `sample_submission/` for submission compatibility. *(note: this is only a minimal example; in real deep learning models the weights file would be generated automatically during training.)*
# #%%
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#%%
# Paths (adjust if your layout differs)
ROOT = Path.cwd().parent if (Path.cwd().name == 'src') else Path.cwd()
DATA = ROOT / "data"
SRC  = ROOT / "src"
SUBM = ROOT / "sample_submission"

# Ensure src is importable
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Create sample_submission dir if missing
SUBM.mkdir(parents=True, exist_ok=True)

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE
#%%
# Load dataset files
info_path = DATA / "dataset_info.json"
if info_path.exists():
    info = json.loads(info_path.read_text(encoding="utf-8"))
    print("dataset_info.json loaded. Keys:", list(info.keys()))
    print(json.dumps({k: info[k] for k in ['features','input_len','horizon_len','outputs']}, indent=2))
else:
    print("dataset_info.json not found at", info_path)

# Peek train / x_test
train_path = DATA / "train.parquet"
x_test_path  = DATA / "x_test.pkl"
y_local_path = DATA / "y_test_local.pkl"

train = pd.read_parquet(train_path)
# x_test  = pd.read_pickle(x_test_path)
# y_test_local = pd.read_pickle(y_local_path)

print("train shape:", train.shape, "| columns:", train.columns.tolist())
# print("x_test  shape:", x_test.shape,  "| columns:", x_test.columns.tolist())
# print("y_test_local shape:", y_test_local.shape, "| columns:", y_test_local.columns.tolist())
#
# display(train.head(3))
# display(x_test.head(3))
# display(y_test_local.head(3))
# #%%
# # Use the sampler logic from src/dataset.py to slice windows
from src.dataset import TrainWindowSampler

class WindowsDataset(Dataset):
    """
    Wrap TrainWindowSampler into a PyTorch Dataset.
    Returns:
      X: (60, 2) float32 -> [close, volume]
      y: (10,)  float32 -> future close
    """
    def __init__(self, train_path: str, rolling: bool = True, step_size: int = 1, max_samples: int = None):
        self.sampler = TrainWindowSampler(
            train_path=train_path,
            window=70,
            input_len=60,
            horizon_len=10,
            rolling=rolling,
            step_size=step_size,
            seed=SEED,
        )
        # Materialize (optionally capped) for stable batching
        xs, ys = [], []
        for i, (X, y) in enumerate(self.sampler.iter_windows()):
            xs.append(X.astype(np.float32))
            ys.append(y.astype(np.float32))
            if max_samples is not None and (i + 1) >= max_samples:
                break
        self.X = np.stack(xs, axis=0) if xs else np.zeros((0,60,2), dtype=np.float32)
        self.y = np.stack(ys, axis=0) if ys else np.zeros((0,10), dtype=np.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

# For a quick demo, cap samples. Increase for better quality.
MAX_SAMPLES = 50000  # set to None to use all windows
train_ds = WindowsDataset(str(train_path), rolling=True, step_size=1, max_samples=MAX_SAMPLES)
len(train_ds), train_ds.X.shape, train_ds.y.shape
# #%%
# # ARIMA baseline
from src.baselines.arima import ARIMABaseline

ari = ARIMABaseline(order=(1,1,0), maxiter=50)
ari.fit(train)
#
# Save dummy weights after fitting
import pickle
from pathlib import Path

weights_out = SUBM / "model_weights.pkl"
with open(weights_out, "wb") as f:
    pickle.dump({"config": ari.cfg.__dict__}, f)

print("Saved dummy weights to", weights_out)
#%%
# Fast preview inference on a subset of x_test (NOT for official submission).
# For official submission, run full inference over all windows.

FIRST_N_WINDOWS = 500       # set to an integer (e.g., 500). Set to None to disable.

steps_X = np.arange(train_ds.X.shape[1])
x_test = pd.concat([pd.DataFrame(data = {'window_id': np.full(train_ds.X.shape[1], i),
                                         'time_step': steps_X,
                                         'close': train_ds.X[i][:,0],
                                         'volume': train_ds.X[i][:, 1]}) for i in range(train_ds.X.shape[0])], axis=0)

steps_y = np.arange(train_ds.y.shape[1])
y_local = pd.concat([pd.DataFrame(data = {'window_id': np.full(train_ds.y.shape[1], i),
                                         'time_step': steps_y,
                                         'close': train_ds.y[i]}) for i in range(train_ds.y.shape[0])], axis=0)


all_wids = x_test['window_id'].drop_duplicates().astype('int32').to_numpy()
base_sel = all_wids[:int(FIRST_N_WINDOWS)] if FIRST_N_WINDOWS is not None else all_wids # you need to run on all windows for official submission

must_wids = np.array([1, 2], dtype=np.int32)
exist_mask = np.isin(must_wids, all_wids)
if not exist_mask.all():
    missing = must_wids[~exist_mask].tolist()
    warnings.warn(f"[Preview] Required window_id(s) not in x_test: {missing}")
sel_wids = np.unique(np.concatenate([base_sel, must_wids[exist_mask]]))
print(f"Infer on {len(sel_wids)} / {len(all_wids)} windows "
      f"(forced include: {must_wids[exist_mask].tolist()})")

# Build a subset view (optional when running preview)
x_test_view = x_test[x_test['window_id'].isin(sel_wids)] if FIRST_N_WINDOWS is not None else x_test

# predict -> submission-like DataFrame
submission_df = ari.predict_x_test(x_test_view)   # columns: window_id, time_step, pred_close

# validate shape for selected windows
if not submission_df.empty:
    counts = submission_df.groupby('window_id')['time_step'].nunique()
    assert (counts == 10).all(), "Each selected window_id must have exactly 10 rows (0..9)."

# Save preview (NOT for official submission)
# For official submission, run inference on ALL windows and save to sample_submission/submission.pkl
# out_path = SUBM / "submission.pkl"
out_path = SUBM / "submission_example.pkl"
submission_df.to_pickle(out_path)
print(f"Saved preview to {out_path}  rows={len(submission_df)}  "
      f"windows={submission_df['window_id'].nunique()}")
#display(submission_df.head(12))

print("NOTE: This is a PREVIEW subset. For official submission, you must run full inference on ALL windows.")
#%%
# Local test on window_id {1,2} with y_test_local.pkl
if not y_local_path.exists():
    warnings.warn(f"y_test_local.pkl not found at: {y_local_path}. Skip local eval.")
else:
    # NOTE: updated evaluate_all_metrics expects (y_true, y_pred, x_like, y_true_with_base, horizon_step)
    from src.metrics import evaluate_all_metrics

    target_wids = np.arange(1, FIRST_N_WINDOWS + 1) #[1, 2]
    #y_local = pd.read_pickle(y_local_path)     # ground truth: ['window_id','time_step','close']
    pred_local = submission_df[submission_df["window_id"].isin(target_wids)].copy()

    # Integrity check: each selected window must have exactly 10 prediction steps
    if not pred_local.empty:
        _c = pred_local.groupby("window_id")["time_step"].nunique()
        assert (_c == 10).all(), f"Incomplete prediction steps: {_c.to_dict()}"

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

    # Sanity: ensure no missing base_close
    if y_true_with_base["base_close"].isna().any():
        missing_ids = y_true_with_base.loc[y_true_with_base["base_close"].isna(), "window_id"].unique().tolist()
        raise ValueError(f"Missing base_close for window_id(s): {missing_ids}")

    # Compute metrics: error metrics + strategy-based (CSM/LOTQ/PW) Sharpe, MDD, VaR, ES
    off_stats = evaluate_all_metrics(
        y_true=y_true_local,
        y_pred=pred_local,
        x_test=x_like_local,
        y_true_with_base=y_true_with_base,
        horizon_step=0,
    )

    print("\nLocal Eval on window_id 1 & 2")
    print(pd.DataFrame([off_stats]).T.rename(columns={0: "value"}))