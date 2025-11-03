from typing import Optional
import numpy as np
import torch
from utils.dataset import WindowsDataset
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import pyarrow
from sig_tcm import Config, SigLossTCN, SigPathLoss
from src.data_preparation import make_loaders, data_split, Seq2FuturePriceDataset
import pickle
from torch.utils.data import Dataset, DataLoader
from inference import generate_forecast
from sig_tcm import init_model
from inference import predict_x_test
from src.features_generation import build_features_np

if __name__ == "__main__":
    cnf = Config()
    T_in, forward_steps = cnf.T_in, cnf.steps
    offset = T_in + forward_steps

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths (adjust if your layout differs)
    ROOT = Path.cwd().parent if (Path.cwd().name == 'src') else Path.cwd()
    DATA = ROOT / "data"
    SRC = ROOT / "src"
    SUBM = ROOT / "sample_submission"
    weights_path = SUBM / "model_weights_corr.pkl"
    state_dict = torch.load(weights_path, map_location="cpu")

    # Load data
    x_test = pd.read_parquet(DATA / 'x_test.parquet')
    y_test_local = pd.read_parquet(DATA / 'y_test_local.parquet')

    model = init_model(weights_path)

    submission = predict_x_test(model, x_test, feature_generation=build_features_np)

    target_wids = np.arange(10)
    x_like_local = (
        x_test[(x_test["window_id"].isin(target_wids)) & (x_test["time_step"] == 59)]
        [["window_id", "time_step", "close"]]
        .copy()
    )

    pred_local = submission[submission["window_id"].isin(target_wids)].copy()

    # Merge base_close into y_true for trading-based metrics
    base_close_map = x_like_local.set_index("window_id")["close"].astype("float32")
    y_true_with_base = y_test_local.copy()
    y_true_with_base["base_close"] = y_true_with_base["window_id"].map(base_close_map).astype("float32")

    # # Sanity: ensure no missing base_close
    # if y_true_with_base["base_close"].isna().any():
    #     missing_ids = y_true_with_base.loc[y_true_with_base["base_close"].isna(), "window_id"].unique().tolist()
    #     raise ValueError(f"Missing base_close for window_id(s): {missing_ids}")
    pred_local['event_datetime'] = y_test_local['event_datetime']
    pred_local['token'] = y_test_local['token']
    from src.metrics import evaluate_all_metrics
    # Compute metrics: error metrics + strategy-based (CSM/LOTQ/PW) Sharpe, MDD, VaR, ES
    off_stats = evaluate_all_metrics(
        y_true=y_test_local,
        y_pred=pred_local,
        x_test=x_like_local
    )

    print(pd.DataFrame([off_stats]).T.rename(columns={0: "value"}))

    submission.to_pickle('submission.pkl')