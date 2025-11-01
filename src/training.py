from typing import Optional
import numpy as np
import torch
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
from sig_tcm import Config, SigLossTCN, SigPathLoss
from src.data_preparation import make_loaders, data_split


def train(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    LLP_train: Optional[torch.Tensor] = None,
    LLP_val: Optional[torch.Tensor] = None,
    cnf: Optional[Config] = None,
):
    train_loader, val_loader, ds_train_mean, ds_train_std = make_loaders(
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

                log_Yb_pred = model(Xb, log_P0b)
                S_pred = model.signature(log_Yb_pred)
                S_true = model.signature(log_Yb)
                loss = criterion({"log_y_pred_levels": log_Yb_pred, "log_y_true_levels": log_Yb, "S_pred": S_pred, "S_true": S_true})
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
    return model, best_state, ds_train_mean, ds_train_std


# ==========================
# Example synthetic usage
# ==========================
if __name__ == "__main__":
    cnf = Config()

    T_in, forward_steps = cnf.T_in, cnf.steps
    offset = T_in + forward_steps

    # Paths (adjust if your layout differs)
    ROOT = Path.cwd().parent if (Path.cwd().name == 'src') else Path.cwd()
    DATA = ROOT / "data"
    SRC = ROOT / "src"
    SUBM = ROOT / "sample_submission"

    train_path = DATA / "train.parquet"
    weights_path = SUBM / "lstm_weights_test.pkl"

    # Ensure src is importable
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    # Create sample_submission dir if missing
    SUBM.mkdir(parents=True, exist_ok=True)

    SEED = 1337
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_data = pd.read_parquet(train_path)

    train_data = raw_data[raw_data['series_id']<40]
    test_data = raw_data[raw_data['series_id']>=40]

    train_groups = {sid: g.sort_values('time_step').reset_index(drop=True)
                       for sid, g in train_data.groupby('series_id')}

    tr_df, val_df, val_prc = [], [], 0.2
    for g in train_groups.values():
        df_size = g.shape[0] - offset + 1
        val_size = int(val_prc * df_size)
        train_size = df_size - val_size - offset
        tr_df.append(g.iloc[:train_size])
        val_df.append(g.iloc[train_size + offset:])

    tr_df = pd.concat(tr_df, axis=0)
    val_df = pd.concat(val_df, axis=0)

    X_tr, log_Y_tr, LLP_tr = data_split(step_size=10, max_samples=10000000, df=tr_df)
    X_va, log_Y_va, LLP_va = data_split(step_size=10, max_samples=20000, df=val_df)

    _, best_state, ds_train_mean, ds_train_std = train(X_train = X_tr, Y_train=log_Y_tr, X_val = X_va, Y_val = log_Y_va, LLP_train=LLP_tr, LLP_val=LLP_va, cnf=cnf)

    # Save the weights
    obj_to_save = {'best_state': best_state,
                   'ds_train_mean': ds_train_mean,
                   'ds_train_std': ds_train_std}

    torch.save(obj_to_save, weights_path)
