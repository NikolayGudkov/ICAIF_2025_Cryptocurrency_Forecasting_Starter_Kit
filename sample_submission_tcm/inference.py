import pandas as pd
from pathlib import Path
from model_arima import init_model
import numpy as np
from tqdm import tqdm
import warnings


def predict_x_test(self, x_test: pd.DataFrame) -> pd.DataFrame:
    """
    Returns
    -------
    submission : pd.DataFrame
        Columns: ['window_id','time_step','pred_close']
        dtypes : int32, int8, float32
    """

    out_rows = []
    grouped = x_test.groupby("window_id", sort=False)
    total = x_test["window_id"].nunique()

    for wid, g in tqdm(grouped, total=total):
        g = g.sort_values("time_step")
        # Expect 60 steps per window
        if g["time_step"].nunique() < 60:
            # Skip malformed windows
            continue
        y = g["close"].to_numpy(dtype=np.float64)  # statsmodels expects float64
        # Keep a copy of last value for fallback
        last_close = float(y[-1]) if len(y) > 0 else np.nan

        yhat = self(y, last_close)

        for h in range(len(yhat)):
            out_rows.append({
                "window_id": np.int32(wid),
                "time_step": np.int8(h),
                "pred_close": np.float32(yhat[h]),
            })

    submission = pd.DataFrame(out_rows, columns=["window_id", "time_step", "pred_close"])
    if not submission.empty:
        submission["window_id"] = submission["window_id"].astype("int32")
        submission["time_step"] = submission["time_step"].astype("int8")
        submission["pred_close"] = submission["pred_close"].astype("float32")

        # Basic validation: each window should have 10 steps
        counts = submission.groupby("window_id")["time_step"].nunique()
        if not (counts == 10).all():
            warnings.warn("Some windows did not produce 10 forecast steps.")
    return submission


def generate_forecast(x_test_path: str, out_path: str = "submission.pkl"):
    model = init_model()
    
    x_test = pd.read_pickle(x_test_path)
    submission = model.predict_x_test(x_test)

    out_path = Path(out_path)
    submission.to_pickle(out_path)
    print(f"[OK] Saved forecast to {out_path} with {len(submission)} rows")
    return submission
