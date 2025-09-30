"""
Minimal inference.py for submission.
- Must expose generate_forecast(x_test_path) that outputs submission.pkl.
"""

import pandas as pd
from pathlib import Path
from model import init_model


def generate_forecast(x_test_path: str, out_path: str = "submission.pkl"):
    model = init_model()
    
    x_test = pd.read_pickle(x_test_path)
    submission = model.predict_x_test(x_test)

    out_path = Path(out_path)
    submission.to_pickle(out_path)
    print(f"[OK] Saved forecast to {out_path} with {len(submission)} rows")
    return submission
