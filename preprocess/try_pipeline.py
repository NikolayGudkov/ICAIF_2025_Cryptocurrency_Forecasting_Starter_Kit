"""
This file is building a simple integration test for a feature preprocessing pipeline.

It defines a custom feature adder that adds noise and computes VWAP, then combines it
with a standard scaler in a pipeline. The test checks that the pipeline's output matches
the expected results and verifies the inverse transformation.
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.testing import assert_allclose

from models.preprocess import ColumnWiseTransformer, FeaturePipeline


# -------------------------------
#  Define transforms
# -------------------------------
class CustomFeatureAdder:
    """Adds noise to 'close', adds 1 to 'volume', and computes VWAP."""
    def __init__(self, noise_std: float = 0.1, seed: int = 42):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.columns_ = ["close", "volume", "vwap"]

    def __call__(self, X: np.ndarray) -> pd.DataFrame:
        # Expect columns: close, volume
        close, volume = X[:, 0], X[:, 1]

        # Apply deterministic noise
        noisy_close = close + 3.0 + self.rng.normal(0, self.noise_std, size=len(close))
        shifted_volume = volume + 1.0

        # VWAP: cumulative(price * volume) / cumulative(volume)
        vwap = np.cumsum(noisy_close * shifted_volume) / np.cumsum(shifted_volume)

        df = pd.DataFrame({
            "close": noisy_close,
            "volume": shifted_volume,
            "vwap": vwap,
        })
        return df


# -------------------------------
#  Integration Test
# -------------------------------
def test_feature_pipeline_integration():
    np.random.seed(0)

    # 1) Artificial dataset: close & volume
    n = 10
    X_raw = np.stack(
        [np.linspace(10, 20, n), np.linspace(100, 200, n)], axis=1
    ).astype(np.float32)

    # 2) Build the pipeline
    adder = CustomFeatureAdder(noise_std=0.0, seed=123)  # fix seed, no noise for exact test
    col_tf = ColumnWiseTransformer({"vwap": StandardScaler()})
    pipeline = FeaturePipeline([adder, col_tf])

    # 3) Apply pipeline
    X_trans = pipeline(X_raw)

    # 4) Manual expected computation
    close = X_raw[:, 0] + 3.0
    volume = X_raw[:, 1] + 1.0
    vwap = np.cumsum(close * volume) / np.cumsum(volume)
    vwap_scaled = (vwap - vwap.mean()) / vwap.std()
    expected = np.stack([close, volume, vwap_scaled], axis=1)

    # 5) Compare
    assert_allclose(X_trans, expected, rtol=1e-6, atol=1e-6)

    # 6) Test inverse_transform
    X_inv = pipeline.inverse_transform(X_trans)
    assert_allclose(X_inv["vwap"].to_numpy(), vwap, rtol=1e-6, atol=1e-6)

    print("Integration test passed — pipeline transformation and inversion are correct.")


if __name__ == "__main__":
    test_feature_pipeline_integration()

