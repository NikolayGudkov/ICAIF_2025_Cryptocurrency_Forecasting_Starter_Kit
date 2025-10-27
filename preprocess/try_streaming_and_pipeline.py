"""
Extended integration test for FeatureDataset + FeaturePipeline.

Tests:
1) Streaming single samples from train.pkl
2) Streaming in batches
3) Verifying feature expansion and scaling correctness
4) Verifying inverse_transform restores pre-scaled columns
"""
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from preprocess import (
    BasicFeatureGenerator,
    ColumnWiseTransformer,
    FeatureDataset,
    FeatureIterableDataset,
    FeaturePipeline,
)


def verify_standardization(X_df: pd.DataFrame, cols: list[str], tol: float = 1e-3):
    """Check that specified columns are standardized."""
    for c in cols:
        m, s = X_df[c].mean(), X_df[c].std(ddof=0)  # ddof=0 for population std since StandardScaler uses it
        assert abs(m) < tol, f"{c} mean not ≈ 0 (got {m:.5f})"
        assert abs(s - 1.0) < tol, f"{c} std not ≈ 1 (got {s:.5f})"


def test_streaming_and_inverse(train_path: str | Path, n_samples: int = 3, batch_size: int = 4):
    """
    Integration test for:
    - FeatureDataset streaming from train.pkl
    - Batch loading
    - Column-wise scaling
    - Inverse transformation correctness
    """

    print("🚀 Starting integration test: streaming, batching, and inverse_transform")

    # 1) Define the feature pipeline
    feature_gen = BasicFeatureGenerator(vol_window=5)
    col_transformer = ColumnWiseTransformer({
        "log_ret": StandardScaler(),
        "vol_delta": StandardScaler(),
    })
    pipeline = FeaturePipeline([feature_gen, col_transformer])

    # 2) Initialize dataset and DataLoader
    dataset = FeatureDataset(data_path=train_path, pipeline=pipeline)
    loader_single = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    loader_batch = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3) Stream and test first few single samples
    print(f"\n📦 Streaming first {n_samples} samples (batch_size=1)...\n")
    for i, (X, y) in enumerate(loader_single):
        if i >= n_samples:
            break

        X_np = X[0].numpy()
        feature_names = feature_gen.columns_
        X_df = pd.DataFrame(X_np, columns=feature_names)

        print(f"Sample {i+1} - X shape: {X_np.shape}, y shape: {y.shape}")
        print(X_df.head(3))
        verify_standardization(X_df, ["log_ret", "vol_delta"])

    print("\n✅ Single-sample streaming passed standardization checks.")

    # 4) Test batched streaming
    print(f"\n📦 Streaming in batches (batch_size={batch_size})...\n")
    for i, (X, y) in enumerate(loader_batch):
        X_np = X.numpy()
        b, t, f = X_np.shape
        assert f == 5, f"Expected 5 features, got {f}"
        assert b <= batch_size, "Batch size mismatch"
        print(f"Batch {i+1}: shape={X_np.shape}")
        if i >= 2:
            break

    print("\n✅ Batch streaming verified successfully.")

    # 5) Inverse transform check on a single streamed sample
    print("\n🔄 Checking inverse_transform consistency...")

    X_orig = next(iter(loader_single))[0][0].numpy()
    X_inv_df = pipeline.inverse_transform(X_orig)
    assert isinstance(X_inv_df, pd.DataFrame), "inverse_transform should return DataFrame"
    print("✅ inverse_transform returned DataFrame successfully.")


def test_inverse_transform_against_baseline(train_path: str | Path, sample_index: int = 0):
    # 1) Build pipeline
    feature_gen = BasicFeatureGenerator(vol_window=5)
    col_tf = ColumnWiseTransformer({
        "log_ret": StandardScaler(),
        "vol_delta": StandardScaler(),
    })
    pipeline1 = FeaturePipeline([feature_gen, col_tf])
    pipeline2 = FeaturePipeline([feature_gen])  # baseline (no scaling)

    # 2) Two datasets pointing to the same file:
    # - ds_std: full pipeline (feature_gen + scaler) -> standardized features
    # - ds_base: only feature_gen (no scaler) -> baseline pre-scaled features
    ds_std  = FeatureDataset(data_path=train_path, pipeline=pipeline1,   materialize=True)
    ds_base = FeatureDataset(data_path=train_path, pipeline=pipeline2, materialize=True)

    # 3) Grab the same window from both datasets
    X_std_t, _ = ds_std[sample_index]  # standardized (numpy after .numpy() below)
    X_base_t, _ = ds_base[sample_index]  # baseline (feature_gen only)

    # Convert to numpy and DataFrame with named columns
    cols = feature_gen.columns_  # ['close','volume','log_ret','vol_delta','rolling_vol']
    X_std_df  = pd.DataFrame(X_std_t.numpy(), columns=cols)
    X_base_df = pd.DataFrame(X_base_t.numpy(), columns=cols)

    # 4) Verify the standardized columns are indeed standardized
    verify_standardization(X_std_df, ["log_ret", "vol_delta"])

    # 5) Inverse-transform the standardized features back to pre-scaled space
    X_inv_df = pipeline1.inverse_transform(X_std_df.to_numpy(np.float32))

    # 6) Compare inverse-transform result to baseline (feature_gen only)
    for c in ["log_ret", "vol_delta"]:
        np.testing.assert_allclose(
            X_inv_df[c].to_numpy()[1:],
            X_base_df[c].to_numpy()[1:],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{c} inverse_transform mismatch vs baseline",
        )
    # Sanity check: unchanged columns should remain equal up to numerical noise
    for c in ["close", "volume", "rolling_vol"]:
        # If your ColumnWiseTransformer doesn’t touch these columns,
        # inverse_transform should leave them equal to baseline as well.
        np.testing.assert_allclose(
            X_inv_df[c].to_numpy(),
            X_base_df[c].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{c} unexpectedly changed by inverse_transform",
        )
    print("✅ inverse_transform matches baseline (feature_gen-only) for scaled and untouched columns.")


def test_feature_iterable_dataset_streaming(train_path: str | Path, n_batches: int = 3, batch_size: int = 1):
    print("🚀 Starting integration test for FeatureIterableDataset (streaming mode)")

    # 1) Build pipeline
    feature_gen = BasicFeatureGenerator(vol_window=5)
    col_tf = ColumnWiseTransformer({
        "log_ret": StandardScaler(),
        "vol_delta": StandardScaler(),
    })
    pipeline = FeaturePipeline([feature_gen, col_tf])

    # 2) Streaming dataset (iterable)
    ds_stream = FeatureIterableDataset(data_path=train_path, pipeline=pipeline)
    loader = DataLoader(ds_stream, batch_size=batch_size, num_workers=0)

    # 3) Baseline (materialized) dataset with only feature generation
    ds_base = FeatureDataset(data_path=train_path, pipeline=feature_gen, materialize=True)

    # 4) Stream a few batches and test transformations
    for i, (X_batch, y_batch) in enumerate(loader):
        if i >= n_batches:
            break

        print(f"\n📦 Batch {i+1}: X={tuple(X_batch.shape)}, y={tuple(y_batch.shape)}")

        # Convert first element of the batch to dataframe
        X_np = X_batch[0].numpy()
        feature_names = feature_gen.columns_
        X_df = pd.DataFrame(X_np, columns=feature_names)

        # Check that scaling happened
        verify_standardization(X_df, ["log_ret", "vol_delta"])

        # 5) Inverse-transform check
        X_inv_df = pipeline.inverse_transform(X_df.to_numpy(np.float32))
        X_base_df = pd.DataFrame(ds_base[i][0].numpy(), columns=feature_names)

        # Compare scaled columns after inverse-transform with baseline
        for c in ["log_ret", "vol_delta"]:
            np.testing.assert_allclose(
                X_inv_df[c].to_numpy()[1:],
                X_base_df[c].to_numpy()[1:],
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{c} mismatch after inverse_transform()"
            )

        print(f"Batch {i+1}: inverse_transform matches baseline for scaled columns.")

    print("\n✅ FeatureIterableDataset streaming pipeline test passed successfully!")



if __name__ == "__main__":
    # Get project root
    root = Path(__file__).parent.parent

    # Update this path to match your environment
    train_path = root / "data/train.pkl"
    test_streaming_and_inverse(train_path)
    test_inverse_transform_against_baseline(train_path, sample_index=0)
    test_feature_iterable_dataset_streaming(train_path)
    print("\n🎯 All integration checks passed successfully!\n")
