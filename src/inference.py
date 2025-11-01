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
                                              'event_datetime': event_datetime + pd.Timedelta(
                                                  minutes=11 * window_shifts[i]),
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
