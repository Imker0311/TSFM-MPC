"""Zero-shot TimesFM-3 wrapper.

forecast_batch() scores many candidate futures against one shared context in a single batched
pass, returning {column: {mean, var, quantiles}} so the MPC stays model-agnostic.
"""

import os

# Must be set before numpy/torch import: both bundle OpenMP and inference segfaults without.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
MODEL = "google/timesfm-3.0-pytorch"
TARGET_COLUMNS = ["C_A", "T", "T_C", "h", "Q", "Qc"]
VALVE_COLUMNS = ["valve_cmd", "coolant_valve_cmd"]
DISTURBANCE_COLUMNS = ["E_R", "U_Ac", "T_F", "C_F", "T_CF", "Q_F"]
COVARIATE_COLUMNS = VALVE_COLUMNS + DISTURBANCE_COLUMNS

Z90 = 1.2815515655446004      # normal quantile at 0.9, for the 10-90 interval -> sigma map
# ====================================================================

_pipeline = None


def get_pipeline():
    """Load the model once and reuse it."""
    global _pipeline
    if _pipeline is None:
        import torch
        from timesfm3 import TimesFM3Forecaster
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipeline = TimesFM3Forecaster.from_pretrained(MODEL, device=device)
    return _pipeline


def load_dataframe(path="data/CSTR_ContextData.h5"):
    """Read an HDF5 dataset into a model-shaped frame."""
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        n = f["t"].shape[0]
        data = {k: f[k][:, 0] for k in TARGET_COLUMNS + COVARIATE_COLUMNS}

    df = pd.DataFrame(data)
    df["item_id"] = "cstr"
    df["timestamp"] = pd.date_range("2026-01-01", periods=n, freq=f"{int(dt)}s")
    df = df[["item_id", "timestamp"] + TARGET_COLUMNS + COVARIATE_COLUMNS]
    df.attrs["dt"] = dt
    return df


def _pack(mean, quant, target_columns, quantile_levels):
    """Model output -> {column: {mean, var, quantiles}}, leading axes passed through."""
    i10, i90 = quantile_levels.index(0.1), quantile_levels.index(0.9)
    mean, quant = np.asarray(mean, dtype=float), np.asarray(quant, dtype=float)

    result = {}
    for v, col in enumerate(target_columns):
        q = quant[..., v, :, :]
        std = (q[..., i90] - q[..., i10]) / (2.0 * Z90)
        result[col] = {"mean": mean[..., v, :], "var": std ** 2,
                       "quantiles": {lvl: q[..., i] for i, lvl in enumerate(quantile_levels)}}
    return result


def forecast_batch(context_df, future_covariates, target_columns, covariate_columns,
                   horizon, batch_size=None):
    """Score candidate covariate trajectories (n_candidates, >=horizon, n_cov) in one pass."""
    pipeline = get_pipeline()
    fut = np.asarray(future_covariates, dtype=np.float32)
    n, _, n_cov = fut.shape

    ctx = context_df.iloc[-pipeline.global_context:]
    target = ctx[target_columns].to_numpy(dtype=np.float32).T
    past = ctx[covariate_columns].to_numpy(dtype=np.float32).T

    c = past.shape[1]
    past_future_cov = np.empty((n, n_cov, c + horizon), dtype=np.float32)
    past_future_cov[:, :, :c] = past
    past_future_cov[:, :, c:] = fut[:, :horizon, :].transpose(0, 2, 1)

    # per_core_batch_size defaults to 4, which would silently split this into n/4 passes.
    saved = pipeline.config.per_core_batch_size
    pipeline.config.per_core_batch_size = n if batch_size is None else int(batch_size)
    try:
        outs = list(pipeline.predict_batch(contexts=[target] * n, horizon=horizon,
                                           past_future_covariates=list(past_future_cov),
                                           return_quantiles=True))
    finally:
        pipeline.config.per_core_batch_size = saved

    return _pack(np.stack([o.forecast for o in outs]), np.stack([o.quantiles for o in outs]),
                 target_columns, list(pipeline.config.quantiles))
