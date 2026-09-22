"""Zero-shot TimesFM-3 wrapper.

The call pattern (TimesFM3Forecaster.predict with past_future_covariates and
return_quantiles) is taken verbatim from TimesFM3-Forecast.ipynb. The model is loaded
once, lazily, and reused.

forecast() returns a plain dict keyed by target column so the context-manager and MPC
code downstream stay model-agnostic - swapping the forecaster means matching this
signature and nothing else.
"""

import os

# Both are needed before numpy/torch import on macOS: numpy and torch each bundle their own
# OpenMP, and inference segfaults on a pthread_mutex_init without them (see the notebook).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd

MODEL = "google/timesfm-3.0-pytorch"
ITEM_ID = "cstr"
TARGET_COLUMNS = ["C_A", "T", "T_C", "h", "Q", "Qc"]
# Every process input, not just the valves: the forecaster is given the same information the
# mathematical model gets, so a like-for-like comparison is possible.
VALVE_COLUMNS = ["valve_cmd", "coolant_valve_cmd"]
DISTURBANCE_COLUMNS = ["E_R", "U_Ac", "T_F", "C_F", "T_CF", "Q_F"]
COVARIATE_COLUMNS = VALVE_COLUMNS + DISTURBANCE_COLUMNS

Z90 = 1.2815515655446004      # standard normal quantile at 0.9, for the interval -> sigma map

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from timesfm3 import TimesFM3Forecaster
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipeline = TimesFM3Forecaster.from_pretrained(MODEL, device=device)
        print(f"Loaded TimesFM 3.0 on {device} (context {_pipeline.global_context})")
    return _pipeline


def load_dataframe(path="data/CSTR_ContextData.h5"):
    """TimesFM-shaped frame (item_id, timestamp, targets, covariates) from the HDF5."""
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        n = f["t"].shape[0]
        data = {k: f[k][:, 0] for k in TARGET_COLUMNS + COVARIATE_COLUMNS}

    df = pd.DataFrame(data)
    df["item_id"] = ITEM_ID
    df["timestamp"] = pd.date_range("2026-01-01", periods=n, freq=f"{int(dt)}s")
    df = df[["item_id", "timestamp"] + TARGET_COLUMNS + COVARIATE_COLUMNS]
    df.attrs["dt"] = dt
    return df


def make_future_df(context_df, horizon, covariates, covariate_columns=COVARIATE_COLUMNS):
    """Known-future covariate frame continuing `context_df`. `covariates` is (horizon, n_cov)."""
    if len(context_df) < 2:
        raise ValueError("context_df needs at least 2 rows to infer the sample interval")
    dt = context_df["timestamp"].iloc[-1] - context_df["timestamp"].iloc[-2]
    start = context_df["timestamp"].iloc[-1] + dt
    future = pd.DataFrame(np.asarray(covariates, dtype=float)[:horizon], columns=covariate_columns)
    future.insert(0, "timestamp", pd.date_range(start, periods=horizon, freq=dt))
    future.insert(0, "item_id", context_df["item_id"].iloc[-1])
    return future


def forecast(context_df, future_df, target_columns, covariate_columns, horizon):
    """Joint zero-shot forecast of `target_columns` over `horizon`.

    Targets are masked over the horizon; covariates are known-future and visible across it,
    which is how MPC gets to ask "what would these valve commands do".

    Returns {column: {"mean", "var", "quantiles": {q: array}}}. "mean" is the median
    (TimesFM is quantile-headed, it has no separate mean head) and "var" comes from the
    10-90 interval under a normal assumption; the raw 9 quantiles are kept untouched for
    the chance-constraint work later.
    """
    pipeline = get_pipeline()

    if len(future_df) < horizon:
        raise ValueError(f"future_df has {len(future_df)} rows, need >= horizon ({horizon})")

    ctx = context_df.iloc[-pipeline.global_context:]
    target = ctx[target_columns].to_numpy(dtype=np.float32).T            # (V, context)

    past_future_cov = None
    if covariate_columns:
        past_future_cov = np.concatenate([                              # (n_cov, context + horizon)
            ctx[covariate_columns].to_numpy(dtype=np.float32),
            future_df[covariate_columns].to_numpy(dtype=np.float32)[:horizon],
        ]).T

    out = pipeline.predict(
        target,
        horizon=horizon,
        past_future_covariates=past_future_cov,
        return_quantiles=True,
    )

    quantile_levels = list(pipeline.config.quantiles)
    i10, i90 = quantile_levels.index(0.1), quantile_levels.index(0.9)

    result = {}
    for v, col in enumerate(target_columns):
        q = np.asarray(out.quantiles[v], dtype=float)                   # (horizon, 9)
        std = (q[:, i90] - q[:, i10]) / (2.0 * Z90)
        result[col] = {
            "mean": np.asarray(out.forecast[v], dtype=float),
            "var": std ** 2,
            "quantiles": {level: q[:, i] for i, level in enumerate(quantile_levels)},
        }
    return result
