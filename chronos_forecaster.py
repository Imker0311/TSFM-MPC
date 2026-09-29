"""Chronos-2 as the MPC's internal model, drop-in for forecaster.py.

install() swaps the forecast function onto forecaster, so mpc.py and run_mpc.py are unchanged.

Chronos packs covariates in sorted(names) order, which puts the two valve commands LAST, not
first as everywhere else in this repo.
"""

import os
from pathlib import Path

import numpy as np
import torch

import forecaster as F

# ============================== CONFIG ==============================
BASE_MODEL = "amazon/chronos-2"

# Fine-tuned weights, kept next to the Chronos-2 checkpoints in the sibling
# Masters-Forecasting repo. Relative to this file so the folders can be moved; set
# CHRONOS_MPC_WEIGHTS to override.
WEIGHTS_DIR = Path(os.environ.get(
    "CHRONOS_MPC_WEIGHTS", Path(__file__).resolve().parent.parent / "Masters-Forecasting"))
FINETUNED_DIR = WEIGHTS_DIR / "chronos-2-mpc-finetuned-final"

HF_REPO_ENV = "CHRONOS_MPC_FT_REPO"     # Hub fallback when the local folder is missing
BATCH_SIZE = 256                        # >= n_candidates keeps a solve to one pass
# ====================================================================

TARGET_COLUMNS = F.TARGET_COLUMNS
COVARIATE_COLUMNS = F.COVARIATE_COLUMNS
VALVE_COLUMNS = F.VALVE_COLUMNS
DISTURBANCE_COLUMNS = F.DISTURBANCE_COLUMNS
load_dataframe = F.load_dataframe

_pipeline = None


def resolve_checkpoint(finetuned=True):
    """Local fine-tuned folder, else the Hub, else the zero-shot base model."""
    if not finetuned:
        return BASE_MODEL
    if FINETUNED_DIR.is_dir():
        return str(FINETUNED_DIR)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=os.environ[HF_REPO_ENV], repo_type="model")


def get_pipeline(finetuned=True):
    """Load the model once and reuse it."""
    global _pipeline
    if _pipeline is None:
        from chronos import Chronos2Pipeline
        ckpt = resolve_checkpoint(finetuned)
        kwargs = dict(device_map="cuda" if torch.cuda.is_available() else "cpu")
        # peft needs the base module named explicitly when loading a LoRA adapter.
        if (Path(ckpt) / "adapter_config.json").exists():
            kwargs["import_allowlist"] = ["chronos.chronos2.model"]
        _pipeline = Chronos2Pipeline.from_pretrained(ckpt, **kwargs)
    return _pipeline


def reset_pipeline():
    """Drop the cached pipeline so a different checkpoint can be loaded in one process."""
    global _pipeline
    _pipeline = None


def _template(context_df, covariate_columns, target_columns, horizon):
    """Build one input holding the shared context, so it is packed once for all candidates."""
    from chronos.chronos2.preprocess import from_list_of_dicts

    ctx = context_df.iloc[-get_pipeline().model_context_length:]
    return from_list_of_dicts([{
        "target": ctx[list(target_columns)].to_numpy(np.float32).T,
        "past_covariates": {c: ctx[c].to_numpy(np.float32) for c in covariate_columns},
        "future_covariates": {c: np.full(horizon, float(ctx[c].iloc[-1]), np.float32)
                              for c in covariate_columns},
    }], prediction_length=horizon)[0]


def _pack(quant, target_columns, quantile_levels):
    """Model output -> {column: {mean, var, quantiles}}; mean is the median."""
    i10, i50, i90 = (quantile_levels.index(q) for q in (0.1, 0.5, 0.9))
    result = {}
    for v, col in enumerate(target_columns):
        q = quant[:, v]
        std = (q[:, i90] - q[:, i10]) / (2.0 * F.Z90)
        result[col] = {"mean": q[:, i50], "var": std ** 2,
                       "quantiles": {lvl: q[:, i] for i, lvl in enumerate(quantile_levels)}}
    return result


def forecast_batch(context_df, future_covariates, target_columns, covariate_columns,
                   horizon, batch_size=None):
    """Score candidate covariate trajectories (n_candidates, >=horizon, n_cov) in one pass."""
    pipeline = get_pipeline()
    fut = np.asarray(future_covariates, dtype=np.float32)
    n = len(fut)

    template = _template(context_df, covariate_columns, target_columns, horizon)
    rows = {c: template["n_targets"] + i for i, c in enumerate(sorted(covariate_columns))}

    stacked = template["future_covariates"].unsqueeze(0).repeat(n, 1, 1)
    stacked[:, [rows[c] for c in covariate_columns], :] = torch.from_numpy(
        fut[:, :horizon, :].transpose(0, 2, 1))

    inputs = [dict(template, future_covariates=stacked[i]) for i in range(n)]
    out = pipeline.predict(inputs, prediction_length=horizon,
                           batch_size=int(batch_size or BATCH_SIZE))
    return _pack(np.stack([o.detach().cpu().numpy() for o in out]),
                 target_columns, list(pipeline.quantiles))


def install(finetuned=True):
    """Point mpc.py at Chronos-2 without editing it."""
    get_pipeline(finetuned)
    F.forecast_batch = forecast_batch
