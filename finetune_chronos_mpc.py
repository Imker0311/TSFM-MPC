"""LoRA fine-tune of Chronos-2 on the CSTR, for use as the MPC's internal model.

Both valve commands go in as known-future covariates alongside the six disturbances, because
the valves are what the optimiser chooses between. Context and horizon are set near what the
MPC deploys at rather than at the model's maxima.

    python generate_finetune_data.py
    python finetune_chronos_mpc.py
"""

import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_data_frame

import forecaster as F
import chronos_forecaster as CF

# ============================== CONFIG ==============================
TRAIN_FILE = "data/CSTR_FineTuneTrain.h5"
VAL_FILE = "data/CSTR_FineTuneVal.h5"

PREDICTION_LENGTH = 64      # near the MPC's horizon, not the model's 1024 maximum
CONTEXT_LENGTH = 2048       # near what the MPC deploys at, not the model's 8192 maximum

FINETUNE_MODE = "lora"      # or "full"
LEARNING_RATE = 1e-5
NUM_STEPS = 2000
BATCH_SIZE = 128

OUTPUT_DIR = CF.WEIGHTS_DIR / "chronos-2-mpc-finetuned"        # training checkpoints
SAVE_DIR = CF.FINETUNED_DIR                                    # the one to load
PUSH_TO_HUB = False         # upload SAVE_DIR to CF.HF_REPO after training (needs hf auth login)
# ====================================================================


def build_inputs():
    """Load both datasets as Chronos inputs, with all eight process inputs as covariates."""
    kwargs = dict(target_columns=list(F.TARGET_COLUMNS), prediction_length=PREDICTION_LENGTH,
                  known_covariates_names=list(F.COVARIATE_COLUMNS),
                  id_column="item_id", timestamp_column="timestamp")
    keep = ["item_id", "timestamp"] + list(F.TARGET_COLUMNS) + list(F.COVARIATE_COLUMNS)
    return (from_data_frame(F.load_dataframe(TRAIN_FILE)[keep], **kwargs),
            from_data_frame(F.load_dataframe(VAL_FILE)[keep], **kwargs))


def push():
    """Upload to the Hub. Needs a write token (hf auth login)."""
    from huggingface_hub import HfApi

    repo = CF.HF_REPO
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(SAVE_DIR), repo_id=repo, repo_type="model",
                      commit_message=f"CSTR MPC {FINETUNE_MODE} fine-tune, {NUM_STEPS} steps")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(CF.BASE_MODEL, device_map=device)
    fit_inputs, val_inputs = build_inputs()

    finetuned = pipeline.fit(
        inputs=fit_inputs, prediction_length=PREDICTION_LENGTH, validation_inputs=val_inputs,
        finetune_mode=FINETUNE_MODE, context_length=CONTEXT_LENGTH,
        learning_rate=LEARNING_RATE, num_steps=NUM_STEPS, batch_size=BATCH_SIZE,
        output_dir=str(OUTPUT_DIR), finetuned_ckpt_name="finetuned-ckpt")
    finetuned.save_pretrained(str(SAVE_DIR))

    if PUSH_TO_HUB:
        push()


if __name__ == "__main__":
    main()
