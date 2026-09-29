"""The MPC loop with Chronos-2 as the internal model instead of TimesFM-3.

chronos_forecaster.install() swaps the forecast function, so mpc.py and run_mpc.py are
unchanged and the two models are directly comparable.

    python run_mpc_chronos.py               # fine-tuned weights
    python run_mpc_chronos.py zeroshot      # zero-shot, for the before/after
"""

import sys

import forecaster as F
import mpc
import run_mpc
import chronos_forecaster as CF

# ============================== CONFIG ==============================
# Real-time-feasible settings; cost per step is linear in context length and candidate count.
PROTECTED_ROWS = 300
MAX_CONTEXT = 600
PRUNE_TO = 480
N_CANDIDATES = 100
N_ELITES = 10
N_ITER = 3
N_STEPS = 450
# ====================================================================


def main(finetuned=True):
    run_mpc.PROTECTED_ROWS = PROTECTED_ROWS
    run_mpc.MAX_CONTEXT = MAX_CONTEXT
    run_mpc.PRUNE_TO = PRUNE_TO

    CF.install(finetuned=finetuned)
    cfg = mpc.Config(F.load_dataframe(), n_candidates=N_CANDIDATES, n_elites=N_ELITES,
                     n_iter=N_ITER)
    label = f"Chronos-2 ({'fine-tuned' if finetuned else 'zero-shot'})"
    return run_mpc.run(cfg=cfg, n_steps=N_STEPS,
                       tag="mpc_chronos" + ("" if finetuned else "_zeroshot"), model=label)


if __name__ == "__main__":
    main(finetuned="zeroshot" not in sys.argv)
