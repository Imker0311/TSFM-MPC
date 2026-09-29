"""Excitation datasets for fine-tuning: one to train on, one to validate on.

Two independent runs with different seeds rather than one run split in two, so nothing that
happens only once per run is stranded on one side of a split.

    python generate_finetune_data.py     # -> data/CSTR_FineTune{Train,Val}.h5
"""

import copy

import generate_context_data as G

# ============================== CONFIG ==============================
SEED_TRAIN = 101
# Tried in order, first survivor wins: some seeds stack disturbances hard enough to
# extinguish the reactor, and the generator refuses to write that.
SEED_VAL_CANDIDATES = [202, 203, 204, 205, 206, 207]

# Row count and training step budget have to scale together: at a fixed step budget a bigger
# dataset just means each window is seen less often.
TRAIN_DURATION = 2_000_000.0    # s -> 200,000 rows at the 10 s sample interval
VAL_DURATION = 400_000.0        # s ->  40,000 rows

EVENTS_PER_100K_S = 6           # matches the context dataset's dynamics density

FILE_TRAIN = "CSTR_FineTuneTrain.h5"
FILE_VAL = "CSTR_FineTuneVal.h5"
# ====================================================================


def generate(seed, duration, filename):
    """Run generate_context_data with these settings applied to its config, then restored."""
    saved = {k: getattr(G, k) for k in
             ("SEED", "TOTAL_DURATION", "FILE_DATA", "FILE_EVENTS", "FAULTS")}
    try:
        G.SEED, G.TOTAL_DURATION, G.FILE_DATA = seed, duration, filename
        G.FILE_EVENTS = filename.replace(".h5", "_Events.h5")
        G.FAULTS = copy.deepcopy(saved["FAULTS"])
        for cfg in G.FAULTS.values():
            cfg["count"] = max(1, round(EVENTS_PER_100K_S * duration / 100_000.0))
        G.main()
    finally:
        for k, v in saved.items():
            setattr(G, k, v)


def generate_first_surviving(seeds, duration, filename):
    """Try each seed until one produces a run that never extinguishes."""
    for seed in seeds:
        try:
            generate(seed, duration, filename)
            return seed
        except RuntimeError:
            continue
    raise RuntimeError(f"every seed in {seeds} extinguished the reactor")


def main():
    generate(SEED_TRAIN, TRAIN_DURATION, FILE_TRAIN)
    generate_first_surviving(SEED_VAL_CANDIDATES, VAL_DURATION, FILE_VAL)


if __name__ == "__main__":
    main()
