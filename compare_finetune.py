"""Zero-shot vs fine-tuned, against the true plant.

GAIN holds each valve at a clamp extreme and compares the predicted end-of-horizon state
against the true plant, which is what decides whether the MPC can see what its actuators do.
TRACKING is ordinary forecast accuracy on held-out data.

    python compare_finetune.py      # -> data/finetune_comparison.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant
import forecaster as F
import mpc
import chronos_forecaster as CF

# ============================== CONFIG ==============================
OUT_DIR = "data"
VAL_FILE = "data/CSTR_FineTuneVal.h5"       # held out from fine-tuning entirely
CONTEXT_ROWS = 600
HORIZON = 31
N_TRACKING_CASES = 6
SEED = 5
# ====================================================================

K1 = plant.Cv1 * np.sqrt(plant.dP)
K2 = plant.Cv2 * np.sqrt(plant.dPc)
C = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"


def true_response(context_df, valves, horizon):
    """Roll the true plant forward under a held valve pair; the answer the model should give."""
    last = context_df.iloc[-1]
    z = last[F.TARGET_COLUMNS].to_numpy(float)
    y = np.array([z[0], z[1], z[2], z[3], z[4] / K1, z[5] / K2])
    d = last[F.DISTURBANCE_COLUMNS].to_numpy(float)
    for _ in range(horizon):
        y = plant.step(y, np.asarray(valves, float), d, dt=10.0)
    return plant.outputs(y)


def gain_cases(cfg):
    """Each valve at each end of its clamp, the other held at nominal."""
    lo, hi = cfg.lo, cfg.hi
    return {"outlet low": [lo[0], 0.5], "outlet high": [hi[0], 0.5],
            "coolant low": [0.5, lo[1]], "coolant high": [0.5, hi[1]]}


def run_gain(context_df, cases, horizon):
    """Predicted vs true end-of-horizon change, per case."""
    last = context_df.iloc[-1]
    d_hold = last[F.DISTURBANCE_COLUMNS].to_numpy(float)
    fut = np.stack([np.concatenate([np.tile(u, (horizon, 1)),
                                    np.tile(d_hold, (horizon, 1))], axis=1)
                    for u in cases.values()])
    out = CF.forecast_batch(context_df, fut, F.TARGET_COLUMNS, F.COVARIATE_COLUMNS, horizon)

    base_T, base_h = float(last["T"]), float(last["h"])
    rows = []
    for i, (name, u) in enumerate(cases.items()):
        z = true_response(context_df, u, horizon)
        rows.append(dict(case=name,
                         model_dT=out["T"]["mean"][i, -1] - base_T, true_dT=z[1] - base_T,
                         model_dh=out["h"]["mean"][i, -1] - base_h, true_dh=z[3] - base_h))
    return rows


def run_tracking(df, horizon, n_cases, rng):
    """Forecast accuracy on held-out data, given the true future covariates."""
    errs = {c: [] for c in F.TARGET_COLUMNS}
    for s in rng.integers(CONTEXT_ROWS, len(df) - horizon - 1, size=n_cases):
        ctx, fut = df.iloc[s - CONTEXT_ROWS:s], df.iloc[s:s + horizon]
        out = CF.forecast_batch(ctx, fut[F.COVARIATE_COLUMNS].to_numpy(float)[None],
                                F.TARGET_COLUMNS, F.COVARIATE_COLUMNS, horizon)
        for c in F.TARGET_COLUMNS:
            errs[c].append(out[c]["mean"][0] - fut[c].to_numpy(float))
    return {c: float(np.sqrt(np.mean(np.square(errs[c])))) / float(df[c].std())
            for c in F.TARGET_COLUMNS}


def evaluate(label, finetuned, df_val, cases):
    """Load one checkpoint and run both tests against it."""
    CF.reset_pipeline()
    CF.get_pipeline(finetuned=finetuned)
    return dict(label=label,
                gain=run_gain(df_val.iloc[-CONTEXT_ROWS:], cases, HORIZON),
                tracking=run_tracking(df_val, HORIZON, N_TRACKING_CASES,
                                      np.random.default_rng(SEED)))


def report(results):
    """Print the gain table and the held-out accuracy table."""
    for key, unit in [("dT", "K"), ("dh", "m")]:
        print(f"\n{'case':14s} {'TRUE ' + key:>10}" +
              "".join(f"{r['label']:>14s}" for r in results))
        for i, g in enumerate(results[0]["gain"]):
            line = f"{g['case']:14s} {g['true_' + key]:>+10.4f}"
            line += "".join(f"{r['gain'][i]['model_' + key]:>+14.4f}" for r in results)
            print(line)
    for r in results:
        eT = np.mean([abs(g["model_dT"] - g["true_dT"]) for g in r["gain"]])
        eh = np.mean([abs(g["model_dh"] - g["true_dh"]) for g in r["gain"]])
        print(f"\n{r['label']:12s} mean |gain error|: T {eT:.3f} K   h {eh:.4f} m")

    print(f"\n{'model':14s}" + "".join(f"{c:>9s}" for c in F.TARGET_COLUMNS))
    for r in results:
        print(f"{r['label']:14s}" + "".join(f"{r['tracking'][c]:9.4f}" for c in F.TARGET_COLUMNS))


def figure(results):
    """Bar chart of predicted vs true valve gains, temperature and level."""
    cases = [g["case"] for g in results[0]["gain"]]
    x = np.arange(len(cases))
    width = 0.8 / (len(results) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    fig.patch.set_facecolor("white")
    for ax, key, unit in [(axes[0], "dT", "K"), (axes[1], "dh", "m")]:
        ax.set_facecolor(SURFACE)
        ax.set_ylabel(f"end-of-horizon change  [{unit}]", color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.bar(x, [g[f"true_{key}"] for g in results[0]["gain"]], width,
               color=INK, label="true plant")
        for j, r in enumerate(results):
            ax.bar(x + (j + 1) * width, [g[f"model_{key}"] for g in r["gain"]], width,
                   color=C[j % len(C)], label=r["label"])
        ax.set_xticks(x + width * len(results) / 2)
        ax.set_xticklabels(cases, fontsize=8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_title("Does the model know what the valves do? (temperature)", fontsize=10)
    axes[1].set_title("...and the level", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "finetune_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    df_val = F.load_dataframe(VAL_FILE)
    cases = gain_cases(mpc.Config(F.load_dataframe()))
    results = [evaluate("zero-shot", False, df_val, cases),
               evaluate("fine-tuned", True, df_val, cases)]
    report(results)
    figure(results)
    return results


if __name__ == "__main__":
    main()
