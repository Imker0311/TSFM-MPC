"""Does TimesFM-3 predict the CSTR's response to a valve step?

Straight comparison, no differencing. Pick a row late in the dataset where a valve moves,
give the model everything before it plus the true future inputs, and plot what it predicts
for the 6 states against what the real CSTR did under exactly those inputs.

The model gets ALL process inputs as covariates - both valve commands and all six
disturbances - so it has the same information the mathematical model has. Disturbances are
free to be active at the same time as the valve step; that is part of the test.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forecaster as F

# ============================== CONFIG ==============================
HORIZON = 60            # rows to predict = 10 min at 10 s
LEAD_IN = 90            # rows of history drawn on the plot (display only)
N_CASES = 4             # how many valve steps to evaluate
MIN_ROW = 6000          # only look this far in, so the model always has plenty of context
MIN_STEP = 0.01         # what counts as a valve step, in valve position
OUT_DIR = "data"
# ====================================================================

STATES = F.TARGET_COLUMNS
VALVES = F.VALVE_COLUMNS

C_TRUE, C_PRED, C_BAND = "#0b0b0b", "#2a78d6", "#2a78d6"
MUTED, GRID, SURFACE, INK = "#52514e", "#dcdbd6", "#fcfcfb", "#0b0b0b"


def find_valve_steps(df):
    """Rows where a valve command moves, far enough in and far enough apart."""
    found = []
    for name in VALVES:
        v = df[name].to_numpy()
        moves = np.where(np.abs(np.diff(v)) > MIN_STEP)[0] + 1
        for r in moves:
            if r < MIN_ROW or r + HORIZON >= len(df):
                continue
            if any(abs(r - f["row"]) < HORIZON for f in found):
                continue
            found.append(dict(row=r, valve=name, size=v[r] - v[r - 1]))
    found.sort(key=lambda f: abs(f["size"]), reverse=True)
    return found


def run_case(df, case):
    """Context = everything before the step. Future covariates = the true inputs."""
    r = case["row"]
    context_df = df.iloc[:r]
    future_df = df.iloc[r:r + HORIZON]
    out = F.forecast(context_df, future_df, STATES, F.COVARIATE_COLUMNS, HORIZON)
    truth = future_df[STATES].to_numpy(dtype=float)
    return out, truth, context_df, future_df


def _ax(ax, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plot_case(df, case, out, truth, context_df, future_df, dt, idx):
    r = case["row"]
    lead = context_df.iloc[-LEAD_IN:]
    t_lead = (np.arange(-len(lead), 0)) * dt / 60.0
    t_fut = np.arange(HORIZON) * dt / 60.0

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), facecolor=SURFACE)
    axes = axes.ravel()

    ax = axes[0]
    for name, c in zip(VALVES, ["#2a78d6", "#eb6834"]):
        ax.plot(t_lead, lead[name], color=c, lw=1.4)
        ax.plot(t_fut, future_df[name], color=c, lw=1.4, label=name)
    ax.axvline(0, color=MUTED, ls=":", lw=1)
    _ax(ax, "valve position [-]")
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="best")

    ax = axes[1]
    for name, c in zip(F.DISTURBANCE_COLUMNS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]):
        base = np.median(df[name])
        dev = 100.0 * (np.concatenate([lead[name], future_df[name]]) - base) / abs(base)
        ax.plot(np.concatenate([t_lead, t_fut]), dev, color=c, lw=1.2, label=name)
    ax.axvline(0, color=MUTED, ls=":", lw=1)
    _ax(ax, "disturbances [% dev]")
    ax.legend(fontsize=7, frameon=False, ncol=3, labelcolor=MUTED, loc="best")

    for k, s in enumerate(STATES):
        ax = axes[2 + k]
        ax.plot(t_lead, lead[s], color=C_TRUE, lw=1.3, alpha=0.5)
        ax.plot(t_fut, truth[:, k], color=C_TRUE, lw=1.8, label="true CSTR")
        ax.plot(t_fut, out[s]["mean"], color=C_PRED, lw=1.6, ls="--", label="TimesFM-3")
        ax.fill_between(t_fut, out[s]["quantiles"][0.1], out[s]["quantiles"][0.9],
                        color=C_BAND, alpha=0.15, lw=0, label="10-90%")
        ax.axvline(0, color=MUTED, ls=":", lw=1)
        _ax(ax, s, "minutes from the step" if k >= 4 else None)
        if k == 0:
            ax.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="best")

    fig.suptitle(f"{case['valve']} steps {case['size']:+.3f} at row {r} "
                 f"({r * dt / 3600:.1f} h in)   -   context = {r} rows before the step",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, f"forecast_case{idx}.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    print("wrote", path)


def main():
    df = F.load_dataframe()
    dt = df.attrs["dt"]
    cases = find_valve_steps(df)[:N_CASES]
    if not cases:
        raise RuntimeError("no valve steps found past MIN_ROW - lower MIN_STEP or MIN_ROW")

    print(f"dataset {len(df)} rows at {dt:g} s ({len(df)*dt/3600:.1f} h), "
          f"horizon {HORIZON} rows ({HORIZON*dt/60:.0f} min)\n")
    header = f"{'case':>4} {'row':>6} {'valve':>18} {'step':>7} | " + " ".join(f"{s:>9}" for s in STATES)
    print(header)
    print(f"{'':>4} {'':>6} {'':>18} {'':>7} | " + " ".join(f"{'RMSE/std':>9}" for _ in STATES))
    for i, case in enumerate(cases, 1):
        out, truth, ctx, fut = run_case(df, case)
        nrmse = [np.sqrt(np.mean((out[s]["mean"] - truth[:, k]) ** 2)) / df[s].std()
                 for k, s in enumerate(STATES)]
        print(f"{i:>4} {case['row']:>6} {case['valve']:>18} {case['size']:>+7.3f} | "
              + " ".join(f"{x:>9.3f}" for x in nrmse), flush=True)
        plot_case(df, case, out, truth, ctx, fut, dt, i)


if __name__ == "__main__":
    main()
