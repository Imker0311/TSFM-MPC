"""Closed loop: true CSTR in the loop, a foundation model as the MPC's internal model.

Each step measures the outputs and disturbances noisily, asks mpc.py for a move, applies it
to the true plant, and appends the result to the context. Disturbances are injected into the
plant only - the controller never sees them coming.

    python run_mpc.py        # -> data/mpc_run.npz, data/mpc_*.png
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant
import forecaster as F
import mpc
from context_manager import ContextManager, select_protected_block

# ============================== CONFIG ==============================
SEED = 3
OUT_DIR = "data"
DT = 10.0                       # s, must match the dataset interval and mpc.DT
N_STEPS = 450                   # control steps

# Cost per control step is linear in both context length and candidate count.
PROTECTED_ROWS = 400
MAX_CONTEXT = 2_000
PRUNE_TO = 1_600

# Plausible instrument sigmas, roughly 2-5% of each channel's spread. Not measured.
NOISE = {"C_A": 5e-4, "T": 0.15, "T_C": 0.15, "h": 2e-3, "Q": 2e-3, "Qc": 1e-3,
         "E_R": 2.0, "U_Ac": 1.5, "T_F": 0.15, "C_F": 1e-3, "T_CF": 0.10, "Q_F": 5e-4}

# Ramp-hold-ramp faults injected into the true plant, same shape as the generator emits.
SCENARIO = [
    dict(t=600.0,  channel="T_F",  magnitude=-1.5,  duration=1500.0),
    dict(t=2400.0, channel="U_Ac", magnitude=-33.0, duration=1500.0),
]
RAMP_FRACTION = 0.15

EXTINCTION_C_A = 0.3                    # mol/L
SNAPSHOTS = [5, 70, 250]                # steps whose plan is kept for the prediction figure
# ====================================================================

TARGETS = F.TARGET_COLUMNS
COVARIATES = F.COVARIATE_COLUMNS
DIST = F.DISTURBANCE_COLUMNS

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"


def _axes(ax, ylabel, xlabel=None):
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


def settled_state():
    """Settle the plant at nominal inputs; this is the state the dataset ends at."""
    y = plant.NOMINAL_STATE.copy()
    u = np.array([plant.NOMINAL_DIST[5] / (plant.Cv1 * np.sqrt(plant.dP)), 0.5])
    for _ in range(300):
        y = plant.step(y, u, plant.NOMINAL_DIST, dt=DT)
    return y, plant.outputs(y), u


def scenario_disturbances(n_steps, scenario=None):
    """Build the true disturbance trajectory and the event spans for the figures."""
    scenario = SCENARIO if scenario is None else scenario
    D = np.tile(plant.NOMINAL_DIST, (n_steps, 1))
    events = []
    for ev in scenario:
        col = DIST.index(ev["channel"])
        start = int(round(ev["t"] / DT))
        dur = int(round(ev["duration"] / DT))
        up = down = max(1, int(round(dur * RAMP_FRACTION)))
        mag = ev["magnitude"]
        shape = np.concatenate([np.linspace(0, mag, up), np.full(max(0, dur - up - down), mag),
                                np.linspace(mag, 0, down)])
        end = min(n_steps, start + len(shape))
        if end <= start:
            continue
        D[start:end, col] += shape[:end - start]
        events.append(dict(channel=ev["channel"], magnitude=mag, start=start, end=end))
    return D, events


def measure(values, columns, rng):
    """Add measurement noise to a vector of channel values."""
    return np.array([v + rng.normal(0.0, NOISE.get(c, 0.0)) for v, c in zip(values, columns)])


def run(cfg=None, n_steps=N_STEPS, seed=SEED, tag="mpc", model="zero-shot TimesFM-3"):
    """Run the closed loop and write the results and figures."""
    rng = np.random.default_rng(seed)
    df = F.load_dataframe()
    cfg = cfg if cfg is not None else mpc.Config(df, dt=DT)

    y, z_nominal, u_nominal = settled_state()
    scale = df[TARGETS].std().to_numpy()
    block = select_protected_block(df, PROTECTED_ROWS, TARGETS, z_nominal, scale=scale)
    mgr = ContextManager(block, TARGETS, COVARIATES, nominal=z_nominal, scale=scale, dt=DT,
                         max_context=MAX_CONTEXT, prune_to=PRUNE_TO)

    controller = mpc.MPC(cfg, np.random.default_rng(cfg.seed))
    D, events = scenario_disturbances(n_steps)

    Z = np.empty((n_steps, len(TARGETS)))
    Zm, U = np.empty_like(Z), np.empty((n_steps, len(cfg.manipulated)))
    wall = np.empty(n_steps)
    snapshots = {}

    u_prev = u_nominal.copy()
    t_start = time.perf_counter()
    for k in range(n_steps):
        z = plant.outputs(y)
        z_m = measure(z, TARGETS, rng)
        d_m = measure(D[k], DIST, rng)

        out = controller.step(mgr.get_context(), dict(zip(TARGETS, z_m)), d_m, u_prev)
        u = out["u"]

        if k in SNAPSHOTS:
            snapshots[k] = dict(prediction={c: out["prediction"][c].copy() for c in cfg.controlled},
                                band={c: (out["band"][c][0].copy(), out["band"][c][1].copy())
                                      for c in cfg.controlled})

        mgr.append(dict(zip(TARGETS, z_m)) | dict(zip(COVARIATES, np.concatenate([u, d_m]))))
        Z[k], Zm[k], U[k], wall[k] = z, z_m, u, out["wall_clock"]

        y = plant.step(y, u, D[k], dt=DT)
        u_prev = u

    r = dict(Z=Z, Zm=Zm, U=U, D=D, wall=wall, events=events,
             elapsed=time.perf_counter() - t_start, prunes=len(mgr.prune_log))
    report(cfg, r)
    np.savez_compressed(os.path.join(OUT_DIR, f"{tag}_run.npz"),
                        Z=Z, Zm=Zm, U=U, D=D, wall=wall, dt=cfg.dt)
    figures(cfg, r, snapshots, tag, model)
    return r


def report(cfg, r):
    """Print the metrics the run is judged on."""
    Z, U, wall = r["Z"], r["U"], r["wall"]
    quiet = r["events"][0]["start"] if r["events"] else len(Z)

    for c in cfg.controlled:
        e = Z[:, TARGETS.index(c)] - cfg.controlled[c]["setpoint"]
        q = e[:quiet]
        print(f"{c:3s} RMSE {np.sqrt((e ** 2).mean()):.4f}  max |dev| {np.abs(e).max():.4f}  "
              f"quiet RMSE {np.sqrt((q ** 2).mean()):.4f}")

    for i, m in enumerate(cfg.manipulated):
        span = (U[:, i].max() - U[:, i].min()) / (cfg.hi[i] - cfg.lo[i])
        rate_hit = np.mean(np.abs(np.diff(U[:, i])) >= cfg.rate[i] - 1e-9)
        print(f"{m:18s} [{U[:, i].min():.4f}, {U[:, i].max():.4f}] in clamp "
              f"[{cfg.lo[i]:.4f}, {cfg.hi[i]:.4f}]  uses {span:.1%}  rate-limited {rate_hit:.1%}")

    ca = Z[:, TARGETS.index("C_A")]
    print(f"C_A [{ca.min():.4f}, {ca.max():.4f}] mol/L, margin to extinction "
          f"{EXTINCTION_C_A - ca.max():.4f}")
    print(f"wall clock mean {wall.mean():.2f} s  worst {wall.max():.2f} s  "
          f"over interval {np.mean(wall > cfg.dt):.1%}  "
          f"{'PASS' if wall.max() <= cfg.dt else 'FAIL'}")
    print(f"context prunes {r['prunes']}   run {r['elapsed'] / 60:.1f} min")


def _events(ax, r, dt):
    for e in r["events"]:
        ax.axvspan(e["start"] * dt / 60, e["end"] * dt / 60, color="#f0ede6", zorder=0)


def figures(cfg, r, snapshots, tag, model):
    """Write the tracking, wall-clock and predicted-vs-realised figures."""
    t = np.arange(len(r["Z"])) * cfg.dt / 60.0
    cvs = list(cfg.controlled)

    fig, axes = plt.subplots(len(cvs) + 3, 1, figsize=(11, 3 + 2.1 * (len(cvs) + 3)), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, c in zip(axes, cvs):
        _axes(ax, c)
        _events(ax, r, cfg.dt)
        ax.axhline(cfg.controlled[c]["setpoint"], color=MUTED, ls="--", lw=1.0, label="setpoint")
        ax.plot(t, r["Zm"][:, TARGETS.index(c)], color=GRID, lw=0.8, label="measured")
        ax.plot(t, r["Z"][:, TARGETS.index(c)], color=C[0], lw=1.4, label="true")
        ax.legend(fontsize=7, frameon=False, ncol=3)

    for i, m in enumerate(cfg.manipulated):
        ax = axes[len(cvs) + i]
        _axes(ax, m)
        _events(ax, r, cfg.dt)
        ax.axhspan(cfg.lo[i], cfg.hi[i], color="#eef4fb", zorder=0)
        ax.axhline(cfg.lo[i], color=C[1], ls=":", lw=1.0)
        ax.axhline(cfg.hi[i], color=C[1], ls=":", lw=1.0, label="safety clamp")
        ax.plot(t, r["U"][:, i], color=C[2], lw=1.2, drawstyle="steps-post")
        ax.legend(fontsize=7, frameon=False)

    ax = axes[-1]
    _axes(ax, "C_A  [mol/L]", "time  [min]")
    _events(ax, r, cfg.dt)
    ca = r["Z"][:, TARGETS.index("C_A")]
    ax.plot(t, ca, color=C[3], lw=1.3, label=f"C_A, [{ca.min():.4f}, {ca.max():.4f}] mol/L")
    ax.axhline(EXTINCTION_C_A, color="#c02020", ls="--", lw=1.0, label="extinction")
    ax.legend(fontsize=7, frameon=False)
    axes[0].set_title(f"Closed loop: true CSTR, {model} as the internal model\n"
                      "(shaded = disturbance active, unknown to the controller)",
                      color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_tracking.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor("white")
    _axes(ax, "wall clock  [s]", "control step")
    ax.plot(r["wall"], color=C[0], lw=1.0)
    ax.axhline(cfg.dt, color="#c02020", ls="--", lw=1.2, label=f"interval {cfg.dt:g} s")
    ax.axhline(r["wall"].mean(), color=MUTED, ls=":", lw=1.0,
               label=f"mean {r['wall'].mean():.2f} s")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Time to solve one control step "
                 f"({cfg.n_iter} forecasts x {cfg.n_candidates} candidates)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_walltime.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(cvs), len(snapshots),
                             figsize=(4.2 * len(snapshots), 3.0 * len(cvs)), squeeze=False)
    fig.patch.set_facecolor("white")
    for j, (k, snap) in enumerate(sorted(snapshots.items())):
        for i, c in enumerate(cvs):
            ax = axes[i][j]
            pred = snap["prediction"][c]
            lo, hi = snap["band"][c]
            h = np.arange(len(pred))
            real = r["Z"][k:k + len(pred), TARGETS.index(c)]
            _axes(ax, c if j == 0 else "", "horizon step" if i == len(cvs) - 1 else None)
            ax.fill_between(h, lo, hi, color=C[0], alpha=0.15, label="10-90%")
            ax.plot(h, pred, color=C[0], lw=1.4, label="predicted")
            ax.plot(h[:len(real)], real, color=INK, lw=1.4, ls="--", label="realised")
            ax.axhline(cfg.controlled[c]["setpoint"], color=MUTED, ls=":", lw=0.9)
            if i == 0:
                ax.set_title(f"step {k}  (t = {k * cfg.dt / 60:.0f} min)", color=INK, fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Predicted vs realised. The prediction is one step's plan held open loop; the "
                 "realised trace is re-planned every step.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_prediction.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
