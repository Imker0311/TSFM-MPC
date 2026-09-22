"""Diagnostic plots for the open-loop excitation run: valve commands, the six target
channels, disturbance overlays, and excitation coverage."""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "data"
# Categorical slots 1-6 of the reference palette, assigned in fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SURFACE = "#fcfcfb"

COVARIATES = ["valve_cmd", "coolant_valve_cmd"]
TARGETS = ["C_A", "T", "T_C", "h", "Q", "Qc"]
DISTURBANCES = ["E_R", "U_Ac", "T_F", "C_F", "T_CF", "Q_F"]


def _axes(ax, ylabel):
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def timeseries(t, U, Z, D, dnames, active, window=None, name="context_timeseries"):
    tm = t / 60.0
    panels = [
        ("valve position\ncommand [-]", [(U[:, 0], "valve_cmd", C[0]), (U[:, 1], "coolant_valve_cmd", C[1])]),
        ("C_A [mol/L]", [(Z[:, 0], "C_A", C[0])]),
        ("T, T_C [K]",  [(Z[:, 1], "T", C[0]), (Z[:, 2], "T_C", C[1])]),
        ("h [m]",       [(Z[:, 3], "h", C[0])]),
        ("Q [L/s]",     [(Z[:, 4], "Q", C[0])]),
        ("Qc [L/s]",    [(Z[:, 5], "Qc", C[0])]),
    ]
    # Disturbances get their own short panel each: U_Ac swings ~10x further than T_F or T_CF,
    # so sharing one axis would flatten the small channels into the zero line.
    n = len(panels) + len(dnames)
    fig = plt.figure(figsize=(13, 17), facecolor=SURFACE)
    gs = fig.add_gridspec(n, 1, height_ratios=[1.0] * len(panels) + [0.42] * len(dnames),
                          hspace=0.16)
    axes = [fig.add_subplot(gs[i]) for i in range(n)]

    for ax, (ylabel, series) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.fill_between(tm, 0, 1, where=active > 0, transform=ax.get_xaxis_transform(),
                        color=MUTED, alpha=0.07, lw=0, zorder=0)
        for y, label, colour in series:
            ax.plot(tm, y, lw=1.2, color=colour, label=label)
        _axes(ax, ylabel)
        if len(series) > 1:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8,
                      frameon=False, ncol=len(series), labelcolor=MUTED,
                      borderpad=0.0, handlelength=1.4)
        else:
            ax.text(0.004, 0.94, series[0][1], transform=ax.transAxes, va="top",
                    fontsize=8, color=MUTED)

    for ax, j in zip(axes[len(panels):], range(len(dnames))):
        ax.set_facecolor(SURFACE)
        base = np.median(D[:, j])
        ax.plot(tm, 100.0 * (D[:, j] - base) / abs(base), lw=1.2, color=C[j])
        _axes(ax, "")
        ax.set_ylabel(f"{dnames[j]}\n[% dev]", color=MUTED, fontsize=8)
        ax.axhline(0, color=GRID, lw=0.8)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time [min]", color=MUTED, fontsize=9)
    lo, hi = window if window else (tm[0], tm[-1])
    for ax in axes:
        ax.set_xlim(lo, hi)

    title = "Open-loop CSTR excitation run  -  shaded spans have >=1 disturbance active"
    if window:
        title += f"  |  zoom {lo:g}-{hi:g} min"
    axes[0].set_title(title, color=INK, fontsize=12, loc="left", pad=22)
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", path)


def coverage(U, Z):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), facecolor=SURFACE)
    for ax, (x, name, colour) in zip(axes[:2], [(U[:, 0], "valve_cmd", C[0]),
                                                (U[:, 1], "coolant_valve_cmd", C[1])]):
        ax.set_facecolor(SURFACE)
        ax.hist(x, bins=30, color=colour, edgecolor=SURFACE, lw=0.8)
        _axes(ax, "samples")
        ax.set_xlabel(name, color=MUTED, fontsize=9)

    ax = axes[2]
    ax.set_facecolor(SURFACE)
    ax.hist2d(U[:, 0], U[:, 1], bins=24, cmap="Blues")
    _axes(ax, "coolant_valve_cmd")
    ax.set_xlabel("valve_cmd", color=MUTED, fontsize=9)
    ax.set_title("joint occupancy", color=INK, fontsize=10, loc="left")

    ax = axes[3]
    ax.set_facecolor(SURFACE)
    ax.scatter(Z[:, 1], Z[:, 0], s=4, color=C[2], alpha=0.4, lw=0)
    _axes(ax, "C_A [mol/L]")
    ax.set_xlabel("T [K]", color=MUTED, fontsize=9)
    ax.set_title("operating points visited", color=INK, fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "context_coverage.png"), dpi=130, facecolor=SURFACE)
    print("wrote", os.path.join(OUT_DIR, "context_coverage.png"))


def main():
    with h5py.File(os.path.join(OUT_DIR, "CSTR_ContextData.h5"), "r") as f:
        t = f["t"][:, 0]
        U = np.column_stack([f[n][:, 0] for n in COVARIATES])
        Z = np.column_stack([f[n][:, 0] for n in TARGETS])
        D = np.column_stack([f[n][:, 0] for n in DISTURBANCES])
    with h5py.File(os.path.join(OUT_DIR, "CSTR_ContextEvents.h5"), "r") as f:
        active = np.sum([f[f"{n}_active"][:, 0] for n in DISTURBANCES], axis=0)
    dnames = DISTURBANCES

    timeseries(t, U, Z, D, dnames, active)
    timeseries(t, U, Z, D, dnames, active, window=(0, 60), name="context_zoom")
    coverage(U, Z)


if __name__ == "__main__":
    main()
