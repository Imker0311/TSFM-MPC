"""Exercise ContextManager on the real plant, without the forecaster or MPC.

The tail is driven by stepping plant.py open loop with a scripted input sequence that
repeatedly leaves the nominal operating point and returns to it - mirrored outlet-valve
pulses (whose flow areas cancel, so the level comes back), coolant-valve steps that return
to rest, and two ramp-hold-ramp disturbances. The length cap is set small enough that
several prunes fire over a 1,600-step run.

Three scenarios, covering both cut paths and the refusal path:
  1. the production setup - the excitation dataset now ends at nominal, so the selected block
     joins the live tail with a zero seam and its end is the oldest, cleanest cut point;
     every prune therefore anchors on the boundary and no stale head accumulates;
  2. a loop that never returns to nominal (the coolant valve parked off-rest), where the
     manager must refuse to splice rather than cut a large mismatched span;
  3. a block that does NOT end at nominal (a naive mid-run slice), so the boundary is not an
     eligible cut point and every cut has to land between two live near-nominal windows -
     and the bad boundary must be left exactly as it was found.

Q_F is deliberately never disturbed by the drive: the level is a pure integrator, so a
feed-flow offset moves h permanently and the state would never return to nominal again -
which is scenario 2's job, not a confound for the others.

    python test_context_manager.py
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant
import forecaster as F
from context_manager import ContextManager, select_protected_block

# ============================== CONFIG ==============================
SEED = 7
OUT_DIR = "data"
DT = 10.0                       # s per context row, the dataset's sample interval

PROTECTED_START = 4_000         # where a naive fixed slice would be taken from, for contrast
PROTECTED_ROWS = 400

# Deliberately tiny, so a 1,600-step run forces several prune cycles.
MAX_CONTEXT = 1_200
PRUNE_TO = 1_000
MATCH_WINDOW = 12
TOL = 0.10                      # spreads; 0.10 is ~0.7 K on T, ~0.005 m on h
TOL_MAX = 0.30
TOL_HEAD = 0.20                 # scenario 3 only: loose enough that the protected block's own
                                # end qualifies as a cut point
MAX_PRUNE_FRACTION = 0.5
MIN_PRUNE_SPAN = 40
KEEP_RECENT = 50

N_STEPS = 1_600

# --- input script (rows) ---
REST = 35                       # at nominal, after the reactor has settled
PULSE = 12                      # each half of the mirrored outlet pair
REST_MID = 25
COOL = 18
REST_END = 25
PULSE_AMP = (0.020, 0.035)      # outlet valve, within the level budget at this hold
COOLANT_RANGE = (0.38, 0.62)
VALVE_CENTRE = plant.NOMINAL_DIST[5] / (plant.Cv1 * np.sqrt(plant.dP))
COOLANT_REST = 0.5

# --- second scenario: the state never comes back to nominal ---
STUCK_STEPS = 700
STUCK_COOLANT = 0.30

SEAM_TOL = 0.75                 # channel spreads, allowed protected/live boundary jump. Looser
                                # than a prune splice on purpose: the block is fixed, so this
                                # seam is only as good as the dataset's closest approach.
JUMP_TOL = 0.25                 # channel spreads, allowed mismatch between two glued ends
JUMP_STEPS = 3.0                # a seam may not step further than this x the drive's own
                                # largest one-step move
# ====================================================================

TARGETS = F.TARGET_COLUMNS
COVARIATES = F.COVARIATE_COLUMNS
CHANNELS = TARGETS + COVARIATES

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
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


# ------------------------------------------------------------------ the drive
def settle():
    """Plant state and outputs after resting at the nominal inputs - the reference point.

    CSTR-Simulation's initial condition is not quite a steady state of this plant, so the
    nominal vector is measured rather than assumed.
    """
    y = plant.NOMINAL_STATE.copy()
    u = np.array([VALVE_CENTRE, COOLANT_REST])
    for _ in range(300):
        y = plant.step(y, u, plant.NOMINAL_DIST, dt=DT)
    return y, plant.outputs(y)


def ramp_hold_ramp(D, col, mag, start, dur, ramp=0.15):
    up = down = max(1, int(round(dur * ramp)))
    shape = np.concatenate([np.linspace(0, mag, up), np.full(max(0, dur - up - down), mag),
                            np.linspace(mag, 0, down)])
    end = min(len(D), start + len(shape))
    D[start:end, col] += shape[:end - start]


def scripted_inputs(n, rng):
    """Mirrored outlet pulses and coolant steps, each followed by a rest at nominal."""
    U = np.tile([VALVE_CENTRE, COOLANT_REST], (n, 1))
    D = np.tile(plant.NOMINAL_DIST, (n, 1))

    k = REST
    while k < n:
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(*PULSE_AMP)
        U[k:k + PULSE, 0] = VALVE_CENTRE + amp
        U[k + PULSE:k + 2 * PULSE, 0] = VALVE_CENTRE - amp
        k += 2 * PULSE + REST_MID
        U[k:k + COOL, 1] = rng.uniform(*COOLANT_RANGE)
        k += COOL + REST_END + REST

    # Two disturbances that return to baseline, placed to overlap rests - so not every
    # return to the rest inputs is a return to the nominal STATE.
    ramp_hold_ramp(D, plant.DIST_NAMES.index("T_F"), 1.6, int(0.22 * n), 90)
    ramp_hold_ramp(D, plant.DIST_NAMES.index("U_Ac"), -33.0, int(0.55 * n), 120)
    ramp_hold_ramp(D, plant.DIST_NAMES.index("T_F"), -1.4, int(0.78 * n), 80)
    return U, D


def record(z, u, d):
    return dict(zip(TARGETS, z)) | dict(zip(COVARIATES, np.concatenate([u, d])))


def drive(mgr, U, D, y0):
    """Step the plant, append every row, and track what each prune did."""
    y = y0.copy()
    driven, lengths, seams, splices = [], [], [], []
    for k in range(len(U)):
        z = plant.outputs(y)
        mgr.append(record(z, U[k], D[k]))
        driven.append([*z, *U[k], *D[k]])

        for e in mgr.prune_log[len(splices):]:
            # Seams logged earlier shift down, or vanish inside a later cut.
            seams = [s - e["span"] if s > e["cut_to"] else s
                     for s in seams if not (e["cut_from"] <= s <= e["cut_to"])]
            seams.append(e["cut_from"] - 1)
            splices.append(e)
        lengths.append(len(mgr))
        y = plant.step(y, U[k], D[k], dt=DT)
    return np.array(driven), np.array(lengths), seams


# ------------------------------------------------------------------ checks
def check_protected_unchanged(mgr, protected0):
    ctx = mgr.get_context()
    assert np.array_equal(mgr.protected_block.to_numpy(), protected0), "protected block mutated"
    assert np.array_equal(ctx[CHANNELS].to_numpy()[:len(protected0)], protected0), \
        "context does not start with the protected block"
    print(f"protected immutable  : {len(protected0)} rows, bit-identical after "
          f"{len(mgr.prune_log)} prunes")


def check_under_cap(lengths, mgr):
    assert lengths.max() <= mgr.max_context, \
        f"context reached {lengths.max()} rows, cap is {mgr.max_context}"
    print(f"context length       : max {lengths.max()} rows (cap {mgr.max_context}, "
          f"prune target {mgr.prune_to}), final {lengths[-1]}")


def check_cuts_at_nominal(mgr):
    assert mgr.prune_log, "no prune fired - raise N_STEPS or lower MAX_CONTEXT"
    for e in mgr.prune_log:
        assert e["tol"] <= mgr.tol_max, f"prune used tol={e['tol']:g} above the bound"
        assert e["dist_before"] <= e["tol"] and e["dist_after"] <= e["tol"], \
            f"prune cut at distance {e['dist_before']:.3f}/{e['dist_after']:.3f}, tol {e['tol']:.3f}"
        assert e["span"] <= mgr.max_prune_fraction * e["tail_before"], "prune exceeded its fraction"
        assert e["span"] >= mgr.min_prune_span, "prune below the minimum span"
    print(f"cut points           : every one of {len(mgr.prune_log)} prunes cut between two "
          f"near-nominal windows (worst distance {max(max(e['dist_before'], e['dist_after']) for e in mgr.prune_log):.4f})")


def check_splice_continuity(mgr, driven, expect):
    """Max discontinuity per state across each splice, against the natural one-step move.

    Two numbers per splice. `mismatch` is what the splice introduces - the gap between the
    two states glued together - and is what has to stay small. `jump` is the step actually
    left in the context at the seam; it is legitimately larger when an input moves on the
    row after the cut, so it is checked only against the largest one-step move the drive
    itself makes, not against zero.
    """
    nt = len(TARGETS)
    typical = np.percentile(np.abs(np.diff(driven[:, :nt], axis=0)), 99.9, axis=0)
    kinds = {"boundary" if e["cut_from"] == 0 else "interior" for e in mgr.prune_log}
    assert kinds == {expect}, f"expected only {expect} cuts, got {sorted(kinds)}"

    print(f"\n{'splice':>6} {'cut rows':>13} {'span':>5} {'anchor':>9} {'dist(-)':>8} "
          f"{'dist(+)':>8} {'tol':>5} | " + " ".join(f"{s:>9}" for s in TARGETS))
    print(f"{'':>6} {'':>13} {'':>5} {'':>9} {'':>8} {'':>8} {'':>5} | "
          + " ".join(f"{'mismatch':>9}" for _ in TARGETS))
    worst_mismatch, worst_jump = np.zeros(nt), np.zeros(nt)
    for i, e in enumerate(mgr.prune_log, 1):
        mismatch = e["join_mismatch"][:nt] / mgr.scale
        boundary = e["cut_from"] == 0
        worst_mismatch = np.maximum(worst_mismatch, mismatch)
        worst_jump = np.maximum(worst_jump, e["join_jump"][:nt] / mgr.scale)
        print(f"{i:>6} {str(e['cut_from']) + '-' + str(e['cut_to']):>13} {e['span']:>5} "
              f"{'boundary' if boundary else 'interior':>9} "
              f"{e['dist_before']:>8.4f} {e['dist_after']:>8.4f} {e['tol']:>5.2f} | "
              + " ".join(f"{x:>9.5f}" for x in mismatch))

    kind = "boundary-anchored" if expect == "boundary" else "interior"
    print(f"\nworst discontinuity per state across the {kind} splices, in channel spreads:")
    for k, s in enumerate(TARGETS):
        absolute = worst_mismatch[k] * mgr.scale[k]
        print(f"  {s:5s} mismatch {worst_mismatch[k]:8.5f} spreads ({absolute:10.6f} absolute, "
              f"{absolute / typical[k]:5.2f}x the 99.9th-pct one-step move)   "
              f"seam step {worst_jump[k]:7.4f} spreads "
              f"({worst_jump[k] * mgr.scale[k] / typical[k]:5.2f}x)")
        assert worst_mismatch[k] <= JUMP_TOL, \
            f"{s} mismatches {worst_mismatch[k]:.3f} spreads across a splice"
        assert worst_jump[k] * mgr.scale[k] <= JUMP_STEPS * typical[k], \
            f"{s} steps {worst_jump[k] * mgr.scale[k] / typical[k]:.1f}x its largest one-step move at a seam"


def check_tail_is_subsequence(mgr, driven):
    """Pruning may only delete rows - never reorder, edit or duplicate them."""
    tail, i = mgr.tail, 0
    for row in tail:
        while i < len(driven) and not np.array_equal(driven[i], row):
            i += 1
        assert i < len(driven), "tail row is not a row the plant produced, in order"
        i += 1
    print(f"\ntail integrity       : all {len(tail)} tail rows are the driven rows, in order, "
          f"{len(driven) - len(tail)} deleted")


# ------------------------------------------------------------------ scenarios
def run_scenario(df, rng, protected, label, expect, seam_bound=None):
    """One full drive against a given protected block. `expect` is the cut anchor it should use.

    `seam_bound` applies where the block was chosen to end at nominal. Where it was not, the
    boundary is bad by construction and that is the block's doing, not the manager's - what
    matters there is that interior cuts leave it exactly as they found it.
    """
    y0, nominal = settle()
    scale = df[TARGETS].std().to_numpy()
    mgr = ContextManager(
        protected, TARGETS, COVARIATES, nominal=nominal, scale=scale, dt=DT,
        max_context=MAX_CONTEXT, prune_to=PRUNE_TO, match_window=MATCH_WINDOW,
        tol=TOL, tol_max=TOL_MAX, max_prune_fraction=MAX_PRUNE_FRACTION,
        min_prune_span=MIN_PRUNE_SPAN, keep_recent=KEEP_RECENT)
    protected0 = mgr.protected_block.to_numpy().copy()
    print(f"\n{label}")
    print(f"block end distance   : {mgr.head_distance:.4f} from nominal "
          f"({'eligible' if mgr.head_distance <= TOL else 'NOT eligible'} as a cut point at "
          f"tol {TOL:g}) -> expect {expect} cuts")

    U, D = scripted_inputs(N_STEPS, rng)
    driven, lengths, seams = drive(mgr, U, D, y0)
    print(f"drive                : {N_STEPS} steps at {DT:g} s ({N_STEPS * DT / 3600:.1f} h), "
          f"{len(mgr.near_nominal_indices())} near-nominal rows logged in the surviving tail\n")

    check_protected_unchanged(mgr, protected0)
    seam = mgr.seam_mismatch
    initial = np.abs(driven[0, :len(TARGETS)] - protected0[-1, :len(TARGETS)]) / mgr.scale
    print("protected/live seam  : " + "  ".join(f"{s}={v:.3f}" for s, v in zip(TARGETS, seam))
          + f"  spreads (worst {seam.max():.3f})")
    if seam_bound is not None:
        assert seam.max() <= seam_bound, \
            f"protected/live boundary jumps {seam.max():.2f} spreads, bound {seam_bound:g}"
    else:
        assert np.array_equal(seam, initial), "interior cuts altered the protected/live boundary"
        print(f"                       untouched by {len(mgr.prune_log)} interior cuts "
              f"(bit-identical to step 1)")
    check_under_cap(lengths, mgr)
    check_cuts_at_nominal(mgr)
    check_splice_continuity(mgr, driven, expect)
    check_tail_is_subsequence(mgr, driven)

    ctx = mgr.get_context()
    gaps = ctx["timestamp"].diff().dropna().unique()
    assert len(gaps) == 1, f"context timestamps are not contiguous: {gaps}"
    assert len(ctx) == len(mgr), "context frame length disagrees with the manager"
    print(f"context frame        : {ctx.shape[0]} rows x {ctx.shape[1]} cols, one uniform "
          f"{gaps[0].total_seconds():g} s interval across the seam")
    return mgr, lengths, seams, nominal


def blocks(df):
    """The production block (dataset ends at nominal, so the selector takes the last rows) and
    a naive mid-run slice that does not end at nominal - each exercises a different cut path."""
    _, nominal = settle()
    scale = df[TARGETS].std().to_numpy()
    chosen = select_protected_block(df, PROTECTED_ROWS, TARGETS, nominal, scale=scale,
                                    match_window=MATCH_WINDOW, tol=TOL, tol_max=TOL_MAX)
    naive = df.iloc[PROTECTED_START:PROTECTED_START + PROTECTED_ROWS]
    gap = np.abs(naive[TARGETS].to_numpy()[-1] - nominal) / scale
    print(f"nominal reference    : " + "  ".join(f"{s}={v:.4g}" for s, v in zip(TARGETS, nominal)))
    print(f"selected block       : rows {chosen.attrs['end_row'] + 1 - PROTECTED_ROWS}-"
          f"{chosen.attrs['end_row']}, ends {chosen.attrs['seam_distance']:.4f} from nominal")
    print(f"naive block          : rows {PROTECTED_START}-{PROTECTED_START + PROTECTED_ROWS - 1}, "
          f"joins the live tail {gap.max():.2f} spreads out on its worst channel")
    return chosen, naive


def run_stuck(df):
    """No return to nominal: the manager must refuse to splice, not cut a mismatched span."""
    print("\n" + "-" * 78)
    print("scenario 2: the loop never returns to nominal (coolant valve parked off-rest)")
    y0, nominal = settle()
    protected = df.iloc[PROTECTED_START:PROTECTED_START + 300]
    mgr = ContextManager(
        protected, TARGETS, COVARIATES, nominal=nominal, dt=DT,
        max_context=700, prune_to=600, match_window=MATCH_WINDOW, tol=TOL, tol_max=TOL_MAX,
        max_prune_fraction=MAX_PRUNE_FRACTION, min_prune_span=MIN_PRUNE_SPAN,
        keep_recent=KEEP_RECENT)

    U = np.tile([VALVE_CENTRE, STUCK_COOLANT], (STUCK_STEPS, 1))
    D = np.tile(plant.NOMINAL_DIST, (STUCK_STEPS, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drive(mgr, U, D, y0)

    assert not mgr.prune_log, "pruned a tail that never returned to nominal"
    assert mgr.skipped_prunes > 0, "expected the prune to be skipped"
    assert any("no near-nominal pair" in str(w.message) for w in caught), "expected a warning"
    assert len(mgr) == 300 + STUCK_STEPS, "rows were dropped despite no valid cut"
    d = mgr.distances[~np.isnan(mgr.distances)]
    print(f"  {STUCK_STEPS} steps, closest approach to nominal {d.min():.3f} "
          f"(tol {TOL:g}, widened to at most {TOL_MAX:g})")
    print(f"  prunes {len(mgr.prune_log)}, skipped {mgr.skipped_prunes}, context "
          f"{len(mgr)} rows - held intact and over its cap, as intended")
    print(f"  warning: {caught[0].message}")
    return mgr


# ------------------------------------------------------------------ plots
def plot(mgr, lengths, seams, nominal):
    ctx = mgr.get_context()
    p = mgr.protected_length
    x = np.arange(len(ctx))
    fig = plt.figure(figsize=(13, 13), facecolor=SURFACE)
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 0.9, 0.9], hspace=0.28)

    for i, (name, colour) in enumerate([("T", C[0]), ("h", C[2])]):
        ax = fig.add_subplot(gs[i])
        ax.axvspan(0, p, color=MUTED, alpha=0.07, lw=0)
        ax.plot(x, ctx[name], lw=1.1, color=colour)
        ax.axhline(nominal[TARGETS.index(name)], color=MUTED, ls=":", lw=1)
        for s in seams:
            ax.axvline(p + s + 0.5, color=C[1], lw=1.1, alpha=0.8)
        _axes(ax, f"{name}")
        if i == 0:
            ax.set_title("protected block (shaded) + live tail, splices marked",
                         color=INK, fontsize=10, loc="left")

    ax = fig.add_subplot(gs[2])
    d = mgr.distances
    ax.plot(p + np.arange(len(d)), d, lw=0.9, color=C[4])
    near = mgr.near_nominal_indices()
    ax.plot(p + near, d[near], ".", ms=3, color=C[2], label="logged near-nominal")
    ax.axhline(mgr.tol, color=C[1], ls="--", lw=1, label=f"tol {mgr.tol:g}")
    ax.axhline(mgr.tol_max, color=C[1], ls=":", lw=1, label=f"tol_max {mgr.tol_max:g}")
    for s in seams:
        ax.axvline(p + s + 0.5, color=C[1], lw=1.1, alpha=0.5)
    ax.set_yscale("log")
    _axes(ax, "window distance\nfrom nominal")
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.0))

    ax = fig.add_subplot(gs[3])
    ax.plot(np.arange(len(lengths)), lengths, lw=1.2, color=C[0])
    ax.axhline(mgr.max_context, color=C[1], ls="--", lw=1, label="max_context")
    ax.axhline(mgr.prune_to, color=C[3], ls="--", lw=1, label="prune_to")
    ax.axhline(p, color=MUTED, ls=":", lw=1, label="protected")
    _axes(ax, "context length\n[rows]", "control step")
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="lower right", ncol=3)

    # Each splice, in normalised deviation from nominal, both states either side.
    ax = fig.add_subplot(gs[4])
    nt = len(TARGETS)
    w = 40
    full = ctx[TARGETS].to_numpy()
    for i, s in enumerate(seams):
        lo, hi = max(0, p + s - w), min(len(full), p + s + w)
        seg = (full[lo:hi] - nominal) / mgr.scale
        rel = np.arange(lo, hi) - (p + s) + i * (2 * w + 20)
        for k in range(nt):
            ax.plot(rel, seg[:, k], lw=0.9, color=C[k % len(C)],
                    label=TARGETS[k] if i == 0 else None)
        ax.axvline(i * (2 * w + 20) + 0.5, color=INK, lw=1.0, alpha=0.7)
    _axes(ax, "deviation from\nnominal [spreads]",
          "rows either side of each splice (the protected/live boundary shows as the step "
          "left of the first)")
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED, ncol=6,
              loc="lower left", bbox_to_anchor=(0.0, 1.0))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "context_manager_test.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nwrote {path}")


def main():
    rng = np.random.default_rng(SEED)
    df = F.load_dataframe()
    print(f"dataset {len(df)} rows at {DT:g} s, {PROTECTED_ROWS} of them protected\n")

    chosen, naive = blocks(df)
    mgr, lengths, seams, nominal = run_scenario(
        df, rng, chosen, "scenario 1: production block - the dataset ends at nominal",
        "boundary", seam_bound=SEAM_TOL)
    run_scenario(df, np.random.default_rng(SEED), naive,
                 "scenario 3: a block that does NOT end at nominal - cuts must land in the tail",
                 "interior")
    run_stuck(df)
    plot(mgr, lengths, seams, nominal)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
