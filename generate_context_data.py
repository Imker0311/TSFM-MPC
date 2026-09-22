"""Open-loop excitation run: rich valve excitation + overlapping ramp-hold-ramp
disturbances, simulated through the true plant, downsampled and saved.

The run ENDS AT THE NOMINAL OPERATING POINT: the last SETTLE_DURATION seconds carry no
disturbances and no valve moves, so the closed loop can be appended to this dataset without a
step change nothing caused. See "the settling tail" below.

Covariate channels : valve_cmd, coolant_valve_cmd, E_R, U_Ac, T_F, C_F, T_CF, Q_F
Target channels    : C_A, T, T_C, h, Q, Qc
Every process input is a covariate, so the forecaster is given the same information the
mathematical model gets. Per-event active flags go to a second file for plotting only.

Output is HDF5, one (N, 1) dataset per channel, as in CSTR-Simulation.ipynb.
"""

import os
import h5py
import numpy as np
import plant

# ============================== CONFIG ==============================
SEED = 42
OUT_DIR = "data"
FILE_DATA = "CSTR_ContextData.h5"           # covariates + targets (model channels)
FILE_EVENTS = "CSTR_ContextEvents.h5"       # per-disturbance active flags, plots only

# --- timing ---
TOTAL_DURATION = 100_000.0  # s, 27.8 h -> 10,000 saved rows
DT = plant.DT_OUTER         # s, simulation/outer step (1 s)
DOWNSAMPLE = 10             # -> 10 s sample interval = 6 samples/min (reactor tau ~1 min)

# --- the settling tail ------------------------------------------------------
# This dataset is the protected head of the MPC context: the live closed-loop tail is appended
# straight onto its last row. The loop runs at the nominal operating point, so unless the
# dataset ENDS there the join is a state step change with no input behind it - exactly the
# artifact the context manager works to avoid at its own splices. So the run finishes quiet:
# no events, both valves at their nominal positions, long enough for the reactor to settle.
SETTLE_DURATION = 2000.0    # s = 200 saved rows, ~33 reactor time constants
COOLANT_NOMINAL = 0.5       # coolant valve position the live loop sits at

# h is a pure integrator, so it is the one state a quiet tail does NOT bring home: at the
# level-balance valve position dh/dt is zero and the level simply stays wherever it drifted
# to. The mirrored valve pairs cancel their own contribution exactly, which leaves the Q_F
# events as the only source of drift - and their area is known before anything is simulated.
# One small outlet-valve pulse at the start of the settling tail unwinds it.
LEVEL_CORRECTION = 300.0    # s, duration of that pulse (amplitude comes out at ~0.1% of travel)

# --- outlet valve excitation ------------------------------------------------
# Open loop the level is a pure integrator, so a sustained flow imbalance empties or
# floods the tank within a couple of minutes. Steps are therefore emitted as MIRRORED
# PAIRS about the level-balance position: (l0+d, dur) then (l0-d, dur), whose flow areas
# cancel exactly, so h returns to where it started after every pair. Each hold is further
# capped so the excursion inside a pair stays within H_EXCURSION.
VALVE_CENTRE   = plant.NOMINAL_DIST[5] / (plant.Cv1 * np.sqrt(plant.dP))  # Q_F/Qmax -> dh/dt = 0
VALVE_AMP      = 0.30        # upper bound on the drawn |step| (the budget below usually binds first)
H_EXCURSION    = 0.12        # m, level budget per half-pair
VALVE_HOLD     = (150.0, 450.0)  # s = 2.5-7.5 reactor time constants, so each step settles
                                 # before the next move. Not longer: the level budget caps the
                                 # amplitude at H_EXCURSION*1000*A/(hold*Qmax), so a longer hold
                                 # buys settling at the cost of a weaker step. That trade is
                                 # physics, not preference - a held outlet offset drains the
                                 # tank. 450 s holds would allow only a 1% step.
VALVE_AMP_FRACTION = (0.6, 1.0)  # of what the drawn hold can afford. Drawing the amplitude
                                 # uniformly over [0, amp_max] instead leaves most of the
                                 # energy near the centre, because the log-uniform hold makes
                                 # long (hence small-amplitude) steps the common case.
VALVE_REST     = (300.0, 900.0)  # s, dwell back at the centre between pairs, so the reactor
                                 # settles and level excursions cannot compound across pairs

# --- coolant valve excitation -----------------------------------------------
# Free of any integrator, so this is a plain random multi-step over a wide range. Bounds are
# a safety margin: high coolant flow shrinks the reactor's extinction margin (see below), and
# below l_c ~ 0.22 it thermally runs away under a stacked worst-case disturbance.
COOLANT_RANGE  = (0.35, 0.80)
COOLANT_HOLD   = (600.0, 2400.0) # s, 10-40 reactor time constants. No integrator on this
                                 # valve, so these are true steps held to full steady state.

# --- disturbances (ramp-hold-ramp, from CSTR-Input.ipynb; OVERLAP ALLOWED) ---
# Shape, variance, sign and AR1 drift follow CSTR-Input. The DELTAS DO NOT: with the PI/cascade
# controllers removed nothing rejects a disturbance any more, and this reactor sits on the
# ignited branch of a bistable pair. The trailing comment on each line is the sustained
# single-channel deviation at which the ignited steady state folds and the reactor
# extinguishes, measured by continuation at the worst coolant valve position (0.85-0.9).
#
# Deltas sit well inside those folds because the binding case is several channels overlapping
# while the outlet valve is also moving the level (and so the residence time). Disturbances are
# deliberately the cheap side of that trade: spending margin here rather than on H_EXCURSION
# buys back outlet-valve range, which is the channel the model actually has to learn from.
# Events are long here (25-65 min), so a magnitude that was safe as a short pulse has much
# more time to push the reactor. The extinction guard in main() is the backstop.
RAMP_FRACTION = 0.15        # of event duration, ramping up, and again ramping down

FAULTS = {                                                                  # extinction fold
    "E_R":  dict(baseline=8750.0,   delta=15.0,   variance=0.1, signed=False, count=6, duration=(1500, 4000)),   # +140
    "U_Ac": dict(baseline=5e4 / 60, delta=-33.0, variance=0.1, signed=False, count=6, duration=(1500, 4000)),   # +186 / -333
    "T_F":  dict(baseline=320.0,    delta=1.5,    variance=0.2, signed=True,  count=6, duration=(1500, 4000)),   # -14.4
    "C_F":  dict(baseline=1.0,      delta=0.009,  variance=0.3, signed=True,  count=6, duration=(1500, 4000)),   # -0.088
    "T_CF": dict(baseline=300.0,    delta=1.05,    variance=0.2, signed=True,  count=6, duration=(1500, 4000)),   # -9.9
    "Q_F":  dict(baseline=100 / 60, delta=0.006,   variance=0.1, signed=True,  count=6, duration=(800, 2000)),    # -0.28, and see below
}

# Slow AR1 measurement drift, as in CSTR-Input.ipynb. Not applied to Q_F: open loop, a
# persistent feed-flow bias integrates straight into the level with nothing to pull it back.
# Q_F events are short and small for the same reason.
# The reactor sits on the ignited branch of a bistable pair, and open loop nothing pulls it
# back once it extinguishes - the rest of the run would then sit on the cold branch. Abort
# rather than write such a dataset.
EXTINCTION_C_A = 0.3        # mol/L; cold branch sits near C_F, ignited branch below ~0.3

AR1_CHANNELS = ("T_F", "C_F", "T_CF")
AR1_RHO, AR1_SIGMA = 0.9999, 0.005
AR1_FADE = 5000.0           # s over which the drift is faded out before the settling tail
MIN_GAP = 300               # samples, minimum quiet time between two events on the SAME channel
# ====================================================================


def valve_cmd_signal(N, dt, rng, limit=None):
    """Mirrored-pair random multi-step about the level-balance position.

    Amplitude and hold trade off against one another: |dQ| * hold is the litres of imbalance
    that go into the tank, and that is capped by H_EXCURSION. So the hold is drawn first
    (log-uniform, to cover timescales evenly) and the amplitude uniformly from whatever that
    hold can afford - short steps get to be large, long ones stay small. Drawing the amplitude
    first and clipping it instead would pile most draws onto the cap and turn the outlet valve
    into a 3-level signal.

    Only COMPLETE pairs are emitted, and none is started that would not finish before `limit`
    (the settling tail). Half a pair left hanging at the end would leave the level offset for
    good, since nothing open loop brings it back.
    """
    limit = N if limit is None else limit
    qmax = plant.Cv1 * np.sqrt(plant.dP)
    budget = H_EXCURSION * 1000.0 * plant.A     # L of imbalance allowed per half-pair
    u, n_steps = [], 0
    while len(u) < limit:
        hold = float(np.exp(rng.uniform(*np.log(VALVE_HOLD))))
        amp_max = min(VALVE_AMP, budget / (hold * qmax))
        amp = rng.choice([-1.0, 1.0]) * amp_max * rng.uniform(*VALVE_AMP_FRACTION)
        k = max(1, int(round(hold / dt)))
        if len(u) + 2 * k > limit:
            break
        for sign in (1.0, -1.0):
            u.extend([np.clip(VALVE_CENTRE + sign * amp, 0.0, 1.0)] * k)
            n_steps += 1
        u.extend([VALVE_CENTRE] * int(round(rng.uniform(*VALVE_REST) / dt)))
    u = u[:limit]
    return np.array(u + [VALVE_CENTRE] * (N - len(u))), n_steps


def coolant_cmd_signal(N, dt, rng, limit=None):
    """Random multi-step, parked back at COOLANT_NOMINAL for the settling tail."""
    limit = N if limit is None else limit
    u, n_steps = [], 0
    while len(u) < limit:
        level = rng.uniform(*COOLANT_RANGE)
        k = max(1, int(round(rng.uniform(*COOLANT_HOLD) / dt)))
        u.extend([level] * k)
        n_steps += 1
    u = u[:limit]
    return np.array(u + [COOLANT_NOMINAL] * (N - len(u))), n_steps


def ar1(N, rng, limit=None):
    """Slow multiplicative drift, faded back to exactly 1.0 by `limit`.

    A random walk does not return on its own, so without the fade the settling tail would sit
    at the nominal INPUTS while T_F, C_F and T_CF were still half a percent off - and the
    reactor would settle onto a slightly different state than the live loop runs at.
    """
    e = rng.normal(0.0, AR1_SIGMA, N)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = x[i - 1] * AR1_RHO + e[i]
    x /= 100.0
    if limit is not None:
        x *= np.clip((limit - np.arange(N)) / (AR1_FADE / DT), 0.0, 1.0)
    return x + 1.0


def disturbance_signals(N, dt, rng, limit=None):
    """Ramp-hold-ramp events, with a per-channel active flag. CSTR-Input's scheduler is dropped, so
    channels overlap freely; events within one channel stay disjoint (dirichlet gaps, as in
    CSTR-Input) because stacking a channel on itself would just breach its magnitude budget.

    Every event is scheduled to finish before `limit`, so the settling tail is quiet."""
    limit = N if limit is None else limit
    D, events, flags = {}, {}, {}
    for name, cfg in FAULTS.items():
        sig = np.full(N, cfg["baseline"])
        flag = np.zeros(N)
        n = cfg["count"]
        durs = np.round(rng.uniform(*cfg["duration"], size=n) / dt).astype(int)
        slack = limit - durs.sum() - MIN_GAP * (n + 1)
        if slack < 0:
            raise ValueError(f"{name}: events do not fit in the run - cut count or duration.")
        gaps = np.floor(rng.dirichlet(np.ones(n + 1)) * slack).astype(int) + MIN_GAP

        evs, pos = [], gaps[0]
        for i in range(n):
            dur = int(durs[i])
            mag = cfg["delta"] * (1 + rng.uniform(0, cfg["variance"]))
            if cfg["signed"]:
                mag *= rng.choice([1.0, -1.0])
            up = down = max(1, round(dur * RAMP_FRACTION))
            hold = max(0, dur - up - down)
            shape = np.concatenate([np.linspace(0, mag, up), np.full(hold, mag),
                                    np.linspace(mag, 0, down)])
            end = min(limit, pos + len(shape))
            sig[pos:end] += shape[:end - pos]
            flag[pos:end] = 1.0
            evs.append((pos * dt, (end - pos) * dt, mag))
            pos += dur + gaps[i + 1]

        if name in AR1_CHANNELS:
            sig *= ar1(N, rng, limit)
        D[name], events[name], flags[name] = sig, evs, flag
    return D, events, flags


def level_correction(U, D, start):
    """Unwind the level drift the Q_F events integrated in, with one outlet-valve pulse.

    dh = (Q_F - Q) dt / (1000 A), and the two terms separate: the mirrored valve pairs cancel
    their Q contribution exactly whatever Q_F is doing, so the whole drift is the Q_F excess
    area. That is known from the generated signal, before anything is simulated. Holding the
    valve dl above balance for LEVEL_CORRECTION seconds drains exactly that much back - the
    actuator lag does not change the integral, it only delays both edges equally.
    """
    qmax = plant.Cv1 * np.sqrt(plant.dP)
    excess = float(np.sum(D[:, plant.DIST_NAMES.index("Q_F")] - FAULTS["Q_F"]["baseline"]) * DT)
    dl = excess / (qmax * LEVEL_CORRECTION)
    k = int(round(LEVEL_CORRECTION / DT))
    U[start:start + k, 0] = np.clip(VALVE_CENTRE + dl, 0.0, 1.0)
    return excess, dl


def nominal_outputs():
    """Where the plant settles at the nominal inputs - what the dataset has to end at."""
    y = plant.NOMINAL_STATE.copy()
    u = np.array([VALVE_CENTRE, COOLANT_NOMINAL])
    for _ in range(int(round(3000.0 / DT))):
        y = plant.step(y, u, plant.NOMINAL_DIST)
    return plant.outputs(y)


def write_h5(path, t, channels):
    """One (N, 1) dataset per channel, overwritten in place on rerun."""
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=t.reshape(-1, 1), chunks=True, compression="gzip")
        for name, values in channels.items():
            f.create_dataset(name, data=np.asarray(values).reshape(-1, 1),
                             chunks=True, compression="gzip")
        f.attrs["dt"] = DT * DOWNSAMPLE
    print(f"wrote {path}: {len(t)} samples x {len(channels)} channels")


def main():
    rng = np.random.default_rng(SEED)
    N = int(round(TOTAL_DURATION / DT))
    settle_start = N - int(round(SETTLE_DURATION / DT))

    u_valve, n_valve_steps = valve_cmd_signal(N, DT, rng, settle_start)
    u_cool, n_cool_steps = coolant_cmd_signal(N, DT, rng, settle_start)
    U = np.column_stack([u_valve, u_cool])

    dist, events, flags = disturbance_signals(N, DT, rng, settle_start)
    D = np.column_stack([dist[n] for n in plant.DIST_NAMES])

    excess, dl = level_correction(U, D, settle_start)

    Y, Z = plant.simulate(plant.NOMINAL_STATE, U, D)
    Y, Z = Y[:N], Z[:N]                       # z[k] is the state seen when u[k] is applied
    if not np.all(np.isfinite(Z)):
        raise RuntimeError("Plant diverged - reduce excitation range or disturbance magnitudes.")
    cold = np.mean(Z[:, 0] > EXTINCTION_C_A)
    if cold > 0:
        raise RuntimeError(
            f"Reactor extinguished for {cold:.1%} of the run (first at "
            f"t={np.argmax(Z[:, 0] > EXTINCTION_C_A) * DT:.0f} s). Narrow COOLANT_RANGE, cut "
            f"the FAULTS deltas, or lower H_EXCURSION.")

    t = np.arange(N) * DT
    s = slice(None, None, DOWNSAMPLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    write_h5(os.path.join(OUT_DIR, FILE_DATA), t[s],
             dict(zip(plant.INPUT_NAMES, U[s].T))
             | dict(zip(plant.DIST_NAMES, D[s].T))
             | dict(zip(plant.OUTPUT_NAMES, Z[s].T)))
    # *_active marks the ramp-hold-ramp span of each event, as CSTR-Input's F{n}.plt flags did.
    write_h5(os.path.join(OUT_DIR, FILE_EVENTS), t[s],
             {f"{n}_active": flags[n][s] for n in plant.DIST_NAMES})

    report(t, U, Z, Y, D, events, N, (n_valve_steps, n_cool_steps))
    report_settle(Z, settle_start, excess, dl)


def report_settle(Z, settle_start, excess, dl):
    """The acceptance test for the settling tail: does the last row sit at nominal?"""
    nominal = nominal_outputs()
    spread = Z.std(axis=0)
    dev = (Z[-1] - nominal) / spread
    print(f"\nsettling tail        : last {SETTLE_DURATION:,.0f} s "
          f"({int(SETTLE_DURATION / (DT * DOWNSAMPLE))} saved rows) quiet, both valves nominal")
    print(f"  level correction   : Q_F events integrated {excess:+.2f} L into the tank "
          f"({excess / (1000 * plant.A):+.4f} m), unwound by a {dl:+.5f} valve pulse "
          f"({abs(dl) * 100:.2f}% of travel) over {LEVEL_CORRECTION:.0f} s")
    print(f"  {'channel':8s} {'nominal':>10} {'last row':>10} {'deviation':>10}")
    for i, name in enumerate(plant.OUTPUT_NAMES):
        print(f"  {name:8s} {nominal[i]:>10.4f} {Z[-1, i]:>10.4f} {dev[i]:>9.3f} spreads")
    print(f"  worst deviation    : {np.abs(dev).max():.3f} spreads on "
          f"{plant.OUTPUT_NAMES[int(np.abs(dev).argmax())]}")


def report(t, U, Z, Y, D, events, N, n_steps):
    n_ds = len(t[::DOWNSAMPLE])
    print(f"run duration        : {TOTAL_DURATION:,.0f} s = {TOTAL_DURATION/3600:.2f} h "
          f"({N:,} steps at dt={DT:g} s)")
    print(f"saved samples       : {n_ds:,} at {DT*DOWNSAMPLE:g} s "
          f"({60/(DT*DOWNSAMPLE):g} samples/min, downsample {DOWNSAMPLE}x)")

    print("\nvalve excitation coverage")
    for i, name in enumerate(plant.INPUT_NAMES):
        c = U[:, i]
        edges = np.linspace(c.min(), c.max(), 11)
        occ = np.histogram(c, bins=edges)[0] / N
        print(f"  {name:18s} range [{c.min():.3f}, {c.max():.3f}]  mean {c.mean():.3f}  "
              f"std {c.std():.3f}  decile occupancy min {occ.min():.1%}")
    print(f"  steps emitted      : valve {n_steps[0]}, coolant {n_steps[1]}")

    print(f"\nignition margin      : C_A max {Z[:, 0].max():.3f} of "
          f"{EXTINCTION_C_A} extinction threshold, T min {Z[:, 1].min():.1f} K")
    print("\nachieved flow / state ranges")
    for i, name in enumerate(plant.OUTPUT_NAMES):
        print(f"  {name:6s} [{Z[:, i].min():10.4f}, {Z[:, i].max():10.4f}]")
    print(f"  {'l':6s} [{Y[:, 4].min():10.4f}, {Y[:, 4].max():10.4f}]  (actual valve)")
    print(f"  {'l_c':6s} [{Y[:, 5].min():10.4f}, {Y[:, 5].max():10.4f}]  (actual valve)")

    print("\ndisturbance events")
    for j, name in enumerate(plant.DIST_NAMES):
        evs = events[name]
        mags = [m for _, _, m in evs]
        durs = [d for _, d, _ in evs]
        base = FAULTS[name]["baseline"]
        print(f"  {name:5s} n={len(evs)}  magnitude [{min(mags):+.4g}, {max(mags):+.4g}] "
              f"({min(mags)/base:+.1%}..{max(mags)/base:+.1%})  "
              f"duration [{min(durs):.0f}, {max(durs):.0f}] s  "
              f"signal [{D[:, j].min():.4g}, {D[:, j].max():.4g}]")
    active = np.zeros(N)
    for name in plant.DIST_NAMES:
        for start, dur, _ in events[name]:
            a, b = int(start / DT), int((start + dur) / DT)
            active[a:b] += 1
    print(f"  simultaneous active faults: max {int(active.max())}, mean {active.mean():.2f}, "
          f"quiet {np.mean(active == 0):.1%} of run")



if __name__ == "__main__":
    main()
