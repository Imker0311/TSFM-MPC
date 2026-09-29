"""Open-loop excitation run: valve excitation plus ramp-hold-ramp disturbances, simulated
through the true plant, downsampled and saved.

The run ends at the nominal operating point so a closed-loop tail can be appended without a
step change nothing caused.

Covariates : valve_cmd, coolant_valve_cmd, E_R, U_Ac, T_F, C_F, T_CF, Q_F
Targets    : C_A, T, T_C, h, Q, Qc
"""

import os
import h5py
import numpy as np
import plant

# ============================== CONFIG ==============================
SEED = 42
OUT_DIR = "data"
FILE_DATA = "CSTR_ContextData.h5"           # covariates + targets
FILE_EVENTS = "CSTR_ContextEvents.h5"       # per-disturbance active flags, for plots only

TOTAL_DURATION = 100_000.0  # s
DT = plant.DT_OUTER         # s, simulation step
DOWNSAMPLE = 10             # -> 10 s sample interval

# The run finishes quiet so its last row sits at nominal, which is where the closed loop
# starts. h is a pure integrator and does not come home on its own, so one small outlet-valve
# pulse unwinds the drift the Q_F events integrated in.
SETTLE_DURATION = 2000.0    # s of quiet tail
COOLANT_NOMINAL = 0.5       # coolant valve position the live loop sits at
LEVEL_CORRECTION = 300.0    # s, duration of the level-correction pulse

# Outlet valve: the level is a pure integrator, so steps are emitted as mirrored pairs about
# the level-balance position, whose flow areas cancel. Each hold is capped so the excursion
# within a pair stays inside H_EXCURSION.
VALVE_CENTRE   = plant.NOMINAL_DIST[5] / (plant.Cv1 * np.sqrt(plant.dP))
VALVE_AMP      = 0.30            # upper bound on the drawn step
H_EXCURSION    = 0.12            # m, level budget per half-pair
VALVE_HOLD     = (150.0, 450.0)  # s
VALVE_AMP_FRACTION = (0.6, 1.0)  # of what the drawn hold can afford
VALVE_REST     = (300.0, 900.0)  # s, dwell at centre between pairs

# Coolant valve: no integrator, so plain random multi-steps. Bounds are a safety margin
# against extinction at one end and thermal runaway at the other.
COOLANT_RANGE  = (0.35, 0.80)
COOLANT_HOLD   = (600.0, 2400.0)  # s

RAMP_FRACTION = 0.15        # of event duration, ramping up and again down

# Deltas sit well inside the single-channel extinction fold (trailing comment on each line),
# because the binding case is several channels overlapping while the valves also move.
FAULTS = {                                                                  # extinction fold
    "E_R":  dict(baseline=8750.0,   delta=15.0,   variance=0.1, signed=False, count=6, duration=(1500, 4000)),   # +140
    "U_Ac": dict(baseline=5e4 / 60, delta=-33.0, variance=0.1, signed=False, count=6, duration=(1500, 4000)),   # +186 / -333
    "T_F":  dict(baseline=320.0,    delta=1.5,    variance=0.2, signed=True,  count=6, duration=(1500, 4000)),   # -14.4
    "C_F":  dict(baseline=1.0,      delta=0.009,  variance=0.3, signed=True,  count=6, duration=(1500, 4000)),   # -0.088
    "T_CF": dict(baseline=300.0,    delta=1.05,    variance=0.2, signed=True,  count=6, duration=(1500, 4000)),   # -9.9
    "Q_F":  dict(baseline=100 / 60, delta=0.006,   variance=0.1, signed=True,  count=6, duration=(800, 2000)),    # -0.28
}

EXTINCTION_C_A = 0.3        # mol/L; the run aborts if the reactor extinguishes

AR1_CHANNELS = ("T_F", "C_F", "T_CF")   # slow multiplicative measurement drift
AR1_RHO, AR1_SIGMA = 0.9999, 0.005
AR1_FADE = 5000.0           # s over which the drift fades before the settling tail
MIN_GAP = 300               # samples of quiet between two events on the same channel
# ====================================================================


def valve_cmd_signal(N, dt, rng, limit=None):
    """Mirrored-pair random multi-step about the level-balance position.

    The hold is drawn first and the amplitude from whatever that hold's level budget allows,
    so short steps are large and long ones small. Only complete pairs are emitted.
    """
    limit = N if limit is None else limit
    qmax = plant.Cv1 * np.sqrt(plant.dP)
    budget = H_EXCURSION * 1000.0 * plant.A     # L of imbalance allowed per half-pair
    u = []
    while len(u) < limit:
        hold = float(np.exp(rng.uniform(*np.log(VALVE_HOLD))))
        amp_max = min(VALVE_AMP, budget / (hold * qmax))
        amp = rng.choice([-1.0, 1.0]) * amp_max * rng.uniform(*VALVE_AMP_FRACTION)
        k = max(1, int(round(hold / dt)))
        if len(u) + 2 * k > limit:
            break
        for sign in (1.0, -1.0):
            u.extend([np.clip(VALVE_CENTRE + sign * amp, 0.0, 1.0)] * k)
        u.extend([VALVE_CENTRE] * int(round(rng.uniform(*VALVE_REST) / dt)))
    u = u[:limit]
    return np.array(u + [VALVE_CENTRE] * (N - len(u)))


def coolant_cmd_signal(N, dt, rng, limit=None):
    """Random multi-step, parked back at nominal for the settling tail."""
    limit = N if limit is None else limit
    u = []
    while len(u) < limit:
        level = rng.uniform(*COOLANT_RANGE)
        u.extend([level] * max(1, int(round(rng.uniform(*COOLANT_HOLD) / dt))))
    u = u[:limit]
    return np.array(u + [COOLANT_NOMINAL] * (N - len(u)))


def ar1(N, rng, limit=None):
    """Slow multiplicative drift, faded back to exactly 1.0 so the tail ends at nominal."""
    e = rng.normal(0.0, AR1_SIGMA, N)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = x[i - 1] * AR1_RHO + e[i]
    x /= 100.0
    if limit is not None:
        x *= np.clip((limit - np.arange(N)) / (AR1_FADE / DT), 0.0, 1.0)
    return x + 1.0


def disturbance_signals(N, dt, rng, limit=None):
    """Ramp-hold-ramp events per channel, with an active flag. Channels overlap freely;
    events within one channel stay disjoint. All finish before the settling tail."""
    limit = N if limit is None else limit
    D, flags = {}, {}
    for name, cfg in FAULTS.items():
        sig = np.full(N, cfg["baseline"])
        flag = np.zeros(N)
        n = cfg["count"]
        durs = np.round(rng.uniform(*cfg["duration"], size=n) / dt).astype(int)
        slack = limit - durs.sum() - MIN_GAP * (n + 1)
        gaps = np.floor(rng.dirichlet(np.ones(n + 1)) * slack).astype(int) + MIN_GAP

        pos = gaps[0]
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
            pos += dur + gaps[i + 1]

        if name in AR1_CHANNELS:
            sig *= ar1(N, rng, limit)
        D[name], flags[name] = sig, flag
    return D, flags


def level_correction(U, D, start):
    """Unwind the level drift the Q_F events integrated in, with one outlet-valve pulse.

    The mirrored valve pairs cancel their own contribution, so the whole drift is the Q_F
    excess area, which is known from the signal before anything is simulated.
    """
    qmax = plant.Cv1 * np.sqrt(plant.dP)
    excess = float(np.sum(D[:, plant.DIST_NAMES.index("Q_F")] - FAULTS["Q_F"]["baseline"]) * DT)
    dl = excess / (qmax * LEVEL_CORRECTION)
    k = int(round(LEVEL_CORRECTION / DT))
    U[start:start + k, 0] = np.clip(VALVE_CENTRE + dl, 0.0, 1.0)


def write_h5(path, t, channels):
    """Write one (N, 1) dataset per channel, overwriting in place."""
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=t.reshape(-1, 1), chunks=True, compression="gzip")
        for name, values in channels.items():
            f.create_dataset(name, data=np.asarray(values).reshape(-1, 1),
                             chunks=True, compression="gzip")
        f.attrs["dt"] = DT * DOWNSAMPLE


def main():
    rng = np.random.default_rng(SEED)
    N = int(round(TOTAL_DURATION / DT))
    settle_start = N - int(round(SETTLE_DURATION / DT))

    U = np.column_stack([valve_cmd_signal(N, DT, rng, settle_start),
                         coolant_cmd_signal(N, DT, rng, settle_start)])
    dist, flags = disturbance_signals(N, DT, rng, settle_start)
    D = np.column_stack([dist[n] for n in plant.DIST_NAMES])
    level_correction(U, D, settle_start)

    _, Z = plant.simulate(plant.NOMINAL_STATE, U, D)
    Z = Z[:N]                                 # z[k] is the state seen when u[k] is applied

    # The reactor is bistable and nothing brings it back once it extinguishes, so a run that
    # goes cold is not a usable dataset.
    cold = np.mean(Z[:, 0] > EXTINCTION_C_A)
    if cold > 0:
        raise RuntimeError(f"Reactor extinguished for {cold:.1%} of the run.")

    t = np.arange(N) * DT
    sl = slice(None, None, DOWNSAMPLE)
    os.makedirs(OUT_DIR, exist_ok=True)
    write_h5(os.path.join(OUT_DIR, FILE_DATA), t[sl],
             dict(zip(plant.INPUT_NAMES, U[sl].T))
             | dict(zip(plant.DIST_NAMES, D[sl].T))
             | dict(zip(plant.OUTPUT_NAMES, Z[sl].T)))
    write_h5(os.path.join(OUT_DIR, FILE_EVENTS), t[sl],
             {f"{n}_active": flags[n][sl] for n in plant.DIST_NAMES})


if __name__ == "__main__":
    main()
