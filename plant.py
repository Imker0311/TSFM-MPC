"""True CSTR plant, open loop, as a discrete zero-order-hold stepper.

Dynamics, parameters and the nominal operating point are taken verbatim from
CSTR-Simulation.ipynb. The PI/cascade controllers are removed: the two valves are
driven directly by commanded positions. Valve actuator lag is kept, so the actual
position (states l, l_c) trails the command.

State  y = [C_A, T, T_C, h, l, l_c]
Input  u = [valve_cmd, coolant_valve_cmd]            (commanded valve positions, 0-1)
Dist   d = [E_R, U_Ac, T_F, C_F, T_CF, Q_F]
Output z = [C_A, T, T_C, h, Q, Qc]
"""

import numpy as np

# ============================== CONFIG ==============================
# --- timesteps -------------------------------------------------------
DT_OUTER = 1.0       # s, outer sample/control interval (inputs held constant across it)
DT_INNER = 0.5       # s, internal resolution: RK4 substep, or max_step cap for LSODA
# "lsoda" (adaptive, stiff-capable) or "rk4" (fixed step). Default lsoda: an ignition
# transient on this reactor rises hundreds of K in well under a second and fixed-step RK4
# blows up on it, while lsoda costs about the same for a run of this length.
INTEGRATOR = "lsoda"

# --- physical parameters (CSTR-Simulation.ipynb) ---------------------
A        = 0.1666       # m^2,          tank cross-sectional area
k_0      = 7.2e10 / 60  # 1/s,          Arrhenius pre-exponential
dH       = -5e4         # J/mol,        heat of reaction
rho_Cp   = 239          # J/(L K),      reactor contents
rhoc_Cpc = 4175         # J/(L K),      coolant
V_c      = 10           # L,            jacket volume
Cv1      = 0.4714       # L/(s Psi^0.5), outlet valve coefficient
Cv2      = 0.1          # L/(s Psi^0.5), coolant valve coefficient
dP       = 50.0         # Psi,          outlet valve pressure drop (fixed)
dPc      = 25.0         # Psi,          coolant valve pressure drop (fixed)
TAU_VALVE = 2.0         # s,            actuator lag, both valves

# Tank limits. Open loop the level is a pure integrator, so these act as the
# physical empty/overflow stops and keep 1/h finite if excitation runs long.
H_MIN = 0.05            # m
H_MAX = 2.0             # m

# CSTR-Simulation.ipynb writes the species/energy balances as (Q_F*x_F - Q*x)/V, dropping
# the accumulation term x*(Q_F-Q)/V that appears whenever V is not constant. Under the
# original level PI controller Q ~= Q_F, so it never mattered. Open loop it does: the energy
# balance stops conserving enthalpy as soon as Q != Q_F, a closing valve then heats the
# reactor by ~50 K and the mirrored opening quenches it onto the cold branch - an artifact,
# not physics. It also lets C_A exceed C_F. True adds the term back (default, required for a
# usable open-loop ground truth); False reproduces the notebook bit for bit.
VARIABLE_VOLUME_DILUTION = True

# --- nominal operating point (CSTR-Simulation.ipynb initial conditions) ---
NOMINAL_STATE = np.array([0.0453, 401.7, 345.2, 0.6, 0.5, 0.5])
NOMINAL_DIST  = np.array([8750.0, 5e4 / 60, 320.0, 1.0, 300.0, 100 / 60])
NOMINAL_INPUT = np.array([0.5, 0.5])
# ====================================================================

STATE_NAMES  = ["C_A", "T", "T_C", "h", "l", "l_c"]
INPUT_NAMES  = ["valve_cmd", "coolant_valve_cmd"]
DIST_NAMES   = ["E_R", "U_Ac", "T_F", "C_F", "T_CF", "Q_F"]
OUTPUT_NAMES = ["C_A", "T", "T_C", "h", "Q", "Qc"]

_K1 = Cv1 * np.sqrt(dP)     # L/s at fully open outlet valve
_K2 = Cv2 * np.sqrt(dPc)    # L/s at fully open coolant valve


def flows(y):
    """(Q, Qc) in L/s from the ACTUAL valve positions and the fixed pressure drops."""
    return _K1 * y[4], _K2 * y[5]


def outputs(y):
    """Measured outputs z = [C_A, T, T_C, h, Q, Qc]."""
    Q, Qc = flows(y)
    return np.array([y[0], y[1], y[2], y[3], Q, Qc])


def derivatives(y, u, d):
    C_A, T, T_C, h, l, l_c = y
    E_R, U_Ac, T_F, C_F, T_CF, Q_F = d

    V = 1000.0 * A * max(h, H_MIN)          # L, liquid volume
    Q, Qc = _K1 * l, _K2 * l_c
    r = k_0 * np.exp(-E_R / T) * C_A        # mol/(L s)

    acc = (Q_F - Q) if VARIABLE_VOLUME_DILUTION else 0.0

    return np.array([
        (Q_F * C_F - Q * C_A - acc * C_A) / V - r,
        r * (-dH) / rho_Cp + (Q_F * T_F - Q * T - acc * T) / V + U_Ac * (T_C - T) / (rho_Cp * V),
        Qc * (T_CF - T_C) / V_c + U_Ac * (T - T_C) / (rhoc_Cpc * V_c),
        (Q_F - Q) / (1000.0 * A),
        (u[0] - l) / TAU_VALVE,
        (u[1] - l_c) / TAU_VALVE,
    ])


def step(y, u, d, dt=None, dt_inner=None, method=None):
    """Advance the plant one outer step with u and d held constant (ZOH)."""
    # Resolved here, not as default arguments, so editing the config block above still
    # takes effect after this module has been imported.
    dt = DT_OUTER if dt is None else dt
    dt_inner = DT_INNER if dt_inner is None else dt_inner
    method = INTEGRATOR if method is None else method

    y = np.asarray(y, dtype=float)
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    d = np.asarray(d, dtype=float)

    if method != "rk4":
        from scipy.integrate import solve_ivp
        sol = solve_ivp(lambda t, s: derivatives(s, u, d), (0.0, dt), y,
                        method="LSODA", rtol=1e-8, atol=1e-10, max_step=dt_inner)
        y = sol.y[:, -1]
    else:
        n = max(1, int(round(dt / dt_inner)))
        hs = dt / n
        for _ in range(n):
            k1 = derivatives(y, u, d)
            k2 = derivatives(y + 0.5 * hs * k1, u, d)
            k3 = derivatives(y + 0.5 * hs * k2, u, d)
            k4 = derivatives(y + hs * k3, u, d)
            y = y + (hs / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y[3] = min(max(y[3], H_MIN), H_MAX)

    y[3] = min(max(y[3], H_MIN), H_MAX)
    y[4] = min(max(y[4], 0.0), 1.0)
    y[5] = min(max(y[5], 0.0), 1.0)
    return y


def simulate(y0, U, D, dt=None, dt_inner=None, method=None):
    """Roll the plant over U[k], D[k] (both (N, ...)). Returns states (N+1, 6) and outputs (N+1, 6)."""
    N = len(U)
    Y = np.empty((N + 1, 6))
    Y[0] = y0
    for k in range(N):
        Y[k + 1] = step(Y[k], U[k], D[k], dt, dt_inner, method)
    Z = np.array([outputs(y) for y in Y])
    return Y, Z
