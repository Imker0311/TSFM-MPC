"""True CSTR plant as a discrete zero-order-hold stepper.

State  y = [C_A, T, T_C, h, l, l_c]
Input  u = [valve_cmd, coolant_valve_cmd]
Dist   d = [E_R, U_Ac, T_F, C_F, T_CF, Q_F]
Output z = [C_A, T, T_C, h, Q, Qc]
"""

import numpy as np

# ============================== CONFIG ==============================
DT_OUTER = 1.0          # s, outer step; inputs held constant across it
DT_INNER = 0.5          # s, RK4 substep or LSODA max_step
INTEGRATOR = "lsoda"    # rk4 overflows on ignition transients

A        = 0.1666       # m^2, tank cross-section
k_0      = 7.2e10 / 60  # 1/s
dH       = -5e4         # J/mol
rho_Cp   = 239          # J/(L K)
rhoc_Cpc = 4175         # J/(L K)
V_c      = 10           # L, jacket volume
Cv1      = 0.4714       # L/(s Psi^0.5), outlet valve
Cv2      = 0.1          # L/(s Psi^0.5), coolant valve
dP       = 50.0         # Psi
dPc      = 25.0         # Psi
TAU_VALVE = 2.0         # s, actuator lag

H_MIN, H_MAX = 0.05, 2.0    # m, empty/overflow stops

# Keeps the accumulation term in the balances; without it enthalpy is not conserved once
# Q != Q_F. False reproduces the original notebook.
VARIABLE_VOLUME_DILUTION = True

NOMINAL_STATE = np.array([0.0453, 401.7, 345.2, 0.6, 0.5, 0.5])
NOMINAL_DIST  = np.array([8750.0, 5e4 / 60, 320.0, 1.0, 300.0, 100 / 60])
# ====================================================================

INPUT_NAMES  = ["valve_cmd", "coolant_valve_cmd"]
DIST_NAMES   = ["E_R", "U_Ac", "T_F", "C_F", "T_CF", "Q_F"]
OUTPUT_NAMES = ["C_A", "T", "T_C", "h", "Q", "Qc"]

_K1 = Cv1 * np.sqrt(dP)
_K2 = Cv2 * np.sqrt(dPc)


def outputs(y):
    """Measured outputs from the state; flows come from the actual valve positions."""
    return np.array([y[0], y[1], y[2], y[3], _K1 * y[4], _K2 * y[5]])


def derivatives(y, u, d):
    """Species, energy, level and two valve actuator ODEs."""
    C_A, T, T_C, h, l, l_c = y
    E_R, U_Ac, T_F, C_F, T_CF, Q_F = d

    V = 1000.0 * A * max(h, H_MIN)
    Q, Qc = _K1 * l, _K2 * l_c
    r = k_0 * np.exp(-E_R / T) * C_A
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
    """Advance the plant one outer step with u and d held constant."""
    # Resolved here so edits to the config block take effect after import.
    dt = DT_OUTER if dt is None else dt
    dt_inner = DT_INNER if dt_inner is None else dt_inner
    method = INTEGRATOR if method is None else method

    y = np.asarray(y, dtype=float)
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)

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
            y[3] = np.clip(y[3], H_MIN, H_MAX)

    y[3] = np.clip(y[3], H_MIN, H_MAX)
    y[4:6] = np.clip(y[4:6], 0.0, 1.0)
    return y


def simulate(y0, U, D, dt=None, dt_inner=None, method=None):
    """Roll the plant over input and disturbance sequences."""
    Y = np.empty((len(U) + 1, 6))
    Y[0] = y0
    for k in range(len(U)):
        Y[k + 1] = step(Y[k], U[k], D[k], dt, dt_inner, method)
    return Y, np.array([outputs(y) for y in Y])
