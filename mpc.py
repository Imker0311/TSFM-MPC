"""Receding-horizon MPC with a time-series foundation model as the internal model.

Each step samples candidate valve trajectories, scores them all in one batched forecast,
applies the first move of the best one, and warm-starts the next step from it.
"""

import time

import numpy as np

import forecaster as F

# ============================== CONFIG ==============================
DT = 10.0                   # s, control interval
# Equal horizons, so every scored step is one the controller can still act on. With HP > HC
# the held tail dominates the cost on the integrating level and biases it towards no action.
HP = 6                      # prediction horizon, steps
HC = 6                      # control horizon, steps

# Error is divided by the channel's spread before weighting, so one weight means the same in
# K and in m.
CONTROLLED = {
    "T": dict(setpoint=402.35, weight=1.0),     # K
    "h": dict(setpoint=0.60,   weight=1.0),     # m
}
MANIPULATED = ["valve_cmd", "coolant_valve_cmd"]

# Safety clamp: None reads the excited region off the dataset, which is where the model has
# evidence. The plant is bistable, so a move outside that band can extinguish it.
MV_BOUNDS = None
MV_RATE = {"valve_cmd": 0.010, "coolant_valve_cmd": 0.050}      # per control step
MOVE_WEIGHT = {"valve_cmd": 0.1, "coolant_valve_cmd": 0.1}      # penalty on MV movement

# Cost per control step is n_candidates x n_iter forecast rows. n_iter = 1 is random shooting.
N_CANDIDATES = 200
N_ELITES = 20
N_ITER = 3
INIT_STD = 0.5              # initial sampling std, as a fraction of the rate limit
MIN_STD = 0.05              # floor on the refitted std, stops elite collapse

BIAS_CORRECTION = True
BIAS_ALPHA = 0.3            # low-pass factor on the measured-minus-predicted residual

SEED = 11
# ====================================================================

TARGETS = F.TARGET_COLUMNS
COVARIATES = F.COVARIATE_COLUMNS
DISTURBANCES = F.DISTURBANCE_COLUMNS


def excited_region_bounds(df, columns):
    """Safety clamp: the range each MV actually covered in the dataset."""
    return {c: (float(df[c].min()), float(df[c].max())) for c in columns}


def tracking_cost(pred, moves, u_prev, cfg):
    """Setpoint tracking plus an MV move penalty, vectorised over candidates.

    Index 0 of the prediction is the present sample and is not controllable, so the cost
    starts at index 1. Quantiles are passed through so an economic or chance-constrained
    objective can replace this without the loop changing.
    """
    cost = np.zeros(len(moves))
    for col, spec in cfg.controlled.items():
        e = (pred[col]["mean"][:, 1:] - spec["setpoint"]) / cfg.scales[col]
        cost += spec["weight"] * (e ** 2).mean(axis=1)

    prev = np.broadcast_to(u_prev, (len(moves), len(cfg.manipulated)))[:, None, :]
    du = np.diff(np.concatenate([prev, moves], axis=1), axis=1) / cfg.mv_span
    return cost + (cfg.move_weight * du ** 2).mean(axis=1).sum(axis=1)


class OutputBias:
    """Additive per-CV output bias, added to the whole predicted horizon.

    The residual is taken against the raw prediction so the filter converges to the offset
    rather than chasing its own output.
    """

    def __init__(self, columns, alpha=BIAS_ALPHA, enabled=BIAS_CORRECTION):
        self.columns, self.alpha, self.enabled = list(columns), float(alpha), bool(enabled)
        self.bias = {c: 0.0 for c in self.columns}

    def update(self, measured, predicted):
        """Low-pass the measured-minus-predicted residual into the bias."""
        if not self.enabled or predicted is None:
            return
        for c in self.columns:
            r = float(measured[c]) - float(predicted[c])
            self.bias[c] = (1.0 - self.alpha) * self.bias[c] + self.alpha * r

    def apply(self, pred):
        """Add the current bias to the mean and every quantile."""
        if not self.enabled:
            return pred
        out = dict(pred)
        for c in self.columns:
            b = self.bias[c]
            out[c] = {"mean": pred[c]["mean"] + b, "var": pred[c]["var"],
                      "quantiles": {q: v + b for q, v in pred[c]["quantiles"].items()}}
        return out


class Config:
    """Resolved controller settings, with bounds and scales read from the dataset."""

    def __init__(self, df, dt=DT, hp=HP, hc=HC, mv_bounds=MV_BOUNDS, mv_rate=None,
                 move_weight=None, n_candidates=N_CANDIDATES, n_elites=N_ELITES,
                 n_iter=N_ITER, init_std=INIT_STD, min_std=MIN_STD, objective=tracking_cost,
                 bias_correction=BIAS_CORRECTION, bias_alpha=BIAS_ALPHA, seed=SEED):
        self.dt, self.hp, self.hc = float(dt), int(hp), int(hc)
        self.controlled = {k: dict(v) for k, v in CONTROLLED.items()}
        self.manipulated = list(MANIPULATED)
        self.targets, self.covariates = list(TARGETS), list(COVARIATES)
        self.disturbances = list(DISTURBANCES)

        self.bounds = dict(mv_bounds) if mv_bounds else excited_region_bounds(df, self.manipulated)
        self.scales = {c: float(df[c].std()) for c in self.controlled}
        rate, weight = mv_rate or MV_RATE, move_weight or MOVE_WEIGHT
        self.rate = np.array([rate[m] for m in self.manipulated])
        self.move_weight = np.array([weight[m] for m in self.manipulated])
        self.lo = np.array([self.bounds[m][0] for m in self.manipulated])
        self.hi = np.array([self.bounds[m][1] for m in self.manipulated])
        self.mv_span = self.hi - self.lo

        self.n_candidates, self.n_elites, self.n_iter = int(n_candidates), int(n_elites), int(n_iter)
        self.init_std, self.min_std = float(init_std), float(min_std)
        self.objective = objective
        self.bias_correction, self.bias_alpha = bool(bias_correction), float(bias_alpha)
        self.seed = seed


class MPC:
    """One CEM solve per call to step(); holds the warm start and the output bias."""

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.bias = OutputBias(cfg.controlled, cfg.bias_alpha, cfg.bias_correction)
        self.plan = np.zeros((cfg.hc, len(cfg.manipulated)))
        self._pending = None

    def _integrate(self, du, u_prev):
        """Turn increments into absolute MV positions, rate- and clamp-feasible.

        Decision variables are increments so the rate limit is a box on the sample itself and
        the population stays inside the feasible set instead of collapsing onto its boundary.
        """
        cfg = self.cfg
        u = np.empty((len(du), cfg.hp + 1, len(cfg.manipulated)))
        cur = np.broadcast_to(u_prev, (len(du), len(cfg.manipulated))).copy()
        for j in range(cfg.hc):
            cur = np.clip(cur + du[:, j], cfg.lo, cfg.hi)
            u[:, j] = cur
        u[:, cfg.hc:] = cur[:, None, :]
        return u

    def _future_covariates(self, u, d_hold):
        """Candidate valve trajectories plus disturbances held at their last measured value."""
        fut = np.empty((*u.shape[:2], len(self.cfg.covariates)))
        fut[:, :, :u.shape[2]] = u
        fut[:, :, u.shape[2]:] = np.asarray(d_hold, dtype=float)
        return fut

    def step(self, context_df, measured, d_hold, u_prev):
        """Solve for the next MV move.

        context_df ends at the previous sample, because the input chosen now belongs to a row
        that does not exist yet. The forecast spans HP+1 rows with index 0 on the present
        sample, which also supplies the residual for the bias update.
        """
        cfg = self.cfg
        self.bias.update(measured, self._pending)

        t0 = time.perf_counter()
        n_mv = len(cfg.manipulated)
        mean = self.plan.copy()
        std = np.broadcast_to(cfg.init_std * cfg.rate, (cfg.hc, n_mv)).copy()
        floor = cfg.min_std * cfg.rate

        best = None
        for _ in range(cfg.n_iter):
            du = np.clip(self.rng.normal(mean, std, (cfg.n_candidates, cfg.hc, n_mv)),
                         -cfg.rate, cfg.rate)
            du[0] = np.clip(mean, -cfg.rate, cfg.rate)   # incumbent always competes
            u = self._integrate(du, u_prev)
            raw = F.forecast_batch(context_df, self._future_covariates(u, d_hold),
                                   cfg.targets, cfg.covariates, cfg.hp + 1)
            cost = cfg.objective(self.bias.apply(raw), u[:, :cfg.hc, :], u_prev, cfg)

            order = np.argsort(cost)
            i = int(order[0])
            if best is None or cost[i] < best["cost"]:
                best = dict(cost=float(cost[i]), u=u[i], du=du[i],
                            pred={c: raw[c]["mean"][i] for c in cfg.targets},
                            band={c: (raw[c]["quantiles"][0.1][i], raw[c]["quantiles"][0.9][i])
                                  for c in cfg.controlled})
            elites = order[:cfg.n_elites]
            mean = du[elites].mean(axis=0)
            std = np.maximum(du[elites].std(axis=0), floor)

        self.plan = np.vstack([best["du"][1:], np.zeros((1, n_mv))])
        self._pending = {c: best["pred"][c][1] for c in cfg.controlled}

        return dict(u=best["u"][0], plan=best["u"], prediction=best["pred"], band=best["band"],
                    bias=dict(self.bias.bias), wall_clock=time.perf_counter() - t0)
