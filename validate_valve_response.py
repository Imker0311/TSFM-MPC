"""Validation gate: does zero-shot TimesFM-3 reproduce the plant's valve -> state dynamics?

Nothing here is tuned to make the model look good. The model sees only what MPC will give
it: past targets + past/known-future valve commands. It never sees the disturbances.

Method, per probe:
  baseline   forecast the 6 states over the horizon under the ACTUAL future valve commands
  perturbed  identical context and identical everything else, one valve command offset by a
             sustained step across the horizon
  response   perturbed - baseline, which cancels whatever the unseen disturbances are doing
  truth      the same two valve trajectories pushed through plant.py from the exact state

Ground truth is exact, not approximated: the 1 s signals behind data/CSTR_ContextData.h5 are
regenerated from generate_context_data.py's seed and re-simulated, which reproduces the saved
10 s series bit for bit (asserted at startup).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant
import forecaster as F
import generate_context_data as g

# ============================== CONFIG ==============================
HORIZON = 30            # samples = 5 min at the 10 s sample interval
# None = use every sample before t0 (bounded by the model's 15360). Measured over four
# operating points, fidelity improves monotonically with context and has not saturated by
# 1400 samples: mean gain error falls 0.40 -> 0.17 -> 0.07 for context 128 -> 512 -> 1400.
# A fixed window would also cap where the probes can sit, so probes use all the history they
# have, which is what MPC will have at run time too.
CONTEXT = None
SUB = g.DOWNSAMPLE      # plant steps of DT per dataset sample

# Operating points, as dataset sample indices. Spread across the run so the probes land on
# different C_A / T / h conditions rather than repeating one.
OPERATING_POINTS = [600, 1100, 1600]

# Two step sizes, both directions. They answer different questions and the plant makes the
# distinction necessary: a sustained +0.05 outlet offset drains ~0.29 m of level over the
# horizon, which shortens the residence time enough to tip the reactor off the ignited branch
# at some operating points. You cannot read a linear gain across a bifurcation, so the small
# step measures local fidelity and the large step tests whether the model sees the cliff.
SMALL_STEP = 0.01       # local small-signal probe
LARGE_STEP = 0.05       # large-signal probe; tips the plant at some operating points
VALVE_STEPS = [SMALL_STEP, LARGE_STEP]

# A trial is classified "tipping" from GROUND TRUTH, not from the model: if the perturbed
# plant trajectory's C_A crosses this, it has left the ignited branch (task-1 extinction
# threshold is 0.30). Tipping trials are excluded from gain/tau and scored separately.
EXTINCTION_WARN = 0.25

# A state counts as "affected" by a valve only if the true response clears this fraction of
# that channel's own variability over the context window. Below it, gain and time constant
# are meaningless and are reported as "-" rather than scored.
AFFECTED_FRAC = 0.10

ACCURACY_SLICES = [600, 1000, 1400, 1800]   # held-out t0 for the open-loop accuracy check
OUT_DIR = "data"
# ====================================================================

STATES = F.TARGET_COLUMNS
VALVES = F.COVARIATE_COLUMNS


def build_truth():
    """Regenerate the 1 s inputs and exact plant trajectory behind the saved dataset."""
    rng = np.random.default_rng(g.SEED)
    N = int(round(g.TOTAL_DURATION / g.DT))
    u_valve, _ = g.valve_cmd_signal(N, g.DT, rng)
    u_cool, _ = g.coolant_cmd_signal(N, g.DT, rng)
    U1 = np.column_stack([u_valve, u_cool])
    dist, _, _ = g.disturbance_signals(N, g.DT, rng)
    D1 = np.column_stack([dist[n] for n in plant.DIST_NAMES])
    Y1, Z1 = plant.simulate(plant.NOMINAL_STATE, U1, D1)
    return U1, D1, Y1[:N], Z1[:N]


def context_slice(df, t0):
    """History the model gets at t0: everything before it, or the last CONTEXT samples."""
    return df.iloc[max(0, t0 - CONTEXT):t0] if CONTEXT else df.iloc[:t0]


def plant_response(Y1, U1, D1, t0, horizon, valve=None, step=0.0):
    """Exact plant outputs over the horizon, optionally with a sustained offset on one valve.

    Index 0 is the state at t0 itself, so it is unaffected by any command applied from t0 on -
    the response necessarily starts at index 1. Gains are therefore read at the horizon end.
    """
    i0 = t0 * SUB
    U = U1.copy()
    if valve is not None:
        j = VALVES.index(valve)
        end = i0 + (horizon - 1) * SUB
        U[i0:end, j] = np.clip(U[i0:end, j] + step, 0.0, 1.0)

    y = Y1[i0].copy()
    out = [plant.outputs(y)]
    for k in range(horizon - 1):
        for s in range(SUB):
            i = i0 + k * SUB + s
            y = plant.step(y, U[i], D1[i])
        out.append(plant.outputs(y))
    return np.array(out)


def model_response(df, t0, horizon, valve=None, step=0.0):
    """Forecast means over the horizon, optionally with the same sustained valve offset."""
    context_df = context_slice(df, t0)
    future = df.iloc[t0:t0 + horizon][VALVES].to_numpy(dtype=float).copy()
    if valve is not None:
        j = VALVES.index(valve)
        future[:, j] = np.clip(future[:, j] + step, 0.0, 1.0)
    future_df = F.make_future_df(context_df, horizon, future)
    out = F.forecast(context_df, future_df, STATES, VALVES, horizon)
    return np.column_stack([out[c]["mean"] for c in STATES])


def t63(delta, dt):
    """Time to reach 63% of the final response. For an integrator this just reports a
    fraction of the window rather than a settling time - h is scored by ramp_ratio instead."""
    final = delta[-1]
    if abs(final) < 1e-15:
        return np.nan
    s = np.sign(final)
    hit = np.where(s * delta >= 0.63 * s * final)[0]
    return hit[0] * dt if len(hit) else np.nan


def ramp_ratio(delta):
    """delta(end) / delta(mid). ~2 for a pure integrator, ~1 for a settled first-order."""
    mid = delta[len(delta) // 2]
    return delta[-1] / mid if abs(mid) > 1e-15 else np.nan


def run_valve_probes(df, U1, D1, Y1, dt):
    """Every (operating point, valve, direction, step size) probe, one baseline per point.

    Each trial is tagged `tipping` when the TRUE perturbed trajectory leaves the ignited
    branch, so the classification never depends on what the model predicted.
    """
    trials = []
    for t0 in OPERATING_POINTS:
        base_pred = model_response(df, t0, HORIZON)
        base_true = plant_response(Y1, U1, D1, t0, HORIZON)
        ctx_std = context_slice(df, t0)[STATES].to_numpy().std(axis=0)
        for valve in VALVES:
            for size in VALVE_STEPS:
                for step in (+size, -size):
                    pred = model_response(df, t0, HORIZON, valve, step)
                    true = plant_response(Y1, U1, D1, t0, HORIZON, valve, step)
                    trials.append(dict(
                        t0=t0, valve=valve, step=step, size=size,
                        d_pred=pred - base_pred, d_true=true - base_true, ctx_std=ctx_std,
                        tipping=bool(true[:, STATES.index("C_A")].max() > EXTINCTION_WARN)))
                    print(f"  {valve:>18s} {step:+.3f} at t0={t0}"
                          f"{'   TIPPING' if trials[-1]['tipping'] else ''}", flush=True)
    return trials


def onset(delta, dt, frac=0.1):
    """Samples until the response first reaches `frac` of its final value."""
    final = delta[-1]
    if abs(final) < 1e-15:
        return np.nan
    hit = np.where(np.sign(final) * delta >= frac * abs(final))[0]
    return hit[0] if len(hit) else np.nan


def verdict_table(trials, dt):
    """One row per (valve, state), scored on LOCAL trials only."""
    rows = []
    for valve in VALVES:
        sub = [t for t in trials if t["valve"] == valve and not t["tipping"]]
        for i, state in enumerate(STATES):
            g, sg, tt, tp, ot, op, sh, sp = [], [], [], [], [], [], [], []
            for t in sub:
                dtrue, dpred = t["d_true"][:, i], t["d_pred"][:, i]
                floor = AFFECTED_FRAC * t["ctx_std"][i]
                if abs(dtrue[-1]) <= floor:
                    # Plant barely moves: instead of a meaningless gain, record how much the
                    # model moves anyway - a spurious response here is a real MPC defect.
                    sp.append(np.abs(dpred).max() / max(t["ctx_std"][i], 1e-15))
                    continue
                sg.append(np.sign(dpred[-1]) == np.sign(dtrue[-1]))
                g.append(dpred[-1] / dtrue[-1])
                tt.append(t63(dtrue, dt)); tp.append(t63(dpred, dt))
                ot.append(onset(dtrue, dt)); op.append(onset(dpred, dt))
                sh.append(np.sqrt(np.mean((dpred / dtrue[-1] - dtrue / dtrue[-1]) ** 2)))
            def med(a):
                a = [x for x in a if not np.isnan(x)]
                return float(np.median(a)) if a else np.nan
            within = float(np.mean([abs(x - 1) <= 0.5 for x in g])) if g else np.nan
            rows.append(dict(valve=valve, state=state, n=len(sub), n_aff=len(g),
                             sign_ok=int(np.sum(sg)) if sg else 0,
                             gain=med(g), gain_lo=min(g) if g else np.nan,
                             gain_hi=max(g) if g else np.nan, within=within,
                             tau_true=med(tt), tau_pred=med(tp),
                             onset_true=med(ot), onset_pred=med(op),
                             ramp_true=med([ramp_ratio(t["d_true"][:, i]) for t in sub]),
                             ramp_pred=med([ramp_ratio(t["d_pred"][:, i]) for t in sub]),
                             shape=med(sh), spurious=med(sp)))
    return rows


def tipping_summary(trials):
    """What the model predicted where the plant actually left the ignited branch."""
    out = []
    ca = STATES.index("C_A")
    tv = STATES.index("T")
    for t in [x for x in trials if x["tipping"]]:
        out.append(dict(t0=t["t0"], valve=t["valve"], step=t["step"],
                        dCA_true=t["d_true"][-1, ca], dCA_pred=t["d_pred"][-1, ca],
                        dT_true=t["d_true"][-1, tv], dT_pred=t["d_pred"][-1, tv]))
    return out


def accuracy_check(df):
    """Open-loop HORIZON-step forecast under the true future valves, versus the record."""
    truth, preds = [], []
    for t0 in ACCURACY_SLICES:
        preds.append(model_response(df, t0, HORIZON))
        truth.append(df.iloc[t0:t0 + HORIZON][STATES].to_numpy(dtype=float))
        print(f"  accuracy slice t0={t0}", flush=True)
    return preds, truth


def accuracy_metrics(preds, truth):
    """RMSE, and the anchoring offset at the start of the horizon. A persistent offset is
    fixable downstream by an offset-free / bias-updating MPC; wrong dynamics is not."""
    rmse = {s: float(np.sqrt(np.mean([(p[:, i] - t[:, i]) ** 2 for p, t in zip(preds, truth)])))
            for i, s in enumerate(STATES)}
    bias = {s: float(np.mean([abs(p[0, i] - t[0, i]) for p, t in zip(preds, truth)]))
            for i, s in enumerate(STATES)}
    return rmse, bias


# --- pass criteria, fixed before the run --------------------------------------
GAIN_BAND = (0.5, 2.0)      # median predicted/true gain at horizon end, local trials
GAIN_WITHIN = 0.75          # ...AND this fraction of individual trials must land within +/-50%.
                            # The median alone hides dispersion: a pair can sit at 0.96 median
                            # while half its trials are off by more than 2x, which is not a gain
                            # you can hand to a controller.
TAU_BAND = (0.5, 2.0)       # predicted/true 63% time, where the true response is resolvable
RAMP_MIN = 1.5              # h must still be climbing at horizon end (pure integrator = 2)
SPURIOUS_MAX = 0.25         # a valve the plant ignores must move the model < this x ctx std
SIGN_FRAC = 0.9             # fraction of LOCAL trials that must get the direction right. Scored
                            # per trial, not per (valve, state): one bad trial at one operating
                            # point should not fail three rows that are otherwise clean.


def _scored(rows):
    return [r for r in rows if r["n_aff"] > 0]


def _tau_scorable(r, dt):
    """A time constant below one sample cannot be measured at this sample rate."""
    return r["n_aff"] > 0 and r["tau_true"] > dt


def print_report(rows, tips, rmse, bias, df, dt):
    line = "-" * 98
    print("\n" + line)
    print(f"VALVE RESPONSE GATE   steps +/-{SMALL_STEP} and +/-{LARGE_STEP}, horizon {HORIZON} "
          f"samples ({HORIZON * dt / 60:.0f} min), t0 {OPERATING_POINTS}, "
          f"context {'all history before t0' if CONTEXT is None else CONTEXT}")
    print(f"LOCAL TRIALS ONLY - {len(tips)} tipping trials held out below")
    print(line)

    for valve in VALVES:
        print(f"\n{valve}")
        print(f"  {'state':<6} {'affected':>9} {'sign':>7} {'gain (pred/true)':>25} {'+/-50%':>7} "
              f"{'t63 true/pred':>16} {'shape':>7}")
        for r in [r for r in rows if r["valve"] == valve]:
            if r["n_aff"] == 0:
                sp = "-" if np.isnan(r["spurious"]) else f"{r['spurious']:.2f}x std"
                tag = "" if (np.isnan(r["spurious"]) or r["spurious"] <= SPURIOUS_MAX) else "  <-- SPURIOUS"
                print(f"  {r['state']:<6} {'0/' + str(r['n']):>9} {'-':>7} "
                      f"{'plant ignores this valve':>25} {'-':>7} {sp:>16} {'-':>7}{tag}")
                continue
            gain = f"{r['gain']:+.2f} [{r['gain_lo']:+.2f},{r['gain_hi']:+.2f}]"
            tau = ("immediate" if not _tau_scorable(r, dt)
                   else f"{r['tau_true']:.0f}/{r['tau_pred']:.0f} s")
            bad = []
            if r["sign_ok"] != r["n_aff"]:
                bad.append("SIGN")
            if not (GAIN_BAND[0] <= abs(r["gain"]) <= GAIN_BAND[1]):
                bad.append("GAIN")
            elif r["within"] < GAIN_WITHIN:
                bad.append("SCATTER")
            if _tau_scorable(r, dt) and not (TAU_BAND[0] <= r["tau_pred"] / r["tau_true"] <= TAU_BAND[1]):
                bad.append("TAU")
            mark = ("  <-- " + ",".join(bad)) if bad else ""
            print(f"  {r['state']:<6} {str(r['n_aff']) + '/' + str(r['n']):>9} "
                  f"{str(r['sign_ok']) + '/' + str(r['n_aff']):>7} {gain:>25} "
                  f"{r['within']:>6.0%} {tau:>16} {r['shape']:>7.2f}{mark}")

    print("\n" + line)
    print("LEVEL INTEGRATOR PROBE   sustained outlet-valve offset -> h must RAMP, not settle")
    print(line)
    h = next(r for r in rows if r["valve"] == "valve_cmd" and r["state"] == "h")
    if h["n_aff"] == 0:
        print("  h never cleared the affected threshold - integrator untested, treat as FAIL")
    else:
        print(f"  ramp ratio d(end)/d(mid):  true {h['ramp_true']:.2f}   predicted {h['ramp_pred']:.2f}"
              f"   (pure integrator 2.00, settled first-order 1.00)")
        print(f"  gain {h['gain']:+.2f}, shape error {h['shape']:.2f}")
        print(f"  {'RAMPS - integrating behaviour reproduced' if h['ramp_pred'] >= RAMP_MIN else 'DOES NOT RAMP - model settles where the plant integrates'}")

    print("\n" + line)
    print(f"NONLINEAR / TIPPING TRIALS   plant left the ignited branch (true C_A > {EXTINCTION_WARN})")
    print(line)
    if not tips:
        print("  none - every probe stayed on the ignited branch")
    else:
        print(f"  {'t0':>5} {'valve':>18} {'step':>7} {'dC_A true':>10} {'dC_A pred':>10} "
              f"{'dT true':>9} {'dT pred':>9} {'captured':>9}")
        for t in tips:
            frac = t["dCA_pred"] / t["dCA_true"] if abs(t["dCA_true"]) > 1e-12 else np.nan
            print(f"  {t['t0']:>5} {t['valve']:>18} {t['step']:>+7.3f} {t['dCA_true']:>10.4f} "
                  f"{t['dCA_pred']:>10.4f} {t['dT_true']:>9.2f} {t['dT_pred']:>9.2f} {frac:>8.0%}")

    print("\n" + line)
    print(f"OPEN-LOOP ACCURACY   {HORIZON}-step forecast under the true future valves, t0 {ACCURACY_SLICES}")
    print(line)
    print(f"  {'state':<6} {'RMSE':>12} {'series std':>12} {'RMSE/std':>10} {'anchor/std':>11}")
    for st in STATES:
        sd = df[st].std()
        print(f"  {st:<6} {rmse[st]:>12.5g} {sd:>12.5g} {rmse[st] / sd:>10.2f} {bias[st] / sd:>11.2f}")
    print("  anchor = |forecast[0] - truth[0]|, i.e. offset at the start of the horizon")


def conditioning_table(trials, df):
    """How sensitive the plant itself is at each operating point.

    The small-signal probe is only interpretable where the plant responds smoothly. This
    measures that from ground truth: peak |dT| for a SMALL_STEP probe, and how far the true
    response overshoots its own endpoint (ringing >> 1 means oscillatory, near the fold).
    """
    ti = STATES.index("T")
    out = []
    for t0 in OPERATING_POINTS:
        sub = [t for t in trials if t["t0"] == t0 and t["size"] == SMALL_STEP]
        peak_t = max(np.abs(t["d_true"][:, ti]).max() for t in sub)
        peak_p = max(np.abs(t["d_pred"][:, ti]).max() for t in sub)
        ring = max(np.abs(t["d_true"][:, ti]).max() / max(abs(t["d_true"][-1, ti]), 1e-12)
                   for t in sub)
        ok = tot = 0
        for t in sub:
            for i in range(len(STATES)):
                if abs(t["d_true"][-1, i]) > AFFECTED_FRAC * t["ctx_std"][i]:
                    tot += 1
                    ok += int(np.sign(t["d_pred"][-1, i]) == np.sign(t["d_true"][-1, i]))
        out.append(dict(t0=t0, ctx=len(context_slice(df, t0)),
                        C_A=df["C_A"].iloc[t0], T=df["T"].iloc[t0],
                        peak_true=peak_t, peak_pred=peak_p, ring=ring, ok=ok, tot=tot))
    return out


def print_conditioning(cond):
    line = "-" * 98
    print("\n" + line)
    print(f"PLANT CONDITIONING BY OPERATING POINT   small step +/-{SMALL_STEP}")
    print(line)
    print(f"  {'t0':>5} {'context':>8} {'C_A':>8} {'T [K]':>8} {'peak |dT| true':>15} {'pred':>8} "
          f"{'ringing':>8} {'sign ok':>9}")
    for c in cond:
        note = "  <-- oscillatory, near the stability fold" if c["ring"] > 2.0 else ""
        sign = f"{c['ok']}/{c['tot']}"
        print(f"  {c['t0']:>5} {c['ctx']:>8} {c['C_A']:>8.4f} {c['T']:>8.1f} {c['peak_true']:>15.2f} "
              f"{c['peak_pred']:>8.2f} {c['ring']:>8.1f} {sign:>9}{note}")


def print_failing_detail(rows, trials, dt):
    """Every local trial behind a failing (valve, state) pair, so a bad sign or gain is
    visible as the individual trial that caused it rather than an aggregate."""
    bad = [r for r in _scored(rows)
           if r["sign_ok"] != r["n_aff"]
           or not (GAIN_BAND[0] <= abs(r["gain"]) <= GAIN_BAND[1])
           or (_tau_scorable(r, dt) and not (TAU_BAND[0] <= r["tau_pred"] / r["tau_true"] <= TAU_BAND[1]))]
    if not bad:
        return
    line = "-" * 98
    print("\n" + line)
    print("FAILING PAIRS - every local trial behind the aggregate")
    print(line)
    for r in bad:
        i = STATES.index(r["state"])
        print(f"\n  {r['valve']}/{r['state']}")
        print(f"    {'t0':>6} {'step':>7} {'d true':>12} {'d pred':>12} {'ratio':>8}")
        for t in trials:
            if t["valve"] != r["valve"] or t["tipping"]:
                continue
            dtrue, dpred = t["d_true"][-1, i], t["d_pred"][-1, i]
            if abs(dtrue) <= AFFECTED_FRAC * t["ctx_std"][i]:
                continue
            ratio = dpred / dtrue
            flag = "  <-- sign" if np.sign(dpred) != np.sign(dtrue) else ""
            print(f"    {t['t0']:>6} {t['step']:>+7.3f} {dtrue:>12.5g} {dpred:>12.5g} {ratio:>8.2f}{flag}")


def print_verdict(rows, tips, rmse, df, dt, cond):
    line = "=" * 98
    scored = _scored(rows)
    sign_ok = sum(r["sign_ok"] for r in scored)
    sign_tot = sum(r["n_aff"] for r in scored)
    sign_pass = sign_tot and (sign_ok / sign_tot) >= SIGN_FRAC
    sign_bad = [r for r in scored if r["sign_ok"] != r["n_aff"]]
    gain_bad = [r for r in scored if not (GAIN_BAND[0] <= abs(r["gain"]) <= GAIN_BAND[1])
                or r["within"] < GAIN_WITHIN]
    tau_rows = [r for r in scored if _tau_scorable(r, dt)]
    tau_bad = [r for r in tau_rows
               if not (TAU_BAND[0] <= r["tau_pred"] / r["tau_true"] <= TAU_BAND[1])]
    spur = [r for r in rows if r["n_aff"] == 0 and not np.isnan(r["spurious"])
            and r["spurious"] > SPURIOUS_MAX]
    h = next(r for r in rows if r["valve"] == "valve_cmd" and r["state"] == "h")
    ramp_ok = h["n_aff"] > 0 and h["ramp_pred"] >= RAMP_MIN
    tip_frac = [t["dCA_pred"] / t["dCA_true"] for t in tips if abs(t["dCA_true"]) > 1e-12]
    tip_ok = (not tips) or (np.median(np.abs(tip_frac)) >= 0.5)

    def mk(ok):
        return "PASS" if ok else "FAIL"

    def names(rs):
        return ", ".join(r["valve"] + "/" + r["state"] for r in rs)

    print("\n" + line)
    print("VERDICT")
    print(line)
    print(f"  sign        {mk(sign_pass):<5} {sign_ok}/{sign_tot} local trials get the direction right"
          + (f"   misses confined to: {names(sign_bad)}" if sign_bad else ""))
    print(f"  gain        {mk(not gain_bad):<5} {len(scored) - len(gain_bad)}/{len(scored)} pairs: median in "
          f"{GAIN_BAND} and >={GAIN_WITHIN:.0%} of trials within +/-50%"
          + (f"   off: {names(gain_bad)}" if gain_bad else ""))
    print(f"  dynamics    {mk(not tau_bad):<5} {len(tau_rows) - len(tau_bad)}/{len(tau_rows)} inside t63 ratio {TAU_BAND}"
          + (f"   off: {names(tau_bad)}" if tau_bad else ""))
    print(f"  integrator  {mk(ramp_ok):<5} h ramp ratio pred {h['ramp_pred']:.2f} vs true {h['ramp_true']:.2f} (need >= {RAMP_MIN})")
    print(f"  crosstalk   {mk(not spur):<5} " + (f"spurious response on {names(spur)}" if spur
          else f"valves the plant ignores move the model < {SPURIOUS_MAX:.2f} x std"))
    if tips:
        print(f"  bifurcatn   {mk(tip_ok):<5} {len(tips)} tipping trials, model captures "
              f"{np.median(np.abs(tip_frac)):.0%} of the true C_A excursion (need >= 50%)")
    worst = max(STATES, key=lambda s: rmse[s] / df[s].std())
    print(f"  accuracy     ---  worst channel {worst} at RMSE/std = {rmse[worst] / df[worst].std():.2f}")
    unstable = [c for c in cond if c["ring"] > 2.0]
    if unstable:
        print(f"  conditioning ---  {len(unstable)} of {len(cond)} operating points are oscillatory "
              f"near the fold (t0 {', '.join(str(c['t0']) for c in unstable)}); every sign miss is there")
    local_ok = sign_pass and not gain_bad and not tau_bad and ramp_ok and not spur
    print(line)
    if local_ok and tip_ok:
        print("  GO - zero-shot reproduces the valve dynamics; proceed to MPC")
    elif local_ok:
        print("  CONDITIONAL - local dynamics are right, but the model is blind to the")
        print("  bifurcation. Safe only if MPC is constrained to stay on the ignited branch.")
    else:
        print("  NO-GO - zero-shot does not reproduce the valve dynamics; stop and rethink")
    print(line)
    return local_ok, tip_ok


C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"


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


def plot_step_responses(trials, dt, size, name):
    """Predicted vs true step response per state, one column per valve."""
    t = np.arange(HORIZON) * dt / 60.0
    fig, axes = plt.subplots(len(STATES), len(VALVES), figsize=(11, 13),
                             sharex=True, facecolor=SURFACE)
    for col, valve in enumerate(VALVES):
        sub = [x for x in trials if x["valve"] == valve and x["step"] == size]
        for row, state in enumerate(STATES):
            ax = axes[row, col]
            for k, x in enumerate(sub):
                lbl = f"from row {x['t0']} ({x['t0'] * dt / 60:.0f} min in)" + (
                    " TIPPING" if x["tipping"] else "")
                ax.plot(t, x["d_true"][:, row], color=C[k % 3], lw=1.6,
                        label=f"plant {lbl}" if row == 0 else None)
                ax.plot(t, x["d_pred"][:, row], color=C[k % 3], lw=1.4, ls="--",
                        label=f"TimesFM {lbl}" if row == 0 else None)
            ax.axhline(0, color=GRID, lw=0.8)
            _ax(ax, f"change in {state}" if col == 0 else "",
                "minutes since the valve was moved" if row == len(STATES) - 1 else None)
        axes[0, col].set_title(f"{valve}  step {size:+.3f}", color=INK, fontsize=11, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2, labelcolor=MUTED)
    fig.suptitle("Effect of moving one valve: solid = real plant, dashed = model.  "
                 "x-axis is time SINCE the move, so the three runs (which start at different "
                 "points in the 6 h dataset) are overlaid deliberately.",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = f"{OUT_DIR}/{name}.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    print("wrote", p)


def plot_accuracy(df, preds, truth, dt):
    fig, axes = plt.subplots(len(STATES), len(ACCURACY_SLICES),
                             figsize=(14, 12), facecolor=SURFACE)
    t = np.arange(HORIZON) * dt / 60.0
    for col, (t0, p, tr) in enumerate(zip(ACCURACY_SLICES, preds, truth)):
        for row, state in enumerate(STATES):
            ax = axes[row, col]
            ax.plot(t, tr[:, row], color=INK, lw=1.6, label="plant" if row == 0 and col == 0 else None)
            ax.plot(t, p[:, row], color=C[0], lw=1.4, ls="--",
                    label="TimesFM-3" if row == 0 and col == 0 else None)
            _ax(ax, state if col == 0 else "", "time [min]" if row == len(STATES) - 1 else None)
        axes[0, col].set_title(f"t0 = {t0}", color=INK, fontsize=10, loc="left")
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=MUTED)
    fig.suptitle(f"Open-loop {HORIZON}-step forecast under the true future valve commands",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = f"{OUT_DIR}/validation_accuracy.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    print("wrote", p)


def main():
    df = F.load_dataframe()
    dt = df.attrs["dt"]

    need = CONTEXT or 2
    bad = [t for t in OPERATING_POINTS + ACCURACY_SLICES
           if t < need or t + HORIZON > len(df)]
    if bad:
        raise ValueError(f"t0 values {bad} need {need} samples before and HORIZON={HORIZON} "
                         f"after, within {len(df)} samples")

    U1, D1, Y1, Z1 = build_truth()
    assert np.array_equal(Z1[::SUB], df[STATES].to_numpy()), \
        "regenerated truth does not match the saved dataset - generator config changed?"
    print(f"ground truth regenerated and matches data/CSTR_ContextData.h5 exactly "
          f"({len(Z1)} plant steps -> {len(df)} samples)")

    print("\nvalve probes:")
    trials = run_valve_probes(df, U1, D1, Y1, dt)
    print("\nopen-loop accuracy:")
    preds, truth = accuracy_check(df)

    rows = verdict_table(trials, dt)
    tips = tipping_summary(trials)
    rmse, bias = accuracy_metrics(preds, truth)

    cond = conditioning_table(trials, df)
    print_report(rows, tips, rmse, bias, df, dt)
    print_conditioning(cond)
    print_failing_detail(rows, trials, dt)
    verdict = print_verdict(rows, tips, rmse, df, dt, cond)

    plot_step_responses(trials, dt, SMALL_STEP, "validation_step_small")
    plot_step_responses(trials, dt, LARGE_STEP, "validation_step_large")
    plot_accuracy(df, preds, truth, dt)
    return verdict


if __name__ == "__main__":
    main()
