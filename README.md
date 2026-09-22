# TSFM-MPC

Zero-shot TimesFM-3 MPC of a CSTR. Stage 1: the true plant model, and the open-loop
excitation dataset that seeds the TSFM context.

| File | What it is |
|---|---|
| `plant.py` | True CSTR as an importable module. Discrete zero-order-hold stepper, reused later as ground truth in the MPC loop. |
| `generate_context_data.py` | Open-loop excitation run -> `data/CSTR_ContextData.h5` (10,000 rows) + `data/CSTR_ContextEvents.h5`. |
| `plot_context_data.py` | Diagnostic figures -> `data/context_timeseries.png`, `context_zoom.png`, `context_coverage.png`. |
| `forecaster.py` | Zero-shot TimesFM-3 wrapper. Loads once; `forecast(...)` returns `{col: {mean, var, quantiles}}` with all 9 quantiles retained. |
| `evaluate_forecast.py` | Forecast vs true CSTR at valve steps, plotted per state. |
| `context_manager.py` | Maintains the closed-loop context: fixed protected block + growing live tail, pruned at near-nominal cut points. |
| `test_context_manager.py` | Drives the manager off `plant.py` alone (no forecaster, no MPC) -> `data/context_manager_test.png`. |
| `CSTR-*.ipynb` | Reference notebooks from the previous project (dynamics, fault generator, downsampling, TimesFM-3 call pattern). |

```bash
python generate_context_data.py && python plot_context_data.py
python evaluate_forecast.py
```

TimesFM-3 needs the `timesfm3` package (`pip install "timesfm[torch]"`); this repo was run
against `/Users/imkerhoogenhout/Code/Masters-Forecasting/envs`.

**10,000 rows at a 10 s sample interval (27.8 h).** Covariates are all eight process inputs -
`valve_cmd`, `coolant_valve_cmd`, `E_R`, `U_Ac`, `T_F`, `C_F`, `T_CF`, `Q_F`. Targets are the six
states `C_A`, `T`, `T_C`, `h`, `Q`, `Qc`. Per-event active flags go to `CSTR_ContextEvents.h5`
for plotting only.

**The run ends at the nominal operating point.** The closed-loop tail is appended straight onto
the last row, so if the dataset ended anywhere else the join would be a state step change with
no input behind it. The last `SETTLE_DURATION` (2,000 s = 200 rows) is therefore quiet: no
events, both valves at nominal, the AR1 measurement drift faded back to 1.0, and only complete
mirrored valve pairs emitted beforehand so none is left hanging. The last row lands on nominal
to within 0.000 channel spreads on all six states, and the final 158 rows all qualify as
near-nominal at a 0.01 tolerance.

`h` is the one state a quiet tail does not bring home - it is a pure integrator, so at the
level-balance valve position `dh/dt = 0` and the level simply stays wherever it drifted to. The
mirrored valve pairs cancel their own contribution exactly whatever `Q_F` is doing, which leaves
the `Q_F` events as the only source of drift, and their area is known from the generated signal
before anything is simulated. `level_correction()` unwinds it with a single outlet-valve pulse
(~1.7% of travel over 300 s, at the head of the settling tail).

## Two things that changed versus the notebooks

Both are consequences of removing the PI/cascade controllers. Neither shows up in closed loop,
which is why the notebooks could ignore them.

**1. The species/energy balances were missing their accumulation term.**
`CSTR-Simulation.ipynb` writes them as `(Q_F*x_F - Q*x)/V`, which is only correct when `V` is
constant. The level PI controller held `Q ~= Q_F`, so it never mattered. Open loop it does: the
energy balance stops conserving enthalpy the moment `Q != Q_F`, so closing the outlet valve
*heats* the reactor by ~50 K and the mirrored opening then quenches it onto the cold branch,
and `C_A` can exceed `C_F`. `plant.VARIABLE_VOLUME_DILUTION = True` (the default) adds the term
back; set it `False` to reproduce the notebook bit for bit.

**2. Fixed-step RK4 is not enough for this plant.** An ignition transient rises several hundred
K in well under a second; RK4 at a 0.5 s substep overflows to `inf` where the true solution is a
bounded ~515 K spike. The default integrator is therefore adaptive LSODA
(`plant.INTEGRATOR`), which costs about the same over a run of this length (~1.7 s for 6 h).
The outer interface is still a fixed discrete `dt` with inputs held constant.

## Why the excitation is shaped the way it is

Open loop this reactor is awkward in two specific ways, and the excitation config is built
around them rather than tuned by hand:

- **The level is a pure integrator.** Any sustained `Q != Q_F` empties or floods the tank within
  a couple of minutes. The outlet valve is therefore excited as *mirrored pairs* about the
  level-balance position, whose flow areas cancel exactly, with a per-pair level budget
  (`H_EXCURSION`). That budget, divided by the shortest hold the 10 s sample interval can
  resolve, is what caps the outlet valve's usable range - it is a physical limit, not a choice.
- **The reactor sits on the ignited branch of a bistable pair,** and nothing pulls it back once
  it extinguishes. The per-channel disturbance magnitudes in `FAULTS` are set to a fraction of
  the sustained deviation at which the ignited steady state folds, measured by continuation at
  the worst coolant valve position; each fold value is in a comment on its line.
  `generate_context_data.py` aborts rather than write an extinguished dataset.

## Stage 2: does TimesFM-3 predict the CSTR's response?

`evaluate_forecast.py` makes the comparison directly, with no differencing. It finds rows late
in the dataset where a valve command steps, gives the model everything before that row plus the
true future inputs, and plots what it predicts for the six states against what the real CSTR did
under exactly those inputs. Disturbances are free to be active at the same time; that is part of
the test.

The model gets **all eight process inputs** as past-and-future covariates - both valve commands
and all six disturbances - so it has the same information the mathematical model has.

```bash
python evaluate_forecast.py    # writes data/forecast_case1..4.png
```

Typical result over four valve steps past row 6000, as RMSE over the 10 min horizon divided by
each channel's spread across the whole dataset. **These numbers predate the settling-tail
regeneration of the dataset and have not been re-measured** - rerun `evaluate_forecast.py` in the
TimesFM env to refresh them:

| case | step | C_A | T | T_C | h | Q | Qc |
|---|---|---|---|---|---|---|---|
| 1 | `coolant_valve_cmd` +0.056 | 0.026 | 0.044 | 0.015 | 0.039 | 0.008 | 0.021 |
| 2 | `valve_cmd` -0.043 | 0.054 | 0.043 | 0.032 | 0.074 | 0.113 | 0.020 |
| 3 | `valve_cmd` +0.042 | 0.075 | 0.041 | 0.015 | 0.203 | 0.339 | 0.007 |
| 4 | `valve_cmd` -0.040 | 0.008 | 0.021 | 0.017 | 0.050 | 0.174 | 0.009 |

## Why the outlet valve steps are small

`h` is a pure integrator: `dh/dt = (Q_F - Q)/(1000 A)`. A held outlet-valve offset therefore
drains or floods the tank - there is no steady state except at `Q = Q_F`. Step size and hold
length trade directly against each other through the level budget:

| outlet step | time before `h` moves 0.20 m |
|---|---|
| 1% | 16.7 min |
| 3% | 5.6 min |
| 10% | 1.7 min |

So outlet-valve moves are step-hold-return pulses of 150-450 s (2.5-7.5 reactor time constants,
enough to settle) at 1-4%, and the level swings a clearly visible 0.12 m. The coolant valve has
no integrator, so it gets genuine steps held 10-40 min anywhere in 0.35-0.80. Pushing the outlet
valve harder than this extinguishes the reactor - the generator's guard aborts when it happens,
and it fired repeatedly while this was being tuned.

## Stage 3: keeping the context alive in closed loop

`context_manager.py` holds what the forecaster is given each control step:

```
[ protected generated block ][ live rolling tail ]
```

The protected block is a fixed slice of `CSTR_ContextData.h5` and is never pruned - it carries
the valve->state dynamics a well-behaved closed loop stops exciting. The tail is what actually
happened, one row per control step, and it has to be cut back before it outgrows the window.

**Where to cut.** Deleting an arbitrary span splices two unrelated states together and hands the
model a step change no input caused. So every tail index whose *recent window* sits at the
nominal operating point is logged - a window, not an instant, so the short-term trajectory has
to match too, not just the position - and a prune deletes the span between the two oldest logged
indices. Both cut ends then hold the same state moving the same way. Distance is RMS deviation
from nominal over (window x target channels), each channel divided by its own spread, so one
tolerance is meaningful across mol/L, K and m at once.

Guards: at most `MAX_PRUNE_FRACTION` of the tail per prune, at least `MIN_PRUNE_SPAN` rows to be
worth a splice, the newest `KEEP_RECENT` rows never eligible, and if nothing matches at `TOL` the
tolerance widens by `TOL_GROWTH` up to `TOL_MAX` - and then the prune is **skipped with a
warning** rather than cutting a mismatched span.

```bash
python test_context_manager.py    # writes data/context_manager_test.png
```

Over 1,600 control steps with the cap set to 1,200 rows, four prunes fire and the worst
discontinuity any splice introduces is 0.13 channel spreads (0.9 K on `T`) - under the largest
single-step move the drive itself makes.

**The protected -> live boundary is a splice too**, and it exists from the first control step.
It is only ever as good as where the protected block ends, which is why the excitation run now
finishes at nominal (above). `select_protected_block()` picks the slice that *ends* nearest
nominal; against the current dataset it takes the last 400 rows and the seam is **0.000 spreads
on every channel**. A naive mid-run slice at row 4400 would join 1.70 spreads out instead.

Because that boundary is now an exact match, it is also the oldest and cleanest cut point
available, so prunes anchor there and the tail stays one contiguous recent span - nothing
accumulates in front of it. `test_context_manager.py` covers that path, the interior-cut path
(scenario 3 deliberately hands it a block that does not end at nominal) and the refusal path
(scenario 2, where the loop never returns to nominal and the prune is skipped with a warning).
