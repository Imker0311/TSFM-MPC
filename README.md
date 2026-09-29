# TSFM-MPC

Model predictive control of a simulated CSTR, using a time-series foundation model as the
controller's internal model instead of a first-principles model. Two models are wired in:
zero-shot TimesFM-3, and Chronos-2 fine-tuned on the reactor.

| File | What it does |
|---|---|
| `plant.py` | The true CSTR. Discrete stepper, ground truth for everything else. |
| `generate_context_data.py` | Open-loop excitation run -> `data/CSTR_ContextData.h5`. |
| `plot_context_data.py` | Diagnostic figures for that dataset. |
| `forecaster.py` | Zero-shot TimesFM-3 wrapper. `forecast()` and batched `forecast_batch()`. |
| `chronos_forecaster.py` | Chronos-2 wrapper, drop-in replacement for `forecaster.py`. |
| `context_manager.py` | Maintains the closed-loop context: protected block + pruned rolling tail. |
| `mpc.py` | The controller: config, objective, output-bias correction, CEM optimiser. |
| `run_mpc.py` | The closed loop with the true plant in it, plus instrumentation and figures. |
| `run_mpc_chronos.py` | The same loop driven by Chronos-2. |
| `generate_finetune_data.py` | Two independent excitation runs for fine-tuning. |
| `finetune_chronos_mpc.py` | LoRA fine-tune of Chronos-2 on the reactor. |
| `compare_finetune.py` | Zero-shot vs fine-tuned, on valve gains and held-out accuracy. |

```bash
python generate_context_data.py && python plot_context_data.py
python generate_finetune_data.py
python finetune_chronos_mpc.py
python compare_finetune.py
python run_mpc_chronos.py
```

Needs `timesfm` and `chronos-forecasting` plus a CUDA GPU. Developed against
`Masters-Forecasting/envs`.

### Figures in `data/`

| File | What it shows |
|---|---|
| `context_timeseries.png` | The whole excitation run: both valve commands, the six state channels, and one panel per disturbance. This is what the model is given as context. |
| `context_zoom.png` | The same plot over the first 60 minutes, so individual valve pulses and their state response are actually readable. |
| `context_coverage.png` | How well the excitation covers the input space: per-valve histograms, a joint 2-D occupancy map of the two valves, and a scatter of the `T`-`C_A` operating points visited. Gaps here are regions the model has no evidence for. |
| `finetune_comparison.png` | Zero-shot vs fine-tuned vs the true plant, as bars, for the end-of-horizon change caused by holding each valve at a clamp extreme. Left panel temperature, right panel level. Bar height against the black "true plant" bar is the gain error. |
| `mpc_chronos_h6_tracking.png` | The closed-loop run. `T` and `h` against setpoint (true and measured), both valve commands against their safety clamp, and `C_A` against the extinction threshold. Shaded bands are disturbances, which the controller does not see coming. |
| `mpc_chronos_h6_walltime.png` | Seconds to solve each control step against the 10 s interval. The line has to stay under the red one for the controller to be real-time. |
| `mpc_chronos_h6_prediction.png` | For three sampled steps, what the model predicted over the horizon with its 10-90% band, against what actually happened. The prediction is one step's plan held open loop while the real trace is re-planned every step, so they diverge by design. |

## The plant

Six states (`C_A`, `T`, `T_C`, `h`, and two valve positions), two manipulated valves, six
disturbance channels. Sampled at 10 s, which is also the control interval. Reactor time
constant is about 60 s.

Two properties shape every decision in this repo:

- **The level is a pure integrator.** `dh/dt = (Q_F - Q)/(1000 A)`, so any sustained flow
  imbalance drains or floods the tank. There is no steady state except at `Q = Q_F`.
- **The reactor is bistable.** It sits on the ignited branch, and open loop nothing brings it
  back once it extinguishes. The excitation generator aborts rather than write an
  extinguished dataset.

Two corrections were made against the original notebooks: the species and energy balances
have their accumulation term restored (it only matters once `Q != Q_F`), and the integrator is
adaptive LSODA rather than fixed-step RK4, which overflows on ignition transients.

## The excitation dataset

10,000 rows at 10 s. Outlet-valve steps are emitted as **mirrored pairs** about the
level-balance position so the tank returns to where it started; the coolant valve gets plain
random multi-steps. The run ends at the nominal operating point so the closed-loop tail can be
appended without a step change nothing caused.

## The controller

Every 10 s: measure noisily, sample candidate valve trajectories, score them all in one
batched forecast, apply the first move of the best one, warm-start the next step from it.

**Cost** = setpoint tracking on `T` and `h` + a penalty on valve movement. Both terms are
normalised by the channel's own spread, so one weight means the same thing in Kelvin and in
metres. The objective is a swappable function that receives the predicted quantiles as well as
the means, so economic or chance-constrained variants drop in without touching the loop.

**Optimiser** is CEM over the valve *increments* (not positions), so the rate limit is a box on
the sample itself and the population stays feasible. `N_ITER = 1` is random shooting.

**Key settings** (`mpc.py`):

| | value |
|---|---|
| control interval | 10 s |
| prediction horizon | 6 steps (60 s) |
| control horizon | 6 steps |
| outlet valve clamp | [0.462, 0.538] |
| coolant valve clamp | [0.352, 0.799] |
| rate limits | 0.010 and 0.050 per step |
| move penalty | 0.1 each |
| candidates / elites / iterations | 200 / 20 / 3 |

Three things are deliberate:

**Future disturbances are unknown.** Only the valve commands are known-future covariates. The
six disturbances are fed as measured up to the present and held at their last value across the
horizon. This is the honest real-plant assumption and it costs accuracy while a fault ramps.

**MV bounds are a safety clamp at the excited region, not [0, 1].** They are read off the
dataset, so the clamp is defined by where the model has evidence. Outside it the model is
guessing and the plant can extinguish.

**Prediction and control horizon are equal.** With `Hp > Hc` the moves past `Hc` are held, and
on an integrating level that unactionable tail dominates the cost — at `Hp=30/Hc=8`, a valve
left at its clamp edge on the last free move was charged ~167 mm of predicted drift against
tracking errors of ~4 mm. The optimiser therefore drove the final move to flow balance
whatever the level was doing, biasing the controller towards taking no action, and amplifying
measurement noise into valve movement. Scoring only actionable steps removes both.

## Fine-tuning

TimesFM-3 cannot be fine-tuned with what it ships: the PyTorch class is documented as
inference-only, there is no `fit`, trainer, loss or LoRA anywhere in the package, and
`decode()` is wrapped in `@torch.no_grad()`. The LoRA example it advertises belongs to TimesFM
2.5, a different model without native covariate support. Chronos-2 ships `fit()` with LoRA
built in, so it is what gets fine-tuned here; `forecaster.py` was always a swappable wrapper,
so the swap costs one function call and no changes to `mpc.py` or `run_mpc.py`.

**Data** is two independent excitation runs, not one run split in two — 200,000 rows to train
(seed 101) and 40,000 to validate (seed 203). A tail split would hold out the settling tail,
which is the quiet near-nominal regime the controller actually operates in, and anything that
happens once per run lands wholly on one side of a split.

**Both valve commands go in as covariates**, alongside the six disturbances. The earlier
Chronos fine-tune in `Masters-Forecasting` passed only the disturbances, because there the
valves were on PI loops; here they are the decision variables, so a fine-tune without them
would be blind to the only thing the controller can do.

**Weights** go to `Masters-Forecasting/chronos-2-mpc-finetuned-final/` (gitignored). To load
them from another machine, `chronos_forecaster.py` falls back to the private Hub repo
`HF_REPO` (`Imker0311/chronos-2-cstr-mpc-lora`); run `hf auth login` once per machine. To
upload new weights, set `PUSH_TO_HUB = True` in `finetune_chronos_mpc.py`, or upload the
folder directly with `hf upload`.

## Results

Four closed-loop runs, 450 steps each, same disturbance scenario, 600-row context, 100
candidates x 3 CEM iterations. One thing changed at a time.

| Model | Horizon | Move penalty | `T` RMSE | `h` RMSE | Real time |
|---|---|---|---|---|---|
| TimesFM-3 zero-shot | 30 / 8 | 0.1 | 0.436 K | 1.8 mm | **fail**, worst 34.9 s |
| Chronos-2 fine-tuned | 30 / 8 | 0.1 | 0.702 K | 5.4 mm | pass, worst 6.4 s |
| Chronos-2 fine-tuned | 30 / 8 | 2.0 | 0.455 K | 3.6 mm | pass, worst 6.5 s |
| **Chronos-2 fine-tuned** | **6 / 6** | 2.0 | **0.250 K** | 3.7 mm | pass, worst 6.3 s |

The clearest single number is the quiet-period error, measured before any disturbance arrives:
**0.89 K at `Hp=30/Hc=8`, 0.0008 K at `Hp=Hc=6`**. The controller now sits still when nothing
is wrong, and the startup transient that dominated every earlier run is gone.

Both valves stay inside the safety clamp throughout, use 17-19% of their range, and never hit
their rate limits. `C_A` stays around 0.037 mol/L against a 0.3 extinction threshold.

### What fine-tuning did and did not fix

Fine-tuning was aimed at a specific defect: the zero-shot models get the **sign** of every
valve response right and the **magnitude** badly wrong. It helped, and it did not fix it.

Mean absolute gain error, predicted vs true end-of-horizon change with a valve held at a clamp
extreme: **T 3.65 -> 2.54 K, h 0.117 -> 0.078 m**. Held-out forecast accuracy improved by about
half on `C_A` and `h`, and got slightly worse on `Qc`. The outlet-valve gains are still 2.5-3x
too small, which is the channel the level loop depends on.

**Context length turned out to be the bigger lever.** Same fine-tuned model, same test, only
the context changed:

| context | gain error, T | gain error, h |
|---|---|---|
| 600 rows | 2.543 K | 0.0779 m |
| 2,048 rows | **1.169 K** | **0.0363 m** |

Halving the gain error by showing the model more history is a larger effect than the whole
fine-tune produced at fixed context. But 2,048 rows does not fit the control interval at three
CEM iterations (20.9 s/step); `2048 x 100 candidates x 1 iteration` does, at 6.97 s, and has
not been tried.

## Open items

- **The fine-tune is undertrained.** Validation loss was 0.0636 after 2,000 steps and 0.06365
  after three. More steps, a higher learning rate, or `finetune_mode="full"` are all untried.
- **The training data is excitation data, not teaching data.** Its net outlet-valve push over
  200,000 rows is ~0, because the mirrored-pair design undoes every excursion by construction.
  A model trained on it learns that outlet-valve moves have no lasting effect on the level,
  which is exactly the gain error measured above. Only 21% of the rows are a clean single
  disturbance with the valve still; 30% have the valve moving and a disturbance at once.
- **A level setpoint change has never been run.** Everything so far is regulation at a fixed
  setpoint, which is the easier task and the one that hides a weak level gain.
- **The move penalty default is 0.1 and the evidence says it should be higher.** With a perfect
  predictor and no disturbances, weight 10 gave better tracking *and* seven times less valve
  motion than 0.1.
- **Uncertainty is not calibrated** for this regime — the 10-90 band is ~3.5 K wide against
  actual variation of about +/-1 K, which matters for the planned chance-constraint work.
