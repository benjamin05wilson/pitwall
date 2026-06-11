# 🏁 pitwall — an F1 race-strategy engine

**The pit wall, in a box.** A complete race-strategy decision system: physics
models *calibrated on real F1 data*, a Monte-Carlo race simulator, strategy
optimisation under uncertainty, a **learned surrogate** that turns a multi-second
Monte-Carlo sweep into a **sub-second, full-strategy-space what-if**, and a
**live Bayesian tyre updater** that re-estimates degradation lap-by-lap and
re-optimises the moment a Safety Car deploys.

Built as a working demonstration of the kind of system a modern F1 strategy group
runs — from public data, end to end.

---

## Why this exists

This project targets the work of an **AI / modelling engineer in race strategy**:
the place where machine learning, physics, optimisation and real-time
decision-making meet. Every component maps to a real strategic question.

| What a strategist needs | What pitwall does | Module |
|---|---|---|
| Pace, tyre & overtaking models | Fuel-corrected lap-time model + degradation curves + dirty-air/overtaking submodel | `models/` |
| Calibration on real data | Per-compound degradation fit on real races with driver fixed-effects + track-evolution | `data/calibrate.py` |
| Strategy optimisation | Deterministic DP for pit laps + robust Monte-Carlo selection under SC/traffic uncertainty | `optimize/` |
| Speed for real-time use | PyTorch **surrogate** of the simulator → whole strategy space in milliseconds | `surrogate/` |
| Live, in-race decisions | **Kalman** tyre-degradation filter with calibrated uncertainty + pit-window forecast | `live/` |
| Production-grade compute | **Rust** Monte-Carlo hot loop (PyO3), pure-Python fallback | `rust/`, `sim/native.py` |
| A tool strategists can run | **React + FastAPI web app** (and a Streamlit dashboard + CLI) | `web/`, `server.py`, `cli.py` |

---

## Results — validated on real data, not mocked

**Calibration recovers reality.** Fitting the tyre/pace model on three years of
real Bahrain GP data (2023–2025, ~2,900 clean green laps via the OpenF1 API):

- Degradation comes out at **0.10–0.12 s/lap** with the **soft degrading fastest**
  — correct for an abrasive, high-deg circuit. Fit RMSE ≈ **0.52 s** (honest, for
  pooled public data).
- The fixed-effects fit recovers the **real pecking order** straight from lap
  times: `VER < PER < LEC < ALO < SAI < HAM …` — i.e. the actual 2023 podium.
- It **changes the answer**: with generic priors the optimiser likes a one-stop on
  softs; **calibrated on real data it flips to a two-stop on the hard tyre with
  pit windows around laps 18 & 38** — the strategy the teams actually ran.

**The simulator reconstructs real races.** Fed each driver's *real grid, real
strategy and calibrated pace*, it reproduces the **2023 Bahrain GP** result with
**Spearman rank correlation 0.72**, mean **2.5 positions** error, and the **exact
top two (Verstappen, Pérez)**. Its one podium "miss" — predicting Leclerc P3 —
is the model being *right on pace*: Leclerc actually retired with an engine
failure, a reliability event no pace model can foresee. `pitwall backtest 2023
Bahrain bahrain`.

**Two independent data sources agree.** The degradation fit from FastF1 (the
official feed) and OpenF1 (community) matches to within **~0.005 s/lap** — a
cross-source robustness check, not a single-pipeline artefact.

**The surrogate is genuinely real-time.** Trained on Monte-Carlo outputs across
circuits and grid slots:

- **R² = 0.93**, mean error **1.0 finishing position** vs the simulator, podium
  probability within **4%**, with a calibrated uncertainty head.
- Scores **2,400 strategies in ~2 ms** (~1.1 M/s). A single strategy via
  Monte-Carlo takes ~0.8 s, so the surrogate is **~10⁵–10⁶× faster per strategy** —
  this is what makes sweeping the entire pit-lap × compound space *live* possible.

**The live filter beats the baselines.** One-step-ahead lap-time prediction on a
real stint: Kalman **RMSPE 0.37 s** vs persistence 0.43 s vs growing-window OLS
0.40 s — and unlike a point predictor it emits a **90% credible band** on the
degradation rate and flags anomalous laps.

> _Numbers are reproducible from the scripts below; calibration/live figures pull
> live from OpenF1 + Jolpica._

---

## Quickstart

```bash
pip install -e ".[all]"          # numpy/scipy/torch/sklearn/fastf1/streamlit/...
# (optional) build the Rust accelerator:
cd rust && maturin develop --release && cd ..

# list circuits with their calibrated priors
pitwall circuits

# recommend a strategy from P3 at Bahrain, optimised for podium odds
pitwall optimize bahrain --grid 3 --objective podium

# calibrate the tyre model on a real race and re-optimise on it
pitwall optimize bahrain --grid 3 --calibrate 2023 Bahrain

# backtest: does the simulator reproduce a real race result?
pitwall backtest 2023 Bahrain bahrain --pool-years 2023 2024 2025

# pool three years of real data into a calibration
pitwall calibrate 2025 Bahrain bahrain --years 2023 2024 2025 --save data/calibrated/bahrain.json

# watch the live Bayesian tyre estimate evolve over a real stint
pitwall live 2023 Bahrain

# build the surrogate (parallel data-gen → train → eval)
python scripts/build_surrogate.py

# launch the web app (React UI + FastAPI engine on http://localhost:8000)
./scripts/serve.sh            # production: builds the frontend, serves on one port
./scripts/dev.sh              # dev: FastAPI :8000 + Vite hot-reload :5173

# or the quick Streamlit dashboard
PYTHONPATH=src streamlit run app/dashboard.py
```

### Web app

A product-grade frontend (`web/`, React + TypeScript) on a FastAPI backend
(`pitwall/server.py`) that serves the engine: pick a circuit / grid slot / pace /
objective and get a live, **Rust-accelerated** robust recommendation
(~0.5 s), a millisecond surrogate **pit-lap × compound heatmap**, the
**risk/reward frontier**, a **safety-car counterfactual**, and the candidate
table — all updating as you drag the sliders. `pitwall-serve` runs it.

---

## How it works

**Lap-time master equation** (lap-discretized, per car):

```
t_lap = base_pace + k_fuel·fuel(lap) + tyre(compound, age) + ε
tyre(c, age) = k0[c] + k1[c]·age + k2[c]·age²   (+1.0 s cold out-lap penalty)
```

Defaults are grounded in published work (the TUM race-simulation, a Bayesian
state-space tyre study) and **re-fit per event** from real data — see
[`docs/MODELLING.md`](docs/MODELLING.md) for every equation, default, source and
confidence level.

**Simulator.** A multi-car, cumulative-time loop with grid-staggered starts (real
**track position**), stochastic **Safety Car / VSC / red-flag** events (a
categorical fit; red flags = a free tyre change), ghost-car field bunching,
reliability retirements, a dirty-air/overtaking model, and cheap stops under
neutralisations. Strategies are scored by their **outcome distribution** —
E[finish], P(win/podium/points), and a tail measure (CVaR) — using **common
random numbers** so candidates are compared on identical races.

**Optimiser.** A deterministic dynamic program places pit laps optimally for each
compound sequence (the FIA two-compound rule enforced); the shortlist is then
re-scored robustly under uncertainty and selected on a chosen objective, with a
risk/reward efficient frontier exposed for the strategist.

**Surrogate.** A heteroscedastic MLP (256-128-64) emulates the simulator's
outcome distribution from an encoded (context, strategy) vector, giving instant,
uncertainty-aware answers for every pit-lap/compound/stop-count combination.

**Live updater.** A Kalman filter over `[fresh-pace, degradation-rate]` folds in
each completed green lap (fuel-corrected), adapts as the track evolves, gates out
traffic/mistake laps, resets at pit stops, and projects a **Bayesian pit window**.

---

## What real team data would add

This is built on **public** timing data, which is a proxy for the hundreds of
proprietary sensor channels a team actually has. The architecture is designed so
swapping public data for real telemetry only *improves* it: tyre-temperature and
load channels sharpen the degradation model, FP long-runs calibrate per-event
coefficients, and the live filter ingests a richer observation vector. The
honest residual is stated everywhere (calibration RMSE ≈ 0.5 s); nothing is
overclaimed.

## Data sources

- **FastF1** (official live-timing feed) — primary calibration backend: per-lap
  `TrackStatus` (exact green-lap filtering), `TyreLife`, compounds, sector times,
  trap speeds, weather, classified results.
- **OpenF1** (`api.openf1.org`) — compound-labelled stints + lap durations;
  automatic fallback if the official feed is unreachable, and a cross-check.
- **Jolpica-Ergast** (`api.jolpi.ca`) — results, lap times, pit-stop durations.

## Layout

```
src/pitwall/
  models/      lap-time, tyres, fuel, pit, safety-car, overtaking + per-circuit library
  sim/         Monte-Carlo race simulator (+ Rust accelerator wrapper)
  optimize/    deterministic DP + robust selection
  surrogate/   PyTorch emulator (features, dataset, model, train)
  live/        Kalman tyre updater, pit advisor, online benchmark
  data/        OpenF1 / Jolpica clients + calibration harness
  cli.py
rust/          Rust Monte-Carlo hot loop (PyO3/maturin)
app/           Streamlit strategy dashboard
tests/         pytest suite
docs/          MODELLING.md (build bible) + research provenance
```

_MIT licensed. Built as a portfolio project; not affiliated with any F1 team._
