> Historical design/interview notes. Numerical experimental claims below are unverified unless linked from the current [README evidence ledger](../README.md#evidence-ledger). See the [current limitations register](MODELLING_DECISIONS.md), which supersedes conflicting result and capability claims.

# pitwall — modelling build-bible

Every functional form, default value, source and confidence level behind the
engine. Defaults are **priors**; `pitwall.data.calibrate` re-fits the tyre/pace
model per event from real data. Raw research provenance (8 sourced sweeps) is in
[`research/research-findings.json`](research/research-findings.json).

Primary sources: Heilmeier et al., *A Race Simulation for Strategy Decisions in
Circuit Motorsports* (TUM, IEEE ITSC 2018) and its Monte-Carlo extension (Appl.
Sci. 2020); the TUMFTM/race-simulation parameter files; a Bayesian state-space
tyre study (arXiv:2512.00640, used for magnitudes only).

---

## 1. Lap-time master equation (lap-discretized, per car)

```
t_lap(L) = base_pace + k_fuel·fuel(L) + tyre(compound, age) + ε
```

- **base_pace** = `t_quali_ref + race_pace_gap + driver_delta`. `race_pace_gap`
  default **4.0 s** above pole (TUM ≈ 4.04). `driver_delta` is the car/driver
  pace offset (calibrated).
- **ε** ~ N(0, σ²), **σ = 0.6 s** default (driver range 0.46–0.86 s). *High conf.*

### Fuel
`fuel(L) = max(0, m_start − burn·(L−1))`, penalty `= k_fuel·fuel`.
- **k_fuel = 0.030 s/kg** (track range 0.019 short / 0.036 long). *High conf.*
  The 0.107 s/kg figure from an arXiv PDF extraction is ~3× too high — **rejected**.
- **m_start = 110 kg** (F1 max tank); **burn = m_start/n_laps**, floored at 1.6 kg/lap
  (TUM 1.55–2.0). ⇒ a full tank costs ≈ 3.3 s/lap at the start, decaying to 0.

### Tyres
`tyre(c, age) = k0[c] + k1[c]·age + k2[c]·age²`, with a **+1.0 s cold penalty** on
the out-lap (age 0; TUM `t_add_coldtires`). Linear is the robust default; the
quadratic term models the soft-compound cliff.

Ground-effect-era **defaults** (re-fit per event):

| Compound | k0 (offset vs medium) | k1 (s/lap) | k2 (s/lap²) |
|---|---|---|---|
| SOFT | −0.30 | 0.100 | 0.0011 |
| MEDIUM | 0.0 | 0.060 | 0.0006 |
| HARD | +0.30 | 0.035 | 0.0003 |

These racing-era priors make the **hard tyre and 2-stops genuinely competitive**
(soft is fastest fresh but degrades hard; hard is slow fresh but durable) rather
than always favouring a soft 1-stop. Compound pace deltas are kept modest (~0.3
s/lap) because real data shows the FIA target gaps are rarely realised.

Adjacent-compound step ≈ 0.5 s/lap (the FIA target gaps of S–H 2.2 s / M–H 1.2 s
are **not** realised in practice). Slopes 0.01–0.09 s/lap (TUM); softest cliff
compounds up to 0.18. *High conf. on ranges, must calibrate per event.*

Under neutralisations the tyre **ages fractionally**: ×0.25 (SC), ×0.5 (VSC).

---

## 2. Pit loss

On a stop lap add `PIT_LOSS[circuit]` (entry + standstill + exit, vs staying out).
Global default **22 s**; per-circuit Monaco 20 / Bahrain 23 / Monza 24 / Singapore 28
(range 18–30). Stationary time ≈ 2.4 s mean (only ~10% of the loss; world record
1.80 s). FIA **two-compound rule** enforced by the enumerator. Double-stack queue
penalty ≈ +2.5 s. *Medium conf.*

---

## 3. Safety-car / VSC (Heilmeier 2014–2019 categorical fit)

- **# SC phases** ~ categorical `[0.455, 0.413, 0.099, 0.033]` ⇒ P(≥1 SC) = 0.545,
  E = 0.71. Rescaled per circuit (Singapore ≈ 1.0, Baku ≈ 0.86, Monza/Monaco ≈ 0.27).
- **Start lap** ~ bucket weights `[0.364, 0.136, 0.136, 0.080, 0.193, 0.091]` over
  [lap 1, <20%, <40%, <60%, <80%, <100%] (lap-1 spike + late bump).
- **SC duration** (laps 1–10) ~ `[0, .182, .25, .227, .193, .057, .068, .023, 0, 0]`,
  mean ≈ 4. **VSC duration** (1–4) ~ `[.479, .396, .021, .104]`, mean ≈ 1.75.
- **Lap time** = green × **1.6** (SC) / **1.4** (VSC). **Cheap stop**: effective pit
  loss saves **50%** (SC) / **35%** (VSC). **Ghost-car bunching** under SC compresses
  the field to a ~1 s train; VSC preserves gaps.

**Red flags** (P ≈ 0.07/race): a full stoppage modelled as a *free tyre change*
(zero pit loss) plus field bunching — strategically the single biggest swing,
heavily rewarding cars that hadn't yet stopped. The simulator samples SC, VSC
**and** red-flag timelines per race so all three are priced into every
recommendation.

*High conf. on the SC/VSC fit; per-circuit P(SC), red-flag rate and recent-season
drift are medium conf.*

---

## 4. Overtaking / dirty air (TUM track-position model)

Per green lap, on cumulative time: a trailing car passes only if its pace
advantage clears the **circuit threshold** `t_gap_overtake`, else it is held
`min_follow_gap = 0.5 s` behind (track position emerges from this clamp).

Thresholds (s/lap advantage needed): Suzuka 1.26, Silverstone 1.35, Bahrain 1.38,
Monza 1.76, … Monaco/Singapore/Montreal 3.75 (≈ unpassable). **DRS** bonus −0.45 s
within a 1.0 s window (from lap 3, disabled 2 laps post-SC). Velocity modifier
−0.045 s per km/h of top-speed delta. Stochastic pass prob `p_overtake` 0.10–0.40.
Duel/loser penalties 0.3 s damp re-pass oscillation. *High conf.*

**Starting grid** is real: the race begins with cars staggered by ~1.6 s per grid
slot of cumulative time, so a car must actually fight past those ahead. A strong
car from pole finishes far better than from the back (≈5 positions at Bahrain),
and the gap widens at hard-to-overtake circuits — track position, modelled.

---

## 5. Optimisation

1. **Deterministic DP** places pit laps optimally for each compound sequence
   (degradation is convex ⇒ well-behaved); enumerate sequences with the FIA rule.
2. **Robust MC**: re-score the shortlist over ~10⁴ stochastic races (SC, traffic,
   reliability) with **common random numbers**. Select on the *outcome
   distribution* — E[pos], P(podium), CVaR tail — never the deterministic optimum.
3. **Real-time**: on an SC, re-evaluate from the current state with SC pit loss; for
   the sub-second undercut/overcut call, a fresh-vs-old delta lookup.

---

## 6. Surrogate

Heteroscedastic MLP (256-128-64, dropout 0.15, Adam 5e-4) emulating the MC
outcome distribution from an encoded (context, strategy) vector. Predicts mean
finishing position **+ log-variance** (Gaussian NLL) and podium/points
probabilities. Achieves R² ≈ 0.93 / MAE ≈ 1.0 position and scores the full
strategy space in milliseconds.

---

## 7. Live Bayesian tyre updater

Recursive Bayesian linear regression (Kalman filter) over state `z = [a, b]`,
`corrected_lap = a + b·age`. F = I; process noise Q lets the belief adapt as the
track rubbers in; R ≈ 0.1–0.3 s²; innovation gating makes it robust to traffic
laps and yields an anomaly score; reset (covariance re-inflation) at each pit.
Emits a 90% credible band on the degradation rate and a Bayesian pit window.
Benchmark target: beat ARIMA (RMSPE ≈ 1.08 vs 1.52 s); here it beats persistence
and growing-window OLS one-step-ahead on real stints.

---

## 8. Calibration methodology

Pull compound-labelled stints + lap durations (OpenF1) and pit durations
(Jolpica). Drop out/in/SC/outlier laps; **fuel-correct first**; then a single
**fixed-effects regression** with per-(race,driver) intercepts + per-race
track-evolution + **shared** per-compound `k0`/`k1`, fit by IRLS-Huber. Pooling
several race-years stabilises offsets and slopes that a single event leaves
confounded. Reports honest per-compound RMSE.

---

## 9. Open risks (calibrate / validate before trusting)

- Tyre coefficients are circuit- **and** season-specific — defaults must be re-fit;
  compound pace deltas are often statistically indistinct in clean data.
- SC/VSC probabilities are a 2014–2019 fit; recent seasons differ — scale per circuit.
- Dirty-air magnitude in the ground-effect era is published as downforce-%, not
  s/lap — treat as a tunable calibrated to overtake counts; sign flips by circuit.
- VSC can end with ~10–15 s warning, so a perfect-information sim *overvalues* VSC stops.
- Public timing is a proxy for proprietary telemetry — residual RMSE ≈ 0.5–1.5 s;
  proprietary sensors only improve the model. Don't overclaim.
