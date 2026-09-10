# Current modelling decisions and limitations

This register replaces generated interview-preparation prose that mixed implementation facts with unretained experimental claims. Historical versions remain in Git history. The [README evidence ledger](../README.md#evidence-ledger) is the source of truth for demonstrated outcomes. No real-data performance benchmark was regenerated during the readiness pass.

## Model continuity and calibration

`optimize(model, ...)` now uses the supplied focal model for both deterministic enumeration and stochastic scoring. Fitted tyre coefficients, thermal terms, fuel, pit, pace, overtaking and safety-car settings are retained. An explicit `focal_delta` replaces the model's driver offset; omission preserves it. The caller supplies the scenario bank, so it must sample that bank from the intended race-control model. Existing banks are not resampled secretly.

`build_field` uses generic circuit defaults and seeded rival pace offsets. It does not fit the rivals using the focal fit. Strategy Lab uses circuit defaults for both focal and rival models. CLI calibration is opt-in; ghost/backtest paths apply historical fits and priors separately. The API reports its model choice.

The retained Bahrain calibration summary is an in-sample residual fit, not held-out accuracy. Its complete generating command, environment and source-year selection are not retained. Pooled public data and driver fixed effects can confound tyre age, stint selection, track evolution and pace. Thermal coupling may be inert in shipped priors. Public pit transit measurements do not directly identify race-time loss relative to a neutralised field.

## Optimisation and simulation

The DP optimises stint boundaries for fixed compound sequences under an additive free-track model. Robust ranking evaluates a stop-count-diversified shortlist, not every strategy or adaptive policy. Default API budget is deliberately illustrative. A small probability difference is not decision confidence; common random numbers provide repeatable comparisons but variance reduction has not been measured here.

Traffic, overtaking, reliability, safety-car rates, red-flag resets and wet crossover curves are modelling assumptions. Red flags currently reset tyre age for every car for free, with optional scheduled compound changes. Tests establish that implementation, not regulatory fidelity. Fuel mass is indexed by lap: the safety-car fuel multiplier fields are currently inert. Double-stack and dirty-air degradation multiplier fields are also not simulated. Do not infer effects merely because a parameter exists.

Rust has different RNG streams from Python. The adapter falls back to Python for wet scenarios/non-slick strategies, forced retirements, nonzero pit-loss variance and dirty-air pace loss. Direct native calls reject unsupported settings. No native speed or cross-platform parity measurement is claimed by the offline demo. Python remains the reference.

## Historical estimation

All streaming replay events are labelled `historical-replay`; detecting a current OpenF1 session cannot relabel them. The stream is selected historical observations played at wall-clock pace. There is no current-lap ingestion implementation.

Kalman updates carry model covariance and gate unusual innovations. Their nominal intervals have not been validated for coverage. The retained synthetic plot demonstrates response to a planted anomalous lap; it is not a real-race benchmark. Earlier single-stint superiority claims and contradictory aggregate baseline comparisons lack retained evaluation manifests and are withdrawn, rather than replacing one unverified number with another.

Backtests and ghost comparisons use historical pace, strategies, grid and other same-race information. They are retrospective reconstructions, not prospective forecasts. DNF handling, condition stratification and data leakage must be audited before any predictive accuracy claim. No race gains or team deployment are demonstrated.

## Optional surrogate

No checkpoint is bundled. Missing Torch, missing weights or load failure make the heatmap unavailable while core recommendations remain usable. Request errors have visible UI states. The heatmap considers six ordered one-stop slick pairs over pit laps, not all stop counts or the entire strategy space.

Surrogate training code uses a random row split and the validation set for model selection and reporting. Context/circuit separation and a separate final test set remain future work; no training run was part of this pass. The uncertainty head is a relative learned indicator, not calibrated absolute confidence. Shape/probability-bound tests do not establish accuracy.

Previous surrogate R²/error/throughput, native speedup, cross-source agreement, reconstruction correlation and Kalman superiority numbers have no complete retained provenance and are not supported result headlines. Regeneration would need frozen inputs, source commit, environment, seeds, metric definitions, held-out protocol and raw outputs. A training or benchmark script alone is not that evidence.
