"""Benchmark the live updater's one-step-ahead lap-time prediction.

Compares the Kalman tyre filter against two baselines on a real stint:
* **persistence** — next lap = this lap (the naive bar);
* **growing-window OLS** — refit a static linear deg model on all laps seen.

Reported as RMSPE (root-mean-square prediction error, s). The published
state-space tyre model reaches ~1.08 s RMSPE vs ARIMA's ~1.52 s; this online
filter is the operational, real-time analogue.
"""

from __future__ import annotations

import numpy as np

from ..models.params import FuelModel
from .updater import KalmanTyreModel


def online_prediction_benchmark(
    lap: np.ndarray, tyre_age: np.ndarray, lap_time: np.ndarray,
    *, n_laps: int, fuel: FuelModel | None = None, warmup: int = 3,
    prior_pace: float | None = None,
) -> dict:
    """Run all methods one-step-ahead over a stint; return RMSPE per method."""
    fuel = fuel or FuelModel()
    lap = np.asarray(lap, dtype=int)
    tyre_age = np.asarray(tyre_age, dtype=int)
    lap_time = np.asarray(lap_time, dtype=float)
    n = len(lap_time)
    if n < warmup + 2:
        return {}

    kf = KalmanTyreModel(
        n_laps=n_laps, fuel=fuel,
        prior_pace=prior_pace if prior_pace is not None else float(lap_time[0]),
        prior_deg=0.05,
    )
    err_kf, err_naive, err_ols = [], [], []
    for i in range(n):
        kf.update(int(lap[i]), int(tyre_age[i]), float(lap_time[i]))
        if i < warmup or i + 1 >= n:
            continue
        # One-step-ahead predictions for lap i+1.
        nxt_age, nxt_lap, actual = int(tyre_age[i + 1]), int(lap[i + 1]), float(lap_time[i + 1])
        m, _ = kf.predict_corrected(nxt_age)
        pred_kf = m + fuel.penalty(nxt_lap, n_laps)
        err_kf.append(pred_kf - actual)
        err_naive.append(float(lap_time[i]) - actual)
        # Growing-window OLS on fuel-corrected laps 0..i.
        y = lap_time[: i + 1] - np.array([fuel.penalty(int(l), n_laps) for l in lap[: i + 1]])
        A = np.column_stack([np.ones(i + 1), tyre_age[: i + 1]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred_ols = coef[0] + coef[1] * nxt_age + fuel.penalty(nxt_lap, n_laps)
        err_ols.append(pred_ols - actual)

    def rmspe(e):
        return float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")

    return {
        "n_predictions": len(err_kf),
        "kalman_rmspe": rmspe(err_kf),
        "persistence_rmspe": rmspe(err_naive),
        "ols_rmspe": rmspe(err_ols),
        "final_deg_rate": float(kf.z[1]),
        "final_deg_std": kf.belief.deg_rate_std,
    }
