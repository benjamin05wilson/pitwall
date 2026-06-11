"""Live, in-race Bayesian tyre-degradation updater.

A recursive Bayesian linear regression (a Kalman filter) over the state

    z = [a, b]   with   corrected_lap_time = a + b * tyre_age

where ``a`` is fresh-tyre pace and ``b`` the live degradation rate (s/lap). Each
completed green lap, the raw lap time is fuel-corrected and folded in; the filter
returns the *posterior mean and covariance*, i.e. a calibrated belief about how
the current tyre is degrading **right now**, with credible bands — not a point
estimate. This is the component the strategy literature singles out as missing
from deep-learning tyre models ("lacks interpretability and explicit uncertainty
quantification — crucial in operational race environments").

Why a Kalman filter rather than a static fit:
* it is exact, O(1) per lap, and runs comfortably in real time;
* the process noise ``Q`` lets the degradation belief *adapt* as the track
  rubbers in or temperatures shift (a static regression cannot);
* innovation gating makes it robust to traffic/mistake laps and yields a natural
  **anomaly score**;
* the covariance propagates into pit-lap forecasts with honest uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.params import FuelModel


@dataclass
class TyreBelief:
    fresh_pace: float          # a (s), fuel-corrected
    deg_rate: float            # b (s/lap)
    cov: np.ndarray            # 2x2 posterior covariance

    @property
    def deg_rate_std(self) -> float:
        return float(np.sqrt(max(0.0, self.cov[1, 1])))

    def credible_deg(self, z: float = 1.645) -> tuple[float, float]:
        """Symmetric credible interval for the degradation rate (default 90%)."""
        s = self.deg_rate_std
        return (self.deg_rate - z * s, self.deg_rate + z * s)


@dataclass
class LapUpdate:
    lap: int
    tyre_age: int
    observed: float            # raw observed lap time (s)
    predicted: float           # filter's pre-update prediction (s)
    innovation: float          # observed - predicted
    innovation_z: float        # standardised (for anomaly detection)
    anomaly: bool
    belief: TyreBelief


class KalmanTyreModel:
    """Recursive Bayesian estimator of [fresh pace, degradation rate] per stint."""

    def __init__(
        self,
        n_laps: int,
        *,
        prior_pace: float = 95.0,
        prior_deg: float = 0.05,
        prior_var_pace: float = 4.0,
        prior_var_deg: float = 0.02,
        meas_var: float = 0.09,      # R: per-lap measurement noise (~0.3 s sd)
        drift_pace: float = 1e-3,    # Q[a]: track-evolution drift
        drift_deg: float = 5e-5,     # Q[b]: degradation can change as temps move
        anomaly_z: float = 3.0,      # |innovation z| above this => anomaly
        gate_z: float = 4.0,         # robustly down-weight updates beyond this
        fuel: FuelModel | None = None,
    ):
        self.n_laps = n_laps
        self.R = float(meas_var)
        self.Q = np.diag([drift_pace, drift_deg])
        self.anomaly_z = anomaly_z
        self.gate_z = gate_z
        self.fuel = fuel or FuelModel()
        self._prior = (prior_pace, prior_deg, prior_var_pace, prior_var_deg)
        self.history: list[LapUpdate] = []
        self.reset(prior_pace, prior_deg)

    def reset(self, prior_pace: float | None = None, prior_deg: float | None = None) -> None:
        """Start a fresh stint: re-seed the mean and re-inflate covariance."""
        p, d, vp, vd = self._prior
        self.z = np.array([prior_pace if prior_pace is not None else p,
                           prior_deg if prior_deg is not None else d], dtype=float)
        self.P = np.diag([vp, vd]).astype(float)

    # ------------------------------------------------------------------ #
    def _fuel_correct(self, lap: int, raw_lap_time: float) -> float:
        return raw_lap_time - self.fuel.penalty(lap, self.n_laps)

    def update(self, lap: int, tyre_age: int, raw_lap_time: float) -> LapUpdate:
        """Fold one completed green lap into the belief and return diagnostics."""
        H = np.array([1.0, float(tyre_age)])
        # Predict (random-walk state, add process noise).
        P_pred = self.P + self.Q
        y = self._fuel_correct(lap, raw_lap_time)
        y_hat = float(H @ self.z)
        S = float(H @ P_pred @ H + self.R)
        innov = y - y_hat
        innov_z = innov / np.sqrt(S)

        # Robust gating: inflate measurement noise for outlier laps so a single
        # traffic/mistake lap cannot wrench the degradation estimate.
        R_eff = self.R
        if abs(innov_z) > self.gate_z:
            R_eff = self.R * (abs(innov_z) / self.gate_z) ** 2
            S = float(H @ P_pred @ H + R_eff)

        K = (P_pred @ H) / S
        self.z = self.z + K * innov
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        belief = TyreBelief(float(self.z[0]), float(self.z[1]), self.P.copy())
        upd = LapUpdate(lap, tyre_age, raw_lap_time, y_hat + self.fuel.penalty(lap, self.n_laps),
                        innov, float(innov_z), abs(innov_z) > self.anomaly_z, belief)
        self.history.append(upd)
        return upd

    # ------------------------------------------------------------------ #
    @property
    def belief(self) -> TyreBelief:
        return TyreBelief(float(self.z[0]), float(self.z[1]), self.P.copy())

    def predict_corrected(self, tyre_age: int) -> tuple[float, float]:
        """Mean and sd of fuel-corrected lap time at a given tyre age."""
        H = np.array([1.0, float(tyre_age)])
        mean = float(H @ self.z)
        var = float(H @ self.P @ H + self.R)
        return mean, np.sqrt(var)

    def forecast_stint(self, from_age: int, n_ahead: int, at_lap: int) -> dict:
        """Forecast raw lap times for the next ``n_ahead`` laps with 90% bands."""
        ages = np.arange(from_age, from_age + n_ahead)
        means, los, his = [], [], []
        for i, age in enumerate(ages):
            m, sd = self.predict_corrected(int(age))
            fuel = self.fuel.penalty(at_lap + i, self.n_laps)
            means.append(m + fuel)
            los.append(m + fuel - 1.645 * sd)
            his.append(m + fuel + 1.645 * sd)
        return {"age": ages, "mean": np.array(means),
                "lo90": np.array(los), "hi90": np.array(his)}
