"""Calibration + live-updater tests on synthetic data with known ground truth.

These use no network — synthetic stints with planted degradation let us assert
the estimators actually recover the truth, not just run."""

import numpy as np
import pandas as pd
import pytest

from pitwall.data.calibrate import fit_compounds
from pitwall.live import KalmanTyreModel
from pitwall.models.params import FuelModel
from pitwall.types import Compound


def _synthetic_race(n_laps=57, seed=0):
    """Two drivers, planted compound degradation, real fuel burn-off + noise."""
    rng = np.random.default_rng(seed)
    fuel = FuelModel()
    truth = {Compound.SOFT: (0.0, 0.12), Compound.MEDIUM: (0.4, 0.06),
             Compound.HARD: (0.8, 0.03)}  # (k0 vs soft baseline-ish, k1)
    driver_base = {"AAA": 90.0, "BBB": 90.8}
    rows = []
    plan = {"AAA": [(Compound.SOFT, 1, 20), (Compound.HARD, 21, 57)],
            "BBB": [(Compound.MEDIUM, 1, 25), (Compound.HARD, 26, 57)]}
    for drv, stints in plan.items():
        for comp, l0, l1 in stints:
            k0, k1 = truth[comp]
            for lap in range(l0, l1 + 1):
                age = lap - l0
                lt = (driver_base[drv] + k0 + k1 * age
                      + fuel.penalty(lap, n_laps) + rng.normal(0, 0.05))
                rows.append({"driver": drv, "lap": lap, "compound": comp,
                             "tyre_age": age, "lap_time": lt})
    return pd.DataFrame(rows), truth, n_laps


def test_calibration_recovers_degradation():
    df, truth, n_laps = _synthetic_race()
    res = fit_compounds(df, n_laps=n_laps, track_evolution=False)
    assert res is not None
    for comp, (_, k1) in truth.items():
        assert res.compounds[comp].k1 == pytest.approx(k1, abs=0.02), comp


def test_calibration_recovers_driver_order():
    df, _, n_laps = _synthetic_race()
    res = fit_compounds(df, n_laps=n_laps, track_evolution=False)
    # BBB was planted 0.8s slower than AAA.
    assert res.driver_pace["BBB"] - res.driver_pace["AAA"] == pytest.approx(0.8, abs=0.1)


def test_kalman_recovers_known_deg():
    n_laps, a, b = 57, 90.0, 0.08
    fuel = FuelModel()
    rng = np.random.default_rng(3)
    kf = KalmanTyreModel(n_laps=n_laps, prior_pace=a, prior_deg=0.0)
    for age in range(0, 25):
        lap = 10 + age
        lt = a + b * age + fuel.penalty(lap, n_laps) + rng.normal(0, 0.08)
        kf.update(lap, age, lt)
    assert kf.belief.deg_rate == pytest.approx(b, abs=0.02)
    lo, hi = kf.belief.credible_deg()
    assert lo <= b <= hi  # truth inside the credible band


def test_kalman_flags_anomaly():
    n_laps = 57
    fuel = FuelModel()
    kf = KalmanTyreModel(n_laps=n_laps, prior_pace=90.0, prior_deg=0.05)
    flags = []
    for age in range(0, 20):
        lap = 10 + age
        lt = 90.0 + 0.05 * age + fuel.penalty(lap, n_laps)
        if age == 12:
            lt += 4.0  # a traffic/lock-up spike
        flags.append(kf.update(lap, age, lt).anomaly)
    assert flags[12], "the planted spike should be flagged"
    assert not any(flags[15:]), "filter should recover after the spike"
