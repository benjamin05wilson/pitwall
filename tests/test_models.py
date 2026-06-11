"""Model-core tests: domain types, lap-time equation, safety-car sampler."""

import numpy as np
import pytest

from pitwall.models import RaceModel, SafetyCarSampler
from pitwall.types import Compound, Stint, Strategy


# -- domain types ----------------------------------------------------------
def test_strategy_geometry():
    s = Strategy([Stint(Compound.SOFT, 20), Stint(Compound.MEDIUM, 18), Stint(Compound.HARD, 19)])
    assert s.n_stops == 2
    assert s.total_laps == 57
    assert s.pit_laps == (20, 38)
    assert s.compounds == (Compound.SOFT, Compound.MEDIUM, Compound.HARD)


def test_fia_two_compound_rule():
    assert Strategy([Stint(Compound.SOFT, 30), Stint(Compound.HARD, 27)]).uses_two_compounds()
    assert not Strategy([Stint(Compound.MEDIUM, 30), Stint(Compound.MEDIUM, 27)]).uses_two_compounds()


def test_strategy_is_hashable():
    s = Strategy([Stint(Compound.SOFT, 30), Stint(Compound.HARD, 27)])
    assert len({s, s}) == 1


def test_stint_rejects_nonpositive():
    with pytest.raises(ValueError):
        Stint(Compound.SOFT, 0)


# -- lap-time model --------------------------------------------------------
def test_degradation_is_monotonic():
    lt = RaceModel.for_circuit("bahrain").lap_time
    # At fixed lap (fixed fuel), worn tyres are slower than newer ones.
    times = [lt.components(Compound.MEDIUM, age, lap=30).tyre for age in range(1, 25)]
    assert all(b >= a for a, b in zip(times, times[1:]))


def test_fuel_effect_decreases_through_race():
    lt = RaceModel.for_circuit("bahrain").lap_time
    early = lt.components(Compound.MEDIUM, 5, lap=1).fuel
    late = lt.components(Compound.MEDIUM, 5, lap=57).fuel
    assert early > late and early - late == pytest.approx(3.2, abs=0.5)


def test_cold_outlap_penalty():
    lt = RaceModel.for_circuit("bahrain").lap_time
    # Age-0 (out-lap) carries the +1.0s cold penalty vs the underlying curve.
    out = lt.components(Compound.SOFT, 0, lap=10).tyre
    one = lt.components(Compound.SOFT, 1, lap=10).tyre
    assert out > one


# -- safety-car sampler ----------------------------------------------------
def test_sc_probability_matches_override():
    m = RaceModel.for_circuit("bahrain")  # P(SC) override 0.45
    samp = SafetyCarSampler(m.safety_car)
    rng = np.random.default_rng(0)
    races = [samp.sample(57, rng) for _ in range(4000)]
    p1 = np.mean([1.0 if r.sc_laps else 0.0 for r in races])
    assert p1 == pytest.approx(0.45, abs=0.04)


def test_singapore_sc_near_certain():
    m = RaceModel.for_circuit("marina_bay")  # override 1.0
    samp = SafetyCarSampler(m.safety_car)
    rng = np.random.default_rng(1)
    races = [samp.sample(62, rng) for _ in range(2000)]
    assert np.mean([1.0 if r.sc_laps else 0.0 for r in races]) > 0.95
