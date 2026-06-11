"""Offline tests for the data-fusion upgrades: thermal degradation, pit-loss
variance, dirty-air pace loss, tyre-set allocation, and the calibrated priors."""

import numpy as np

from pitwall.data.priors import apply_to_model, constructor_id, reliability_for
from pitwall.live_optimizer import reoptimize
from pitwall.models import RaceModel
from pitwall.models.params import PitModel, TyreModel, WetModel
from pitwall.optimize.deterministic import (
    DEFAULT_ALLOCATION,
    allocation_ok,
    enumerate_candidates,
)
from pitwall.types import Compound
from pitwall.wet_strategy import optimal_wet_strategy


# --------------------------------------------------------------------------- #
# Thermal degradation                                                          #
# --------------------------------------------------------------------------- #
def test_thermal_slope_steepens_with_temperature():
    cool = TyreModel(temp_coeff=0.004, temp_ref=30.0, track_temp=30.0)
    hot = TyreModel(temp_coeff=0.004, temp_ref=30.0, track_temp=45.0)
    # At the reference temp the slope equals the intrinsic k1.
    assert cool.slope(Compound.MEDIUM) == cool.compounds[Compound.MEDIUM].k1
    # 15 degrees hotter -> slope up by temp_coeff * 15.
    assert hot.slope(Compound.MEDIUM) - cool.slope(Compound.MEDIUM) == 0.004 * 15
    # And a 20-lap-old tyre is slower in the heat.
    assert hot.penalty(Compound.MEDIUM, 20) > cool.penalty(Compound.MEDIUM, 20)


def test_thermal_inert_by_default():
    m = TyreModel()
    assert m.temp_coeff == 0.0
    assert m.slope(Compound.HARD) == m.compounds[Compound.HARD].k1


# --------------------------------------------------------------------------- #
# Pit-loss variance                                                            #
# --------------------------------------------------------------------------- #
def test_pit_sample_loss_zero_sd_is_deterministic():
    p = PitModel(pit_loss_s=24.0, pit_loss_sd=0.0)
    rng = np.random.default_rng(0)
    assert p.sample_loss("green", rng) == p.effective_loss("green") == 24.0


def test_pit_sample_loss_varies_and_floors():
    p = PitModel(pit_loss_s=24.0, pit_loss_sd=2.0)
    rng = np.random.default_rng(1)
    draws = [p.sample_loss("green", rng) for _ in range(2000)]
    assert np.std(draws) > 0.5                     # real variance shows up
    assert min(draws) >= 24.0 - 24.0 * 0.18 - 1e-9  # fast-stop floor respected
    assert abs(np.mean(draws) - 24.0) < 0.4        # unbiased around the mean


# --------------------------------------------------------------------------- #
# Tyre-set allocation                                                          #
# --------------------------------------------------------------------------- #
def test_allocation_ok():
    alloc = {Compound.SOFT: 2, Compound.MEDIUM: 2, Compound.HARD: 2}
    assert allocation_ok((Compound.SOFT, Compound.HARD), alloc)
    assert allocation_ok((Compound.SOFT, Compound.SOFT, Compound.HARD), alloc)
    # 3 softs when only 2 sets exist -> illegal.
    assert not allocation_ok((Compound.SOFT,) * 3 + (Compound.HARD,), alloc)
    assert allocation_ok((Compound.SOFT,) * 9, None)  # no constraint


def test_enumerate_candidates_respects_allocation():
    model = RaceModel.for_circuit("bahrain")
    tight = {Compound.SOFT: 1, Compound.MEDIUM: 1, Compound.HARD: 1}
    cands = enumerate_candidates(model, max_stops=3, allocation=tight)
    for c in cands:
        counts = {}
        for comp in c.strategy.compounds:
            counts[comp] = counts.get(comp, 0) + 1
        assert all(v <= 1 for v in counts.values())


def test_reoptimize_respects_allocation():
    model = RaceModel.for_circuit("bahrain")
    # Already used 2 hards + 1 soft; with only 2 hard sets, no plan may add a 3rd.
    alloc = {Compound.SOFT: 3, Compound.MEDIUM: 2, Compound.HARD: 2}
    counts = {Compound.HARD: 2, Compound.SOFT: 1}
    r = reoptimize(model, lap=30, compound=Compound.HARD, age=10,
                   belief_a=None, belief_b=None, regime="green",
                   used_compounds={Compound.HARD, Compound.SOFT},
                   allocation=alloc, set_counts=counts)
    # The recommended next compound (if any) must not be a 3rd hard.
    if r["first_comp"] is not None:
        assert r["first_comp"] != Compound.HARD.value or counts[Compound.HARD] < 2


# --------------------------------------------------------------------------- #
# Calibrated priors                                                            #
# --------------------------------------------------------------------------- #
def test_constructor_alias():
    assert constructor_id("Red Bull Racing") == "red_bull"
    assert constructor_id("Williams") == "williams"
    assert constructor_id(None) is None


def test_priors_applied_when_available():
    base = RaceModel.for_circuit("bahrain")
    cal = apply_to_model(base, circuit_id="bahrain", constructor="Williams")
    # Dirty-air pace loss is turned on for the high-fidelity path.
    assert cal.overtake.dirty_air_loss_s > 0
    # If the priors store is built, pit loss + reliability come from real data.
    rel = reliability_for("williams")
    if rel is not None:
        assert cal.safety_car.team_failure_prob == rel
        assert 0.0 < cal.safety_car.team_failure_prob < 0.5


def test_dirty_air_fitted_and_ordered():
    # If the priors store is built, the fitted dirty-air loss flows through and
    # respects the physical ordering: harder-to-pass tracks lose more pace.
    from pitwall.data.priors import circuit_prior
    monaco = circuit_prior("monaco").get("dirty_air_loss_s")
    monza = circuit_prior("monza").get("dirty_air_loss_s")
    if monaco is not None and monza is not None:
        assert monaco > monza            # Monaco wake >> Monza wake
        assert 0.0 < monza < monaco < 1.2
        m = apply_to_model(RaceModel.for_circuit("monaco"), circuit_id="monaco")
        assert m.overtake.dirty_air_loss_s == monaco


def test_default_allocation_is_sane():
    # The default never forbids a normal 2-stop (e.g. H-M-H or S-M-S).
    assert allocation_ok((Compound.HARD, Compound.MEDIUM, Compound.HARD), DEFAULT_ALLOCATION)
    assert allocation_ok((Compound.SOFT, Compound.MEDIUM, Compound.SOFT), DEFAULT_ALLOCATION)


# --------------------------------------------------------------------------- #
# Wet weather / crossover                                                      #
# --------------------------------------------------------------------------- #
def test_wetmodel_picks_right_tyre_by_wetness():
    w = WetModel()
    assert w.best_tyre(0.0) == Compound.MEDIUM          # dry -> slick
    assert w.best_tyre(0.5) == Compound.INTERMEDIATE    # damp -> inter
    assert w.best_tyre(0.95) == Compound.WET            # soaked -> full wet
    # Crossovers are ordered and inside (0, 1).
    x1 = w.crossover(Compound.MEDIUM, Compound.INTERMEDIATE)
    x2 = w.crossover(Compound.INTERMEDIATE, Compound.WET)
    assert 0.0 < x1 < x2 < 1.0


def test_optimal_wet_strategy_switches_on_drying_track():
    model = RaceModel.for_circuit("zandvoort", n_laps=20)
    # Dry start, heavy rain laps 8-14, drying back out.
    wetness = [0.0] * 21
    for L in range(8, 15):
        wetness[L] = 0.7
    plan = optimal_wet_strategy(model, wetness, start_tyre=Compound.MEDIUM)
    # It must go to a wet tyre during the downpour and come back to slicks.
    assert any(t != Compound.MEDIUM for t in plan.tyres[8:15])
    assert plan.tyres[20] == Compound.MEDIUM
    # At least one switch into and one back out of the wet phase.
    assert len(plan.switches) >= 2


def test_optimal_wet_strategy_stays_dry_when_dry():
    model = RaceModel.for_circuit("bahrain", n_laps=20)
    plan = optimal_wet_strategy(model, [0.0] * 21, start_tyre=Compound.MEDIUM)
    assert plan.switches == []
    assert all(t == Compound.MEDIUM for t in plan.tyres[1:])


def test_wet_lap_time_slower_than_dry():
    model = RaceModel.for_circuit("zandvoort")
    lt = model.lap_time
    dry = lt.green_lap(Compound.MEDIUM, 5, 10)
    wet_on_inters = lt.wet_lap(Compound.INTERMEDIATE, 5, 10, 0.5)
    wet_on_slicks = lt.wet_lap(Compound.MEDIUM, 5, 10, 0.5)
    assert wet_on_inters > dry                  # wet running is slower
    assert wet_on_slicks > wet_on_inters        # slicks in the wet are a disaster
