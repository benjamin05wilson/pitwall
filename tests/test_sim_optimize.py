"""Simulator + optimiser behavioural tests."""

import numpy as np
import pytest

from pitwall.models import RaceControl, RaceModel
from pitwall.optimize import enumerate_candidates, optimize
from pitwall.sim import (
    ScenarioSet,
    build_field,
    evaluate,
    nominal_strategy,
    with_focal,
)
from pitwall.types import Compound, Stint, Strategy


def _field(circuit="bahrain", grid=5, delta=0.3):
    model = RaceModel.for_circuit(circuit)
    rivals = build_field(circuit, n_cars=20, seed=1)
    strat = nominal_strategy(model.config.n_laps)
    field = with_focal(rivals, strat, circuit_id=circuit, focal_grid=grid, focal_delta=delta)
    return model, field


def test_focal_position_is_in_field_bounds():
    model, field = _field()
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, 50, seed=3)
    ens = evaluate(field, scen)
    assert ens.positions.min() >= 1
    assert ens.positions.max() <= len(field)


def test_faster_car_beats_slower_on_clear_track():
    """With no SC and a big pace edge from pole, the focal car should usually win."""
    circuit = "bahrain"
    model = RaceModel.for_circuit(circuit)
    rivals = build_field(circuit, n_cars=20, seed=2)
    strat = nominal_strategy(model.config.n_laps)
    field = with_focal(rivals, strat, circuit_id=circuit, focal_grid=1, focal_delta=-0.8)
    # Deterministic green race (no safety cars) repeated; reliability still bites,
    # so require a strong win-rate rather than certainty.
    rc = RaceControl(model.config.n_laps)
    scen = ScenarioSet([__import__("pitwall.sim.monte_carlo", fromlist=["Scenario"]).Scenario(rc, s)
                        for s in range(60)], model.config.n_laps)
    ens = evaluate(field, scen)
    assert ens.p_win > 0.6


def test_enumerate_respects_fia_rule():
    model = RaceModel.for_circuit("bahrain")
    cands = enumerate_candidates(model, max_stops=2, top_k=50)
    assert cands, "should produce candidates"
    for c in cands:
        assert c.strategy.uses_two_compounds()


def test_deterministic_dp_optimal_pitlap():
    """The DP-placed pit lap should beat hand-picked neighbours on free-track time."""
    from pitwall.optimize.deterministic import StintCostTable, _optimal_boundaries
    model = RaceModel.for_circuit("bahrain")
    table = StintCostTable(model)
    comps = (Compound.MEDIUM, Compound.HARD)
    bounds, best = _optimal_boundaries(table, comps, model.pit.pit_loss_s)
    p = bounds[0]
    # Perturbing the pit lap by +/-3 should not be faster.
    for dp in (-3, 3):
        q = p + dp
        if 1 <= q < model.config.n_laps:
            t = (table.cost[comps[0]][1, q] + model.pit.pit_loss_s
                 + table.cost[comps[1]][q + 1, model.config.n_laps - q])
            assert t >= best - 1e-6


def test_same_scenario_bank_is_repeatable():
    """Re-evaluating the same strategy on the same ScenarioSet is deterministic."""
    model, field = _field()
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, 80, seed=9)
    a = evaluate(field, scen).mean_position
    b = evaluate(field, scen).mean_position
    assert a == b


def test_grid_position_matters():
    """Starting on pole should finish meaningfully better than starting last,
    for an identical car — track position must be real."""
    circuit = "bahrain"
    model = RaceModel.for_circuit(circuit)
    rivals = build_field(circuit, n_cars=20, seed=1)
    strat = nominal_strategy(model.config.n_laps)
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, 200, seed=4)
    front = evaluate(with_focal(rivals, strat, circuit_id=circuit, focal_grid=1, focal_delta=0.3), scen)
    back = evaluate(with_focal(rivals, strat, circuit_id=circuit, focal_grid=20, focal_delta=0.3), scen)
    assert back.mean_position - front.mean_position > 2.0


def test_red_flag_race_classifies_full_field():
    """A red-flag race still produces a complete classification."""
    from pitwall.models import RED, RaceControl
    model = RaceModel.for_circuit("bahrain")
    rivals = build_field("bahrain", n_cars=20, seed=1)
    strat = nominal_strategy(model.config.n_laps)
    field = with_focal(rivals, strat, circuit_id="bahrain", focal_grid=5, focal_delta=0.3)
    from pitwall.sim import RaceSimulator
    sim = RaceSimulator(field, model.config.n_laps)
    rc = RaceControl(model.config.n_laps)
    rc.regime[20] = RED  # plant a red flag on lap 20
    res = sim.run_once(np.random.default_rng(0), race_control=rc, focal_id=99)
    # The race still classifies a full field with valid positions.
    assert set(res.position.values()) == set(range(1, len(field) + 1))


def test_optimizer_returns_ranked_best():
    model = RaceModel.for_circuit("bahrain")
    rivals = build_field("bahrain", seed=1)
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, 120, seed=7)
    res = optimize(model, rivals, scen, circuit_id="bahrain", focal_grid=3,
                   focal_delta=0.2, objective="podium", shortlist=8)
    assert res.best is res.ranked[0]
    # Best podium strategy has the highest podium prob in the set.
    assert res.best.ensemble.p_podium == max(s.ensemble.p_podium for s in res.ranked)
