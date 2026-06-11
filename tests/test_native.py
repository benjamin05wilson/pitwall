"""Rust accelerator parity test (skipped if the extension isn't built)."""

import numpy as np
import pytest

from pitwall.models import RaceModel
from pitwall.sim import ScenarioSet, build_field, evaluate, nominal_strategy, with_focal
from pitwall.sim.native import HAS_NATIVE

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="native extension not built")


def _field(circuit="bahrain", grid=3, delta=0.3):
    model = RaceModel.for_circuit(circuit)
    rivals = build_field(circuit, n_cars=20, seed=1)
    strat = nominal_strategy(model.config.n_laps)
    field = with_focal(rivals, strat, circuit_id=circuit, focal_grid=grid, focal_delta=delta)
    return model, field


def test_native_agrees_with_python_within_mc_error():
    model, field = _field()
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, 1500, seed=7)
    py = evaluate(field, scen)
    rs = evaluate(field, scen, use_native=True)
    assert abs(py.mean_position - rs.mean_position) < 0.25
    assert abs(py.p_podium - rs.p_podium) < 0.05
    assert rs.positions.min() >= 1 and rs.positions.max() <= len(field)
