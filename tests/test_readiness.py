"""Offline integration regressions for the public demonstration contract."""
from dataclasses import replace
from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitwall.models import RaceControl, RaceModel, RED
from pitwall.models.params import CompoundParams
from pitwall.optimize import optimize
from pitwall.optimize.deterministic import enumerate_candidates
from pitwall.optimize.robust import score_strategies
from pitwall.sim import ScenarioSet, build_field, with_focal
from pitwall.sim.monte_carlo import Scenario
from pitwall.sim.race import CarEntry, RaceSimulator
from pitwall.types import Compound, Stint, Strategy


def distinctive_model(laps=8):
    m = RaceModel.for_circuit('bahrain', n_laps=laps)
    return replace(m, pace=replace(m.pace, t_quali_ref=81, race_pace_gap=7, driver_delta=.7, lap_noise_sigma=0),
                   fuel=replace(m.fuel, k_fuel=.041, m_start=71, burn_per_lap=2.3),
                   tyres=replace(m.tyres, compounds={**m.tyres.compounds, Compound.SOFT: CompoundParams(-.8, .23, .004)}, temp_coeff=.003, track_temp=41),
                   pit=replace(m.pit, pit_loss_s=17, pit_loss_sd=.8, sc_saving_frac=.61),
                   safety_car=replace(m.safety_car, team_failure_prob=0, lap_mult_sc=1.9),
                   overtake=replace(m.overtake, dirty_air_loss_s=.2))


def test_optimizer_preserves_model_at_actual_python_simulator(monkeypatch):
    m = distinctive_model()
    bank = ScenarioSet([Scenario(RaceControl(8), 17)], 8)
    captured = []
    original = RaceSimulator.run_once
    def capture(self, *args, **kwargs):
        captured.append(self.entries[0].model)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(RaceSimulator, 'run_once', capture)
    optimize(m, build_field('bahrain', n_laps=8, n_cars=3), bank, circuit_id='bahrain',
             focal_delta=.42, max_stops=1, shortlist=2)
    assert len(captured) == 2
    for actual in captured:
        assert actual == m.with_driver(.42)
    assert m.pace.driver_delta == .7


def test_fixed_strategy_scoring_uses_supplied_model():
    m = distinctive_model()
    bank = ScenarioSet([Scenario(RaceControl(8), 17)], 8)
    strategy = Strategy([Stint(Compound.SOFT, 4), Stint(Compound.HARD, 4)])
    def time(model):
        return score_strategies([], [strategy], bank, circuit_id='bahrain', focal_model=model)[0].ensemble.total_times[0]
    assert time(replace(m, pace=replace(m.pace, t_quali_ref=91))) - time(m) == pytest.approx(80)
    assert with_focal([], strategy, circuit_id='bahrain', focal_model=m)[0].model == m
    with pytest.raises(ValueError, match='lap count'):
        with_focal([], strategy, circuit_id='bahrain', focal_model=m, n_laps=9)


def test_native_gate_preserves_unsupported_settings(monkeypatch):
    from pitwall.sim import native, evaluate
    m = distinctive_model()
    bank = ScenarioSet([Scenario(RaceControl(8), 17)], 8)
    strategy = Strategy([Stint(Compound.SOFT, 4), Stint(Compound.HARD, 4)])
    field = with_focal([], strategy, circuit_id='bahrain', focal_model=m)
    monkeypatch.setattr(native, 'HAS_NATIVE', True)
    monkeypatch.setattr(native, '_native', SimpleNamespace(simulate_batch=lambda *a: pytest.fail('unsupported native call')))
    assert native.backend_for(field, bank) == 'Python'
    a, b = evaluate(field, bank), evaluate(field, bank, use_native=True)
    np.testing.assert_array_equal(a.total_times, b.total_times)
    with pytest.raises(ValueError, match='pit-loss variance'):
        native.simulate_batch_native_full(field, bank)


@pytest.mark.parametrize('stops', [1, 2])
def test_tiny_dp_matches_exhaustive_lap_and_compound_oracle(stops):
    m = distinctive_model(6)
    slicks = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
    brute = []
    for seq in product(slicks, repeat=stops + 1):
        if len(set(seq)) < 2:
            continue
        for cuts in combinations(range(1, 6), stops):
            bounds = (0, *cuts, 6)
            total = stops * m.pit.pit_loss_s
            for c, start, end in zip(seq, bounds, bounds[1:]):
                total += sum(m.lap_time.green_lap(c, lap-start-1, lap) for lap in range(start+1, end+1))
            brute.append(total)
    best = enumerate_candidates(m, min_stops=stops, max_stops=stops)[0]
    assert best.det_time == pytest.approx(min(brute))


@pytest.mark.parametrize('scheduled_stop', [False, True])
def test_red_flag_resets_age_and_charges_no_pit_loss(scheduled_stop):
    m = distinctive_model(5)
    m = replace(m, pit=replace(m.pit, pit_loss_sd=0))
    rc = RaceControl(5)
    rc.regime[3] = RED
    strat = Strategy([Stint(Compound.SOFT, 3), Stint(Compound.HARD, 2)]) if scheduled_stop else Strategy([Stint(Compound.SOFT, 5)])
    result = RaceSimulator([CarEntry(99, m, strat)], 5).run_once(np.random.default_rng(1), rc, focal_id=99)
    assert result.focal_lap_time[3] == pytest.approx(m.lap_time.green_lap(Compound.SOFT, 2, 3) * m.safety_car.lap_mult_sc)
    comp = Compound.HARD if scheduled_stop else Compound.SOFT
    assert result.focal_lap_time[4] == pytest.approx(m.lap_time.green_lap(comp, 0, 4))
    assert result.focal_lap_time[5] == pytest.approx(m.lap_time.green_lap(comp, 1, 5))


def test_current_session_cannot_relabel_history(monkeypatch):
    import pitwall.replay as replay
    monkeypatch.setattr(replay, 'openf1_live_session', lambda: {'session_key': 'current'})
    meta = dict(year=2023, gp='Bahrain', circuit_id='bahrain', driver='ALB', team='Williams', n_laps=8, pit_loss=22, prior_pace=95)
    monkeypatch.setattr(replay, '_historical', lambda *args: SimpleNamespace(meta=meta, actual_pit_laps=[], events=[]))
    assert next(replay.stream_events(2023, 'Bahrain', 'ALB', interval=0))['source'] == 'historical-replay'


def test_no_weights_api_returns_core_recommendation(monkeypatch):
    from pitwall import server
    monkeypatch.setattr(server, '_surrogate', lambda: None)
    with TestClient(server.app) as client:
        assert client.post('/api/heatmap', json={}).status_code == 503
        response = client.post('/api/optimize', json={'scenarios': 2})
        assert response.status_code == 200
        result = response.json()
        assert result['best']['label']
        assert len(result['ranked']) == 3
        assert result['frontier_optimal']
        assert result['backend'] in ('Python', 'Rust')
        assert 'defaults' in result['focal_model']
        assert client.post('/api/optimize', json={'scenarios': 0}).status_code == 422
        assert client.post('/api/optimize', json={'objective': 'bogus'}).status_code == 422

@pytest.mark.parametrize('setting', ['wet', 'compound', 'retirement', 'dirty_air', 'pit_variance'])
def test_each_unsupported_native_setting_selects_python(monkeypatch, setting):
    from pitwall.sim import native
    m = RaceModel.for_circuit('bahrain', n_laps=8)
    strategy = Strategy([Stint(Compound.SOFT, 4), Stint(Compound.HARD, 4)])
    bank = ScenarioSet([Scenario(RaceControl(8), 17)], 8)
    if setting == 'wet':
        bank.scenarios[0].race_control.wetness[3] = .4
    if setting == 'compound':
        strategy = Strategy([Stint(Compound.INTERMEDIATE, 8)])
    if setting == 'dirty_air':
        m = replace(m, overtake=replace(m.overtake, dirty_air_loss_s=.2))
    if setting == 'pit_variance':
        m = replace(m, pit=replace(m.pit, pit_loss_sd=.4))
    field = [CarEntry(99, m, strategy, retire_lap=3 if setting == 'retirement' else None)]
    monkeypatch.setattr(native, 'HAS_NATIVE', True)
    assert native.unsupported_reason(field, bank)
    assert native.backend_for(field, bank) == 'Python'


def test_backend_label_respects_availability_and_request(monkeypatch):
    from pitwall.sim import native
    field = build_field('bahrain', n_laps=8, n_cars=3)
    bank = ScenarioSet([Scenario(RaceControl(8), 17)], 8)
    monkeypatch.setattr(native, 'HAS_NATIVE', True)
    assert native.backend_for(field, bank) == 'Rust'
    assert native.backend_for(field, bank, use_native=False) == 'Python'
    monkeypatch.setattr(native, 'HAS_NATIVE', False)
    assert native.backend_for(field, bank) == 'Python'
