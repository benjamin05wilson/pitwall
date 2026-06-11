"""Thin Python adapter over the Rust ``_native`` accelerator.

Flattens a fixed field (``list[CarEntry]``) plus a ``ScenarioSet`` into the flat
arrays the Rust ``simulate_batch`` expects and returns the focal car's finishing
positions. The Rust loop reproduces ``RaceSimulator.run_once`` (the pure-Python
source of truth) within Monte-Carlo error — it is a drop-in accelerator, not a
bit-for-bit replica (the RNG streams differ across languages).

If the compiled extension is absent, ``HAS_NATIVE`` is ``False`` and callers
should fall back to ``pitwall.sim.evaluate``.
"""

from __future__ import annotations

import numpy as np

from ..models import GREEN, RED, SC, VSC
from ..types import Compound
from .race import RaceSimulator

try:  # pragma: no cover - import guard
    import _native  # top-level module produced by the Rust crate

    HAS_NATIVE = bool(hasattr(_native, "simulate_batch"))
except Exception:  # noqa: BLE001 - any import failure -> graceful fallback
    _native = None
    HAS_NATIVE = False


# Compound -> index, in the [SOFT, MEDIUM, HARD] order the Rust side assumes.
_COMPOUND_ORDER = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
_COMPOUND_IDX = {c: i for i, c in enumerate(_COMPOUND_ORDER)}

# Regime string -> u8 code matching the Rust loop (0=green, 1=SC, 2=VSC, 3=RED).
_REGIME_CODE = {GREEN: 0, SC: 1, VSC: 2, RED: 3}


def _compound_index(comp: Compound) -> int:
    """Map a Compound to its slick index, clamping non-slicks to HARD so the
    accelerator never indexes out of range (the dry-strategy engine only ever
    plans S/M/H, matching the Python sim's effective behaviour)."""
    return _COMPOUND_IDX.get(comp, 2)


def simulate_batch_native(field, scenarios, focal_id: int = 99):
    """Run every scenario in ``scenarios`` for ``field`` via the Rust loop.

    Mirrors how ``RaceSimulator`` builds per-car pit plans and pulls model
    parameters. Returns the focal car's finishing position per scenario as an
    int numpy array (same shape/semantics as ``EnsembleResult.positions``).

    Raises ``RuntimeError`` if the native extension is unavailable.
    """
    if not HAS_NATIVE:
        raise RuntimeError("native extension (_native) not available")

    positions, _times = _simulate(field, scenarios, focal_id)
    return positions


def simulate_batch_native_full(field, scenarios, focal_id: int = 99):
    """As :func:`simulate_batch_native` but also returns focal total times.

    Returns ``(positions, total_times)`` as numpy arrays."""
    if not HAS_NATIVE:
        raise RuntimeError("native extension (_native) not available")
    return _simulate(field, scenarios, focal_id)


def _simulate(field, scenarios, focal_id):
    n = len(field)
    n_laps = scenarios.n_laps

    # Locate the focal car's row in the field (its index, not its car_id).
    focal_idx = next(i for i, e in enumerate(field) if e.car_id == focal_id)

    # ---- per-car scalar parameters ---------------------------------------- #
    base_pace = np.empty(n, dtype=np.float64)
    k_fuel = np.empty(n, dtype=np.float64)
    m_start = np.empty(n, dtype=np.float64)
    burn = np.empty(n, dtype=np.float64)
    sigma = np.empty(n, dtype=np.float64)
    top_speed_delta = np.empty(n, dtype=np.float64)
    cold_penalty = np.empty(n, dtype=np.float64)
    deg_mult_sc = np.empty(n, dtype=np.float64)
    deg_mult_vsc = np.empty(n, dtype=np.float64)
    lap_mult_sc = np.empty(n, dtype=np.float64)
    lap_mult_vsc = np.empty(n, dtype=np.float64)
    pit_loss = np.empty(n, dtype=np.float64)
    sc_saving_frac = np.empty(n, dtype=np.float64)
    vsc_saving_frac = np.empty(n, dtype=np.float64)
    team_failure_prob = np.empty(n, dtype=np.float64)

    # per-car per-compound tyre coefficients, row-major car-major (len n*3)
    k0 = np.empty(n * 3, dtype=np.float64)
    k1 = np.empty(n * 3, dtype=np.float64)
    k2 = np.empty(n * 3, dtype=np.float64)

    start_compound = np.empty(n, dtype=np.int64)
    grid = np.empty(n, dtype=np.int64)  # starting grid slot per car (for stagger)

    # ragged pit plan, mirroring RaceSimulator._pit_plan construction
    pit_offsets = np.empty(n + 1, dtype=np.uint64)
    pit_laps_list: list[int] = []
    pit_comps_list: list[int] = []

    for i, e in enumerate(field):
        m = e.model
        base_pace[i] = m.pace.base
        k_fuel[i] = m.fuel.k_fuel
        m_start[i] = m.fuel.m_start
        burn[i] = m.fuel.burn(n_laps)
        sigma[i] = m.pace.lap_noise_sigma
        top_speed_delta[i] = e.top_speed_delta
        cold_penalty[i] = m.tyres.cold_lap_penalty
        deg_mult_sc[i] = m.tyres.deg_mult_sc
        deg_mult_vsc[i] = m.tyres.deg_mult_vsc
        lap_mult_sc[i] = m.safety_car.lap_mult_sc
        lap_mult_vsc[i] = m.safety_car.lap_mult_vsc
        pit_loss[i] = m.pit.pit_loss_s
        sc_saving_frac[i] = m.pit.sc_saving_frac
        vsc_saving_frac[i] = m.pit.vsc_saving_frac
        team_failure_prob[i] = m.safety_car.team_failure_prob

        for c_idx, comp in enumerate(_COMPOUND_ORDER):
            cp = m.tyres.compounds[comp]
            k0[i * 3 + c_idx] = cp.k0
            # Effective (thermal-aware) slope so the Rust loop matches the Python
            # reference when a model carries a fitted temperature coupling.
            k1[i * 3 + c_idx] = m.tyres.slope(comp)
            k2[i * 3 + c_idx] = cp.k2

        # Strategy: first compound is the start; subsequent compounds are the
        # new compound at each pit lap (exactly RaceSimulator's pairing of
        # strategy.pit_laps with strategy.compounds[1:]).
        start_compound[i] = _compound_index(e.strategy.compounds[0])
        grid[i] = int(e.grid)

        pit_offsets[i] = len(pit_laps_list)
        for pit_lap, comp in zip(e.strategy.pit_laps, e.strategy.compounds[1:]):
            pit_laps_list.append(int(pit_lap))
            pit_comps_list.append(_compound_index(comp))
    pit_offsets[n] = len(pit_laps_list)

    pit_laps = np.asarray(pit_laps_list, dtype=np.int64)
    pit_comps = np.asarray(pit_comps_list, dtype=np.int64)

    # ---- circuit overtaking params (race-level: first car's model) -------- #
    ov = field[0].model.overtake

    # Grid-stagger gap: same default the pure-Python RaceSimulator uses, so the
    # initial cumulative-time offset (grid-1)*grid_gap matches exactly. It is a
    # keyword-only arg, so its default lives in __kwdefaults__.
    grid_gap = float(RaceSimulator.__init__.__kwdefaults__["grid_gap"])

    # ---- per-scenario regimes + seeds ------------------------------------- #
    k = len(scenarios.scenarios)
    stride = n_laps + 1
    regimes = np.zeros(k * stride, dtype=np.uint8)
    seeds = np.empty(k, dtype=np.uint64)
    for s, scen in enumerate(scenarios.scenarios):
        regime = scen.race_control.regime
        base = s * stride
        # regime is a list[str] of length n_laps+1 (index 0 unused).
        for lap in range(min(len(regime), stride)):
            regimes[base + lap] = _REGIME_CODE.get(regime[lap], 0)
        seeds[s] = np.uint64(scen.seed)

    positions, times = _native.simulate_batch(
        int(n),
        int(n_laps),
        int(focal_idx),
        base_pace.tolist(),
        k_fuel.tolist(),
        m_start.tolist(),
        burn.tolist(),
        sigma.tolist(),
        top_speed_delta.tolist(),
        cold_penalty.tolist(),
        deg_mult_sc.tolist(),
        deg_mult_vsc.tolist(),
        lap_mult_sc.tolist(),
        lap_mult_vsc.tolist(),
        pit_loss.tolist(),
        sc_saving_frac.tolist(),
        vsc_saving_frac.tolist(),
        team_failure_prob.tolist(),
        k0.tolist(),
        k1.tolist(),
        k2.tolist(),
        start_compound.tolist(),
        grid.tolist(),
        float(grid_gap),
        pit_offsets.tolist(),
        pit_laps.tolist(),
        pit_comps.tolist(),
        float(ov.threshold_s),
        float(ov.velocity_mod),
        float(ov.p_overtake),
        float(ov.drs_bonus),
        float(ov.drs_window),
        int(ov.drs_enable_lap),
        int(ov.drs_sc_delay),
        float(ov.min_follow_gap),
        float(ov.duel_penalty),
        float(ov.loser_penalty),
        regimes.tolist(),
        seeds.tolist(),
    )

    return np.asarray(positions, dtype=np.int64), np.asarray(times, dtype=np.float64)
