"""Race reconstruction / backtest.

The strongest validation a strategy simulator can offer: take a *real* race, feed
in each driver's **real grid slot, real strategy (compounds + stint lengths) and
calibrated pace**, simulate, and check the predicted finishing order matches what
actually happened. If the engine can re-run history, its forward predictions are
worth trusting.

Uses FastF1 (real stints + classified results). Reports Spearman rank correlation
vs the official result, mean absolute position error, and podium accuracy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..sim import CarEntry, RaceSimulator
from ..sim.monte_carlo import ScenarioSet
from ..types import Compound, Stint, Strategy
from .calibrate import calibrate_race, calibrate_races

_COMPOUND_MAP = {"SOFT": Compound.SOFT, "MEDIUM": Compound.MEDIUM, "HARD": Compound.HARD,
                 "INTERMEDIATE": Compound.INTERMEDIATE, "WET": Compound.WET}


@dataclass
class BacktestResult:
    circuit_id: str
    year: int
    n_drivers: int
    n_scenarios: int
    spearman: float            # rank correlation predicted vs actual
    mae_position: float        # mean |predicted - actual| over classified finishers
    podium_accuracy: float     # fraction of the real podium the sim puts on the podium
    predicted_order: list[str]
    actual_order: list[str]

    def summary(self) -> dict:
        return {"circuit": self.circuit_id, "year": self.year, "drivers": self.n_drivers,
                "spearman_rank_corr": round(self.spearman, 3),
                "mae_position": round(self.mae_position, 2),
                "podium_accuracy": round(self.podium_accuracy, 2)}


def _real_strategies(session) -> dict[str, Strategy]:
    """Reconstruct each driver's real strategy from their stint laps."""
    laps = session.laps
    out: dict[str, Strategy] = {}
    for drv, dl in laps.groupby("Driver"):
        stints = []
        for _, st in dl.groupby("Stint"):
            comp = _COMPOUND_MAP.get(str(st["Compound"].iloc[0]).upper())
            length = int(st["LapNumber"].max() - st["LapNumber"].min() + 1)
            if comp is not None and comp.is_slick and length > 0:
                stints.append(Stint(comp, length))
        if stints:
            out[drv] = Strategy(stints)
    return out


def reconstruct_race(
    year: int, country: str, circuit_id: str, *, n_scenarios: int = 300,
    pool_years: list[int] | None = None,
) -> BacktestResult | None:
    """Backtest the simulator against a real race result."""
    import fastf1

    from .fastf1_loader import _enable_cache
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, country, "R")
        ses.load(telemetry=False, weather=False, messages=False)

    results = ses.results
    if results is None or len(results) == 0:
        return None
    n_laps = int(ses.laps["LapNumber"].max())

    # Calibrated pace + degradation (pool extra years for stable deg if asked).
    if pool_years:
        cal = calibrate_races([(y, country) for y in pool_years], circuit_id)
        cal_single = calibrate_race(year, country, circuit_id)  # per-driver pace for THIS race
        driver_pace = cal_single.driver_pace if cal_single else {}
        ref = cal_single.reference_pace if cal_single else 0.0
        compounds = cal.compounds if cal else {}
    else:
        cal = calibrate_race(year, country, circuit_id)
        if cal is None:
            return None
        driver_pace, ref, compounds = cal.driver_pace, cal.reference_pace, cal.compounds

    model = (cal if not pool_years else cal).to_race_model(circuit_id) if cal else None
    if model is None:
        return None
    # Inject this race's calibrated compounds.
    from dataclasses import replace
    if compounds:
        merged = dict(model.tyres.compounds)
        merged.update(compounds)
        model.tyres = replace(model.tyres, compounds=merged)

    strategies = _real_strategies(ses)

    entries: list[CarEntry] = []
    actual: list[tuple[int, str]] = []
    for _, row in results.iterrows():
        abbr = row["Abbreviation"]
        if abbr not in strategies:
            continue
        delta = driver_pace.get(abbr, ref + 1.0) - ref
        grid = int(row["GridPosition"]) if row["GridPosition"] and row["GridPosition"] > 0 else 20
        cid = len(entries)
        entries.append(CarEntry(car_id=cid, model=model.with_driver(delta),
                                strategy=strategies[abbr], grid=grid, name=abbr))
        pos = int(row["Position"]) if row["Position"] and not np.isnan(row["Position"]) else 99
        actual.append((pos, abbr))

    if len(entries) < 6:
        return None

    sim = RaceSimulator(entries, n_laps)
    scen = ScenarioSet.sample(model.safety_car, n_laps, n_scenarios, seed=7)
    mean_pos = {e.car_id: [] for e in entries}
    for s in scen.scenarios:
        res = sim.run_once(np.random.default_rng(s.seed), race_control=s.race_control)
        for cid, p in res.position.items():
            mean_pos[cid].append(p)
    pred = {e.name: float(np.mean(mean_pos[e.car_id])) for e in entries}

    # Compare predicted mean position vs actual classified order.
    actual_sorted = [a for _, a in sorted(actual)]
    pred_sorted = [n for n, _ in sorted(pred.items(), key=lambda kv: kv[1])]
    common = [a for a in actual_sorted if a in pred]
    actual_rank = {a: i for i, a in enumerate(common)}
    pred_rank = {n: i for i, n in enumerate([n for n in pred_sorted if n in actual_rank])}

    from scipy.stats import spearmanr
    xs = [actual_rank[a] for a in common]
    ys = [pred_rank[a] for a in common]
    rho = float(spearmanr(xs, ys).correlation)
    actual_finish_pos = {a: i + 1 for i, a in enumerate(common)}
    pred_finish_pos = {n: i + 1 for i, n in enumerate([n for n in pred_sorted if n in actual_rank])}
    mae = float(np.mean([abs(actual_finish_pos[a] - pred_finish_pos[a]) for a in common]))
    real_podium = set(common[:3])
    pred_podium = set(pred_sorted[:3])
    podium_acc = len(real_podium & pred_podium) / 3.0

    return BacktestResult(
        circuit_id=circuit_id, year=year, n_drivers=len(entries), n_scenarios=n_scenarios,
        spearman=rho, mae_position=mae, podium_accuracy=podium_acc,
        predicted_order=pred_sorted, actual_order=actual_sorted,
    )
