"""Robust strategy selection under uncertainty.

Stage 2 of the optimiser. Each deterministic candidate is re-scored across the
shared :class:`ScenarioSet` (common random numbers), then selected on an explicit
objective. Crucially we do **not** just pick the deterministic time-optimum: we
optimise the *outcome distribution* and let the caller trade expected position
against tail risk (CVaR). The Pareto frontier of (expected, tail) candidates is
exposed for the dashboard's risk-appetite slider.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..sim import CarEntry, EnsembleResult, ScenarioSet, evaluate, with_focal
from ..types import Strategy
from .deterministic import Candidate, enumerate_candidates

OBJECTIVES = ("expected", "podium", "win", "points", "robust")


@dataclass
class ScoredCandidate:
    strategy: Strategy
    det_time: float
    ensemble: EnsembleResult

    def loss(self, objective: str = "podium", risk_aversion: float = 0.5) -> float:
        """Lower is better. Position objectives minimise; probability objectives
        are negated; 'robust' penalises the tail above the mean."""
        e = self.ensemble
        if objective == "expected":
            return e.mean_position
        if objective == "podium":
            return -e.p_podium
        if objective == "win":
            return -e.p_win
        if objective == "points":
            return -e.p_points
        if objective == "robust":
            # Mean-variance: penalise outcome spread (SC/traffic exposure).
            return e.mean_position + risk_aversion * e.std_position
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")


@dataclass
class OptimizeResult:
    objective: str
    ranked: list[ScoredCandidate]   # best-first under the objective
    scenarios: int

    @property
    def best(self) -> ScoredCandidate:
        return self.ranked[0]

    def frontier(self) -> list[ScoredCandidate]:
        """Pareto-optimal candidates in (expected position, tail CVaR) — the
        risk/reward efficient frontier. Both minimised."""
        pts = sorted(self.ranked, key=lambda s: s.ensemble.mean_position)
        out, best_tail = [], float("inf")
        for s in pts:
            tail = s.ensemble.cvar_position(0.10)
            if tail < best_tail:
                out.append(s)
                best_tail = tail
        return out

    def table(self, n: int = 8) -> list[dict]:
        rows = []
        for s in self.ranked[:n]:
            rows.append({"strategy": s.strategy.label(), "det_time": round(s.det_time, 1),
                         **s.ensemble.summary()})
        return rows


def score_strategies(
    rivals: list[CarEntry],
    strategies: list[Strategy],
    scenarios: ScenarioSet,
    *,
    circuit_id: str,
    focal_grid: int = 1,
    focal_delta: float | None = None,
    focal_top_speed: float = 0.0,
    focal_id: int = 99,
    focal_model=None,
    det_times: list[float] | None = None,
    use_native: bool = False,
) -> list[ScoredCandidate]:
    scored = []
    for i, strat in enumerate(strategies):
        field = with_focal(
            rivals, strat, circuit_id=circuit_id, focal_grid=focal_grid,
            focal_delta=focal_delta, n_laps=scenarios.n_laps, focal_id=focal_id,
            focal_top_speed=focal_top_speed, focal_model=focal_model,
        )
        ens = evaluate(field, scenarios, focal_id=focal_id, use_native=use_native)
        scored.append(ScoredCandidate(strat, det_times[i] if det_times else 0.0, ens))
    return scored


def optimize(
    model,
    rivals: list[CarEntry],
    scenarios: ScenarioSet,
    *,
    circuit_id: str,
    focal_grid: int = 1,
    focal_delta: float | None = None,
    focal_top_speed: float = 0.0,
    objective: str = "podium",
    risk_aversion: float = 0.5,
    max_stops: int = 3,
    mandatory_start=None,
    shortlist: int = 12,
    focal_id: int = 99,
    use_native: bool = False,
) -> OptimizeResult:
    """End-to-end: enumerate deterministic candidates, shortlist the fastest,
    re-score robustly, and rank under ``objective``.

    The shortlist is **diversified by stop count**: the best free-track candidate
    of *each* stop count (1/2/3-stop) is always included before filling the rest
    by time. Otherwise a long circuit's shortlist can be all 2-stops and the
    robust pass never even evaluates a 1-stop (which on a track-position circuit
    may be the right call)."""
    if focal_delta is not None:
        model = model.with_driver(focal_delta)
    all_cands: list[Candidate] = enumerate_candidates(
        model, max_stops=max_stops, mandatory_start=mandatory_start, top_k=None
    )
    by_stops: dict[int, Candidate] = {}
    for c in all_cands:  # all_cands is sorted by free-track time, so first = best
        by_stops.setdefault(c.strategy.n_stops, c)
    cands = list(by_stops.values())  # best of each stop count, guaranteed
    for c in all_cands:              # fill remaining slots by time
        if len(cands) >= shortlist:
            break
        if c not in cands:
            cands.append(c)
    cands.sort(key=lambda c: c.det_time)
    scored = score_strategies(
        rivals, [c.strategy for c in cands], scenarios,
        circuit_id=circuit_id, focal_grid=focal_grid, focal_delta=focal_delta,
        focal_top_speed=focal_top_speed, focal_id=focal_id,
        det_times=[c.det_time for c in cands], use_native=use_native, focal_model=model,
    )
    scored.sort(key=lambda s: s.loss(objective, risk_aversion))
    return OptimizeResult(objective=objective, ranked=scored, scenarios=len(scenarios))
