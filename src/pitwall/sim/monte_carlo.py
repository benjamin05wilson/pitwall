"""Monte-Carlo ensemble over stochastic race scenarios.

Two ideas make this useful for *decision-making* rather than just simulation:

* **Outcome distributions, not point estimates.** A strategy is scored by its
  finishing-position distribution — E[pos], P(win/podium/points), and a tail
  (CVaR of the worst outcomes) — so the optimiser can trade expected value
  against risk instead of chasing a fragile deterministic optimum.

* **Common random numbers (CRN).** The SC/VSC timelines, lap noise and
  retirements are sampled *once* into a :class:`ScenarioSet`. Every candidate
  strategy is then replayed against those *same* scenarios, so differences in
  outcome are attributable to the strategy, not to luck. This slashes the
  variance of strategy comparisons — the single most important trick for making
  a noisy simulator a reliable optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import RaceControl, SafetyCarModel, SafetyCarSampler
from ..types import Strategy
from .race import CarEntry, RaceSimulator


@dataclass(frozen=True)
class Scenario:
    race_control: RaceControl
    seed: int


@dataclass
class ScenarioSet:
    """A fixed bank of race scenarios for apples-to-apples strategy comparison."""

    scenarios: list[Scenario]
    n_laps: int

    @classmethod
    def sample(cls, sc_model: SafetyCarModel, n_laps: int, n: int, seed: int = 0) -> "ScenarioSet":
        rng = np.random.default_rng(seed)
        sampler = SafetyCarSampler(sc_model)
        scen = []
        for _ in range(n):
            sub = int(rng.integers(0, 2**31 - 1))
            rc = sampler.sample(n_laps, np.random.default_rng(sub))
            scen.append(Scenario(rc, sub))
        return cls(scen, n_laps)

    def __len__(self) -> int:
        return len(self.scenarios)


@dataclass
class EnsembleResult:
    focal_id: int
    n_runs: int
    grid_size: int
    positions: np.ndarray          # finishing position per run
    total_times: np.ndarray        # focal cumulative race time per run
    dnf_rate: float

    @property
    def mean_position(self) -> float:
        return float(np.mean(self.positions))

    @property
    def median_position(self) -> float:
        return float(np.median(self.positions))

    def p_finish(self, k: int) -> float:
        """Probability of finishing in the top-k (1=win, 3=podium, 10=points)."""
        return float(np.mean(self.positions <= k))

    @property
    def p_win(self) -> float:
        return self.p_finish(1)

    @property
    def p_podium(self) -> float:
        return self.p_finish(3)

    @property
    def p_points(self) -> float:
        return self.p_finish(10)

    @property
    def std_position(self) -> float:
        """Spread of finishing position — the strategy-variance risk measure
        (sensitive to SC/traffic exposure, not swamped by reliability)."""
        return float(np.std(self.positions))

    def p_loses_to(self, grid: int) -> float:
        """Probability of finishing *worse* than the starting grid slot."""
        return float(np.mean(self.positions > grid))

    def cvar_position(self, alpha: float = 0.10) -> float:
        """Conditional Value-at-Risk: mean finishing position in the worst
        ``alpha`` fraction of races (higher = worse). The tail-risk measure the
        risk-frontier selector uses."""
        n = max(1, int(np.ceil(alpha * len(self.positions))))
        worst = np.sort(self.positions)[-n:]
        return float(np.mean(worst))

    @property
    def mean_time_finishers(self) -> float:
        """Mean cumulative race time over finishing runs only (DNFs carry inf)."""
        finite = self.total_times[np.isfinite(self.total_times)]
        return float(np.mean(finite)) if finite.size else float("nan")

    def summary(self) -> dict:
        return {
            "mean_pos": round(self.mean_position, 3),
            "median_pos": self.median_position,
            "p_win": round(self.p_win, 3),
            "p_podium": round(self.p_podium, 3),
            "p_points": round(self.p_points, 3),
            "cvar10_pos": round(self.cvar_position(0.10), 3),
            "dnf_rate": round(self.dnf_rate, 3),
            "mean_time_s": round(self.mean_time_finishers, 2),
        }


def evaluate(
    field: list[CarEntry],
    scenarios: ScenarioSet,
    focal_id: int = 99,
    *,
    use_native: bool = False,
) -> EnsembleResult:
    """Replay ``field`` (focal car already inserted) across every scenario.

    With ``use_native=True`` and the Rust extension built, the inner loop runs in
    Rust for supported settings;
    otherwise the pure-Python simulator runs (the source of truth)."""
    if use_native:
        from .native import backend_for, simulate_batch_native_full
        if backend_for(field, scenarios) == "Rust":
            pos, times = simulate_batch_native_full(field, scenarios, focal_id)
            return EnsembleResult(
                focal_id=focal_id, n_runs=len(scenarios), grid_size=len(field),
                positions=pos.astype(float), total_times=times,
                dnf_rate=float(np.mean(~np.isfinite(times))),
            )
    sim = RaceSimulator(field, scenarios.n_laps)
    grid_size = len(field)
    pos = np.empty(len(scenarios), dtype=float)
    times = np.empty(len(scenarios), dtype=float)
    dnf = 0
    for k, scen in enumerate(scenarios.scenarios):
        rng = np.random.default_rng(scen.seed)
        res = sim.run_once(rng, race_control=scen.race_control, focal_id=focal_id)
        pos[k] = res.position[focal_id]
        times[k] = res.total_time[focal_id]
        if focal_id in res.retired:
            dnf += 1
    return EnsembleResult(
        focal_id=focal_id,
        n_runs=len(scenarios),
        grid_size=grid_size,
        positions=pos,
        total_times=times,
        dnf_rate=dnf / max(1, len(scenarios)),
    )
