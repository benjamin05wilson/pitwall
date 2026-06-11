"""Stochastic Safety-Car / Virtual-Safety-Car process.

Samples whole-race neutralisation phases from the Heilmeier categorical fit. The
simulator consumes a :class:`RaceControl` (a per-lap regime vector) so every car
in one simulated race shares the same SC/VSC timeline — neutralisations are a
*race-level* event, not independent per car.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import SafetyCarModel

GREEN, SC, VSC, RED, WET = "green", "SC", "VSC", "RED", "WET"


@dataclass
class RaceControl:
    """Per-lap race-control regime for a single simulated race (1-indexed laps;
    index 0 unused). ``regime[L]`` is one of green / SC / VSC / RED.

    ``wetness[L]`` is the track wetness in [0, 1] on lap L (0 = dry). It is an
    independent axis from the neutralisation regime — a race can be wet *and*
    green — so the wet lap-time model and the tyre-crossover optimiser read it
    directly. Empty/all-zero means a dry race (the common case)."""

    n_laps: int
    regime: list[str] = field(default_factory=list)
    wetness: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.regime:
            self.regime = [GREEN] * (self.n_laps + 1)
        if not self.wetness:
            self.wetness = [0.0] * (self.n_laps + 1)

    def at(self, lap: int) -> str:
        if lap < 0 or lap >= len(self.regime):
            return GREEN
        return self.regime[lap]

    def wet_at(self, lap: int) -> float:
        if lap < 0 or lap >= len(self.wetness):
            return 0.0
        return self.wetness[lap]

    @property
    def is_wet_race(self) -> bool:
        return any(w > 0.0 for w in self.wetness)

    @property
    def sc_laps(self) -> list[int]:
        return [l for l in range(1, self.n_laps + 1) if self.regime[l] == SC]

    def is_restart_lap(self, lap: int) -> bool:
        """First green lap immediately after a neutralisation ends."""
        return self.at(lap) == GREEN and self.at(lap - 1) in (SC, VSC)


def _start_lap_from_bucket(bucket: int, n_laps: int, rng: np.random.Generator) -> int:
    """Map a start-lap bucket to a concrete lap. Bucket 0 == lap 1; buckets 1..5
    are the 20%-width windows of race distance."""
    if bucket == 0:
        return 1
    lo = (bucket - 1) / 5.0
    hi = bucket / 5.0
    frac = rng.uniform(lo, hi)
    return int(np.clip(round(frac * n_laps), 1, n_laps))


class SafetyCarSampler:
    """Samples a :class:`RaceControl` timeline for one race."""

    def __init__(self, model: SafetyCarModel):
        self.m = model

    def _n_phases(self, rng: np.random.Generator) -> int:
        probs = np.asarray(self.m.sc_count_probs, dtype=float)
        # Apply a per-circuit P(>=1 SC) override by rescaling the >=1 mass.
        if self.m.p_at_least_one_override is not None:
            p1 = float(self.m.p_at_least_one_override)
            tail = probs[1:].sum()
            scaled = probs.copy()
            scaled[0] = 1.0 - p1
            if tail > 0:
                scaled[1:] = probs[1:] / tail * p1
            probs = scaled
        probs = probs / probs.sum()
        return int(rng.choice(len(probs), p=probs))

    def _duration(self, rng: np.random.Generator, probs: tuple[float, ...]) -> int:
        p = np.asarray(probs, dtype=float)
        p = p / p.sum()
        return int(rng.choice(np.arange(1, len(p) + 1), p=p))

    def sample(self, n_laps: int, rng: np.random.Generator) -> RaceControl:
        rc = RaceControl(n_laps)
        n_phases = self._n_phases(rng)

        start_w = np.asarray(self.m.sc_start_weights, dtype=float)
        start_w = start_w / start_w.sum()

        for _ in range(n_phases):
            bucket = int(rng.choice(len(start_w), p=start_w))
            start = _start_lap_from_bucket(bucket, n_laps, rng)
            dur = self._duration(rng, self.m.sc_duration_probs)
            for l in range(start, min(start + dur, n_laps + 1)):
                rc.regime[l] = SC

        # Independent VSC events triggered by mechanical failures.
        # Expected failures across the field ~ grid * team_failure_prob.
        n_fail = rng.poisson(20 * self.m.team_failure_prob)
        for _ in range(int(n_fail)):
            if rng.random() < self.m.p_vsc_per_failure:
                start = int(rng.integers(1, n_laps + 1))
                dur = self._duration(rng, self.m.vsc_duration_probs)
                for l in range(start, min(start + dur, n_laps + 1)):
                    if rc.regime[l] == GREEN:  # SC takes precedence
                        rc.regime[l] = VSC

        # Red flag: a rare full stoppage. Strategically pivotal because the rules
        # allow a *free* tyre change — a stop without the usual pit-loss.
        if rng.random() < self.m.red_flag_prob:
            l = int(rng.integers(2, n_laps))  # not lap 1 or the last lap
            rc.regime[l] = RED
        return rc
