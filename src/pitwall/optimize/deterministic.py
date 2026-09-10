"""Deterministic, free-track strategy optimisation.

Stage 1 of the optimiser. With no traffic, no safety cars and no noise, the
optimal pit laps for a *fixed* compound sequence follow from a small dynamic
program over stint boundaries (tyre degradation is convex, so the optimum is
well-behaved). Enumerating compound sequences on top yields a shortlist of
strong candidates in milliseconds, which Stage 2 (``robust.py``) then re-scores
under the full stochastic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations

import numpy as np

from ..models import RaceModel
from ..types import Compound, Stint, Strategy

SLICKS = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)

# Realistic dry-race set availability. The FIA hands out 13 sets (2 Hard / 3
# Medium / 8 Soft); after practice and qualifying a team typically arrives at
# the race with roughly this many *unused* sets of each compound. The optimiser
# may not recommend more stints on a compound than there are sets for it — the
# constraint that stops the engine proposing a plan a team physically can't run.
DEFAULT_ALLOCATION: dict[Compound, int] = {
    Compound.SOFT: 3, Compound.MEDIUM: 2, Compound.HARD: 2,
}


def allocation_ok(compounds, allocation: dict[Compound, int] | None) -> bool:
    """True if the per-compound stint (set) counts fit within ``allocation``."""
    if not allocation:
        return True
    counts: dict[Compound, int] = {}
    for c in compounds:
        counts[c] = counts.get(c, 0) + 1
    return all(counts[c] <= allocation.get(c, 99) for c in counts)


class StintCostTable:
    """Precomputed deterministic stint costs for one car model.

    ``cost[c][s, m]`` = green-flag time to run compound ``c`` for ``m`` laps
    starting at absolute lap ``s`` (fresh tyres, age 0 -> m-1, including the
    cold out-lap penalty and the lap-dependent fuel effect).
    """

    def __init__(self, model: RaceModel):
        self.n_laps = model.config.n_laps
        lt = model.lap_time
        N = self.n_laps
        self.cost: dict[Compound, np.ndarray] = {}
        for c in SLICKS:
            tab = np.full((N + 2, N + 2), np.inf)
            for s in range(1, N + 1):
                running = 0.0
                m = 0
                while s + m <= N:
                    running += lt.green_lap(c, m, s + m)  # age m on absolute lap s+m
                    m += 1
                    tab[s, m] = running
            self.cost[c] = tab


@dataclass(frozen=True)
class Candidate:
    strategy: Strategy
    det_time: float


def _optimal_boundaries(
    table: StintCostTable, compounds: tuple[Compound, ...], pit_loss: float
) -> tuple[list[int], float]:
    """DP for the minimum-time stint boundaries of a fixed compound sequence.

    Returns (boundaries, total_time) where boundaries are the cumulative lap
    counts at the end of each stint (last == n_laps)."""
    N = table.n_laps
    K = len(compounds)
    # f[k][l] = min cost for first k stints covering laps 1..l (incl. k-1 pits).
    f = np.full((K + 1, N + 1), np.inf)
    back = np.full((K + 1, N + 1), -1, dtype=int)
    f[0][0] = 0.0
    for k in range(1, K + 1):
        c = compounds[k - 1]
        ctab = table.cost[c]
        for l in range(k, N + 1):          # need >=1 lap per stint
            best, arg = np.inf, -1
            for lp in range(k - 1, l):     # previous boundary
                prev = f[k - 1][lp]
                if not np.isfinite(prev):
                    continue
                stint = ctab[lp + 1, l - lp]
                if not np.isfinite(stint):
                    continue
                add = pit_loss if k >= 2 else 0.0
                tot = prev + add + stint
                if tot < best:
                    best, arg = tot, lp
            f[k][l] = best
            back[k][l] = arg
    # Recover boundaries (cast off numpy ints so Strategy carries plain ints).
    bounds: list[int] = []
    l = N
    for k in range(K, 0, -1):
        bounds.append(int(l))
        l = int(back[k][l])
    bounds.reverse()
    return bounds, float(f[K][N])


def _to_strategy(compounds: tuple[Compound, ...], bounds: list[int]) -> Strategy:
    stints, prev = [], 0
    for c, b in zip(compounds, bounds):
        stints.append(Stint(c, b - prev))
        prev = b
    return Strategy(stints)


def enumerate_candidates(
    model: RaceModel,
    *,
    max_stops: int = 3,
    min_stops: int = 1,
    compounds_avail: tuple[Compound, ...] = SLICKS,
    mandatory_start: Compound | None = None,
    allocation: dict[Compound, int] | None = None,
    top_k: int | None = None,
) -> list[Candidate]:
    """Generate FIA-legal candidate strategies with deterministically optimal pit
    laps, sorted by free-track race time (fastest first). ``allocation`` caps the
    number of stints per compound to the sets actually available (so the engine
    never proposes a plan a team couldn't run); ``None`` leaves it unconstrained."""
    table = StintCostTable(model)
    seen: set[tuple] = set()
    out: list[Candidate] = []

    for n_stops in range(min_stops, max_stops + 1):
        n_stints = n_stops + 1
        # Order matters for the lap-dependent fuel effect, so permute; dedupe.
        for combo in combinations_with_replacement(compounds_avail, n_stints):
            for seq in sorted(set(permutations(combo)), key=lambda seq: tuple(c.value for c in seq)):
                if mandatory_start is not None and seq[0] != mandatory_start:
                    continue
                if len({c for c in seq if c.is_slick}) < 2:
                    continue  # FIA two-compound rule
                if not allocation_ok(seq, allocation):
                    continue  # not enough sets of some compound
                if seq in seen:
                    continue
                seen.add(seq)
                bounds, t = _optimal_boundaries(table, seq, model.pit.pit_loss_s)
                if np.isfinite(t):
                    out.append(Candidate(_to_strategy(seq, bounds), t))

    out.sort(key=lambda c: c.det_time)
    return out if top_k is None else out[:top_k]
