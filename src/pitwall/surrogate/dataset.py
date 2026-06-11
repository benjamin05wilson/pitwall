"""Generate surrogate training data by sampling (context, strategy) pairs and
labelling each with a Monte-Carlo evaluation.

Labels are the *distributional* outcome the surrogate learns to emulate:
[mean finishing position, position spread, P(podium), P(points)]. To amortise the
expensive field build + scenario sampling, many random strategies are evaluated
against the same context/scenario bank before moving on.
"""

from __future__ import annotations

import numpy as np

from ..models import RaceModel
from ..models.circuits import CIRCUITS
from ..sim import ScenarioSet, build_field, evaluate, with_focal
from ..types import Compound, Stint, Strategy
from .features import encode

SLICKS = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
LABELS = ["mean_pos", "std_pos", "p_podium", "p_points"]


def random_strategy(n_laps: int, rng: np.random.Generator) -> Strategy:
    """Sample an FIA-legal strategy with random stops/compounds/pit laps."""
    for _ in range(20):
        n_stops = int(rng.integers(1, 4))
        n_stints = n_stops + 1
        comps = [SLICKS[i] for i in rng.integers(0, 3, size=n_stints)]
        if len({c for c in comps}) < 2:
            comps[-1] = SLICKS[(SLICKS.index(comps[0]) + 1) % 3]
        # Random interior boundaries.
        cuts = sorted(rng.choice(range(4, n_laps - 3), size=n_stops, replace=False))
        bounds = [*cuts, n_laps]
        lengths, prev = [], 0
        for b in bounds:
            lengths.append(b - prev)
            prev = b
        if all(l >= 3 for l in lengths):
            return Strategy([Stint(c, l) for c, l in zip(comps, lengths)])
    # Fallback: simple 1-stop.
    h = n_laps // 2
    return Strategy([Stint(Compound.MEDIUM, h), Stint(Compound.HARD, n_laps - h)])


def generate_dataset(
    n_samples: int = 2000, *, n_scenarios: int = 120, strategies_per_context: int = 10,
    seed: int = 0, circuits: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    circuits = circuits or list(CIRCUITS.keys())
    X, Y = [], []
    n_contexts = max(1, n_samples // strategies_per_context)
    for _ in range(n_contexts):
        circuit = circuits[int(rng.integers(0, len(circuits)))]
        focal_grid = int(rng.integers(1, 21))
        pace_spread = float(rng.uniform(1.0, 2.2))
        # Sample pace *independently* of grid so the surrogate covers the whole
        # what-if space (e.g. a fast car starting at the back), not just the
        # grid==pace diagonal. Half the samples track grid, half are free.
        if rng.random() < 0.5:
            focal_delta = float((focal_grid - 1) / 19 * pace_spread + rng.normal(0, 0.15))
        else:
            focal_delta = float(rng.uniform(-0.4, 2.4))
        model = RaceModel.for_circuit(circuit)
        n_laps = model.config.n_laps
        rivals = build_field(circuit, n_cars=20, pace_spread=pace_spread,
                             seed=int(rng.integers(0, 1_000_000)))
        scen = ScenarioSet.sample(model.safety_car, n_laps, n_scenarios,
                                  seed=int(rng.integers(0, 1_000_000)))
        for _ in range(strategies_per_context):
            strat = random_strategy(n_laps, rng)
            field = with_focal(rivals, strat, circuit_id=circuit, focal_grid=focal_grid,
                               focal_delta=focal_delta)
            ens = evaluate(field, scen)
            X.append(encode(model, strat, focal_grid=focal_grid,
                            focal_delta=focal_delta, pace_spread=pace_spread))
            Y.append([ens.mean_position, ens.std_position, ens.p_podium, ens.p_points])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), LABELS
