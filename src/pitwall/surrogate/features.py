"""Fixed-length feature encoding of a (race-context, strategy) pair.

The surrogate must answer "what's the expected outcome of *this* strategy in
*this* race?" for any pit-lap/compound/stop-count combination, so we encode both
the context (circuit + the focal car's situation) and the strategy into one
fixed-length vector. Strategies have a variable number of stints, so we pad to
``MAX_STINTS`` and carry ``n_stops`` explicitly.
"""

from __future__ import annotations

import numpy as np

from ..models import RaceModel
from ..types import Compound, Strategy

MAX_STINTS = 4
_COMPOUND_ONEHOT = {
    Compound.SOFT: (1.0, 0.0, 0.0),
    Compound.MEDIUM: (0.0, 1.0, 0.0),
    Compound.HARD: (0.0, 0.0, 1.0),
    Compound.INTERMEDIATE: (0.0, 0.0, 0.0),
    Compound.WET: (0.0, 0.0, 0.0),
}

# Names line up with the encoded vector for explainability in the dashboard.
CONTEXT_FEATURES = [
    "n_laps", "pit_loss", "overtake_threshold", "p_safety_car",
    "focal_grid", "focal_delta", "pace_spread",
]
STRATEGY_SCALARS = ["n_stops", "soft_laps_frac", "med_laps_frac", "hard_laps_frac"]


def encode_context(model: RaceModel, *, focal_grid: int, focal_delta: float,
                   pace_spread: float) -> list[float]:
    return [
        model.config.n_laps / 70.0,
        model.pit.pit_loss_s / 30.0,
        model.overtake.threshold_s / 4.0,
        (model.safety_car.p_at_least_one_override or 0.545),
        focal_grid / 20.0,
        focal_delta / 3.0,
        pace_spread / 3.0,
    ]


def encode_strategy(strat: Strategy, n_laps: int) -> list[float]:
    feats: list[float] = [strat.n_stops / 3.0]
    # Compound-share of the race distance.
    laps_by = {Compound.SOFT: 0, Compound.MEDIUM: 0, Compound.HARD: 0}
    for s in strat.stints:
        if s.compound in laps_by:
            laps_by[s.compound] += s.length
    feats += [laps_by[c] / n_laps for c in (Compound.SOFT, Compound.MEDIUM, Compound.HARD)]
    # Per-stint: compound one-hot + length fraction + cumulative pit-lap fraction.
    cum = 0
    for i in range(MAX_STINTS):
        if i < len(strat.stints):
            st = strat.stints[i]
            feats += list(_COMPOUND_ONEHOT[st.compound])
            feats.append(st.length / n_laps)
            cum += st.length
            feats.append(cum / n_laps if i < len(strat.stints) - 1 else 1.0)
        else:
            feats += [0.0, 0.0, 0.0, 0.0, 1.0]
    return feats


def encode(model: RaceModel, strat: Strategy, *, focal_grid: int,
           focal_delta: float, pace_spread: float) -> np.ndarray:
    ctx = encode_context(model, focal_grid=focal_grid, focal_delta=focal_delta,
                         pace_spread=pace_spread)
    st = encode_strategy(strat, model.config.n_laps)
    return np.asarray(ctx + st, dtype=np.float32)


def feature_dim() -> int:
    # context (7) + strategy scalars (4) + per-stint (5 fields) * MAX_STINTS
    return len(CONTEXT_FEATURES) + 4 + 5 * MAX_STINTS
