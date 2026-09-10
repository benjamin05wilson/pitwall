"""Build a plausible grid of rivals for the focal car to race against.

For strategy work the focal car's plan is the decision variable; the rest of the
field is the *environment*. We spread race pace across the grid and hand each
rival a sensible nominal strategy so traffic, undercut exposure and track
position behave realistically.
"""

from __future__ import annotations

import numpy as np

from ..models import RaceModel
from ..types import Compound, Stint, Strategy
from .race import CarEntry


def nominal_strategy(n_laps: int, *, two_stop: bool = False, aggressive: bool = False) -> Strategy:
    """A reasonable default plan honouring the FIA two-compound rule."""
    if two_stop:
        a = n_laps // 3
        b = n_laps - 2 * a
        comps = (
            (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
            if aggressive
            else (Compound.MEDIUM, Compound.MEDIUM, Compound.HARD)
        )
        return Strategy([Stint(comps[0], a), Stint(comps[1], a), Stint(comps[2], b)])
    first = int(round(n_laps * (0.45 if not aggressive else 0.38)))
    first = max(1, min(n_laps - 1, first))
    c0, c1 = (Compound.SOFT, Compound.HARD) if aggressive else (Compound.MEDIUM, Compound.HARD)
    return Strategy([Stint(c0, first), Stint(c1, n_laps - first)])


def build_field(
    circuit_id: str,
    *,
    n_cars: int = 20,
    pace_spread: float = 1.5,
    n_laps: int | None = None,
    seed: int = 0,
) -> list[CarEntry]:
    """Construct ``n_cars`` rivals. Grid 1 is fastest; race pace deltas fan out
    roughly linearly across the field with a little noise. Returns rivals only;
    the focal car is added by the caller (see :func:`with_focal`)."""
    rng = np.random.default_rng(seed)
    base = RaceModel.for_circuit(circuit_id, n_laps=n_laps)
    laps = base.config.n_laps
    entries: list[CarEntry] = []
    for grid in range(1, n_cars + 1):
        frac = (grid - 1) / max(1, n_cars - 1)
        delta = frac * pace_spread + rng.normal(0, 0.08)
        # Faster cars tend to carry a touch more straight-line/peak performance.
        top_speed = (0.5 - frac) * 6.0 + rng.normal(0, 2.0)
        two_stop = (grid % 2 == 0)
        aggressive = grid > n_cars * 0.6  # midfield/back gambles more
        entries.append(
            CarEntry(
                car_id=grid,
                model=base.with_driver(delta),
                strategy=nominal_strategy(laps, two_stop=two_stop, aggressive=aggressive),
                grid=grid,
                top_speed_delta=float(top_speed),
                name=f"CAR{grid:02d}",
            )
        )
    return entries


def with_focal(
    rivals: list[CarEntry],
    focal_strategy: Strategy,
    *,
    circuit_id: str,
    focal_grid: int = 1,
    focal_delta: float | None = None,
    focal_model: RaceModel | None = None,
    n_laps: int | None = None,
    focal_id: int = 99,
    focal_top_speed: float = 0.0,
) -> list[CarEntry]:
    """Insert/replace the focal car (default car_id 99) into a field, removing any
    rival that occupied the focal grid slot so positions stay consistent.

    A supplied model is preserved; only an explicit focal_delta replaces its
    driver offset. Rivals are left unchanged (build_field uses generic defaults)."""
    base = focal_model if focal_model is not None else RaceModel.for_circuit(circuit_id, n_laps=n_laps)
    if base.config.circuit_id != circuit_id or (n_laps is not None and base.config.n_laps != n_laps):
        raise ValueError("focal model circuit/lap count differs from scoring context")
    focal = CarEntry(
        car_id=focal_id,
        model=base if focal_delta is None else base.with_driver(focal_delta),
        strategy=focal_strategy,
        grid=focal_grid,
        top_speed_delta=focal_top_speed,
        name="FOCAL",
    )
    kept = [e for e in rivals if e.grid != focal_grid][: max(0, len(rivals) - 1)]
    return [focal, *kept]
