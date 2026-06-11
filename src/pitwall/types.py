"""Core domain types shared across the engine.

Kept dependency-free (stdlib only) so every layer — models, simulator, optimiser,
surrogate, live updater — speaks the same vocabulary without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Compound(str, Enum):
    """Dry slick compounds. The C1–C5 construction maps onto the per-weekend
    SOFT/MEDIUM/HARD nomination; we model at the SOFT/MEDIUM/HARD level because
    that is what is observable from timing data, and carry the nominal grade so a
    weekend's specific allocation (e.g. C2/C3/C4) can be attached when known."""

    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"

    @property
    def is_slick(self) -> bool:
        return self in (Compound.SOFT, Compound.MEDIUM, Compound.HARD)


# Conventional short codes used in timing screens / FastF1.
COMPOUND_CODE = {
    Compound.SOFT: "S",
    Compound.MEDIUM: "M",
    Compound.HARD: "H",
    Compound.INTERMEDIATE: "I",
    Compound.WET: "W",
}


@dataclass(frozen=True)
class Stint:
    """A continuous run on one set of tyres."""

    compound: Compound
    length: int  # number of laps completed on this set

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"Stint length must be positive, got {self.length}")


@dataclass(frozen=True)
class Strategy:
    """A full-race plan: an ordered sequence of stints that together cover the
    race distance. Pit laps are derived from the cumulative stint boundaries, so
    a 2-stop is simply three stints. Immutable + hashable so strategies can key a
    cache (important: the optimiser evaluates thousands of candidates)."""

    stints: tuple[Stint, ...]

    def __init__(self, stints: Sequence[Stint]) -> None:
        object.__setattr__(self, "stints", tuple(stints))
        if not self.stints:
            raise ValueError("A strategy needs at least one stint")

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def total_laps(self) -> int:
        return sum(s.length for s in self.stints)

    @property
    def pit_laps(self) -> tuple[int, ...]:
        """Laps on which the car pits (end of each non-final stint), 1-indexed."""
        laps, cum = [], 0
        for s in self.stints[:-1]:
            cum += s.length
            laps.append(cum)
        return tuple(laps)

    @property
    def compounds(self) -> tuple[Compound, ...]:
        return tuple(s.compound for s in self.stints)

    def uses_two_compounds(self) -> bool:
        """FIA dry-race rule: at least two different slick compounds must be used."""
        slicks = {s.compound for s in self.stints if s.compound.is_slick}
        return len(slicks) >= 2

    def label(self) -> str:
        return "-".join(COMPOUND_CODE[s.compound] for s in self.stints) + (
            f" ({'/'.join(map(str, self.pit_laps))})" if self.pit_laps else ""
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Strategy[{self.label()}]"


@dataclass(frozen=True)
class StartState:
    """Where a car begins (or resumes) the simulation."""

    grid_position: int = 1
    start_lap: int = 1
    # Optional gaps (s) to cars ahead/behind at the start lap, for traffic.
    gap_ahead: float = 1e9
    gap_behind: float = 1e9


@dataclass
class RaceConfig:
    """Everything circuit/race specific the simulator needs. Numeric defaults are
    placeholders here; the calibrated values live in ``models.params`` and the
    per-circuit library, and are injected when a race is built."""

    circuit_id: str
    n_laps: int
    pit_loss_s: float
    grid_size: int = 20
    name: str = ""
    overtaking_difficulty: float = 0.5  # 0 easy (Monza) .. 1 hard (Monaco)
    extra: dict = field(default_factory=dict)
