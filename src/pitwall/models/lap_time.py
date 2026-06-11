"""The lap-time master equation.

    t_lap(L) = t_base + k_fuel*fuel(L) + t_tyre(compound, age) + eps

This module returns the **green, deterministic** lap time and its decomposition.
Stochastic noise and neutralisation (SC/VSC) multipliers are applied by the
simulator, which owns the RNG and the race state — keeping this layer a pure,
unit-testable function of (compound, age, lap).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Compound
from .params import FuelModel, PaceModel, TyreModel, WetModel


@dataclass(frozen=True)
class LapComponents:
    base: float
    fuel: float
    tyre: float

    @property
    def total(self) -> float:
        return self.base + self.fuel + self.tyre


class LapTimeModel:
    def __init__(self, pace: PaceModel, fuel: FuelModel, tyres: TyreModel, n_laps: int,
                 wet: WetModel | None = None):
        self.pace = pace
        self.fuel = fuel
        self.tyres = tyres
        self.n_laps = n_laps
        self.wet = wet or WetModel()

    def components(self, compound: Compound, age: int, lap: int) -> LapComponents:
        return LapComponents(
            base=self.pace.base,
            fuel=self.fuel.penalty(lap, self.n_laps),
            tyre=self.tyres.penalty(compound, age),
        )

    def green_lap(self, compound: Compound, age: int, lap: int) -> float:
        """Deterministic green lap time (s)."""
        return self.components(compound, age, lap).total

    def wet_lap(self, compound: Compound, age: int, lap: int, wetness: float) -> float:
        """Deterministic lap time at track ``wetness`` in (0, 1]. The wet-weather
        offset replaces the dry tyre-degradation curve (slicks aquaplane; inters
        and wets carry their own gentle wear), on top of base pace + fuel."""
        base = self.pace.base + self.fuel.penalty(lap, self.n_laps)
        off = self.wet.offset(compound, wetness)
        wear = 0.0 if compound.is_slick else self.wet.wear_per_lap * age
        return base + off + wear
