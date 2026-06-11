"""``RaceModel`` — the full, self-consistent bundle of sub-models the simulator
needs for one race, plus a per-circuit factory that wires in calibrated priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..types import RaceConfig
from .circuits import get_profile
from .lap_time import LapTimeModel
from .params import (
    FuelModel,
    OvertakeModel,
    PaceModel,
    PitModel,
    SafetyCarModel,
    TyreModel,
    WetModel,
)


@dataclass
class RaceModel:
    config: RaceConfig
    pace: PaceModel
    fuel: FuelModel = field(default_factory=FuelModel)
    tyres: TyreModel = field(default_factory=TyreModel)
    pit: PitModel = field(default_factory=PitModel)
    safety_car: SafetyCarModel = field(default_factory=SafetyCarModel)
    overtake: OvertakeModel = field(default_factory=OvertakeModel)
    wet: WetModel = field(default_factory=WetModel)

    @property
    def lap_time(self) -> LapTimeModel:
        return LapTimeModel(self.pace, self.fuel, self.tyres, self.config.n_laps, self.wet)

    def with_driver(self, driver_delta: float) -> "RaceModel":
        """Return a copy representing a car/driver that is ``driver_delta`` s/lap
        off the reference car (negative = faster)."""
        return replace(self, pace=replace(self.pace, driver_delta=driver_delta))

    @classmethod
    def for_circuit(
        cls,
        circuit_id: str,
        *,
        driver_delta: float = 0.0,
        n_laps: int | None = None,
        pole_ref_s: float | None = None,
        grid_size: int = 20,
    ) -> "RaceModel":
        prof = get_profile(circuit_id)
        laps = n_laps if n_laps is not None else prof.n_laps
        config = RaceConfig(
            circuit_id=prof.circuit_id,
            name=prof.name,
            n_laps=laps,
            pit_loss_s=prof.pit_loss_s,
            grid_size=grid_size,
            overtaking_difficulty=_difficulty(prof.overtake_threshold_s),
        )
        pace = PaceModel(
            t_quali_ref=pole_ref_s if pole_ref_s is not None else prof.pole_ref_s,
            driver_delta=driver_delta,
        )
        return cls(
            config=config,
            pace=pace,
            pit=PitModel(pit_loss_s=prof.pit_loss_s),
            safety_car=SafetyCarModel(p_at_least_one_override=prof.p_safety_car),
            overtake=OvertakeModel(threshold_s=prof.overtake_threshold_s),
        )


def _difficulty(threshold_s: float) -> float:
    """Map an overtaking threshold (1.2 easy .. 3.75 ~impossible) to 0..1."""
    return float(min(1.0, max(0.0, (threshold_s - 1.2) / (3.75 - 1.2))))
