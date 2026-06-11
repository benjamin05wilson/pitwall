"""Calibrated model parameters.

Every default in this file is a *placeholder grounded in published work*, chosen
so the engine is credible out-of-the-box and then **re-fit per event** from real
FastF1 data (see ``pitwall.data.calibrate``). Sources are cited inline; the full
build-bible with confidence levels lives in ``docs/MODELLING.md``.

Primary source for functional forms + many constants: Heilmeier et al., "A Race
Simulation for Strategy Decisions in Circuit Motorsports" (TUM, IEEE ITSC 2018)
and its Monte-Carlo extension (Appl. Sci. 2020), with the TUMFTM/race-simulation
parameter files. Tyre-degradation magnitudes cross-checked against the Bayesian
state-space tyre study (arXiv:2512.00640).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Compound


# --------------------------------------------------------------------------- #
# Tyres                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompoundParams:
    """Per-compound tyre model: t_tyre(age) = k0 + k1*age + k2*age^2  (seconds).

    k0 is the fresh-tyre pace offset *relative to MEDIUM* (so MEDIUM.k0 == 0).
    k1 is the linear degradation slope (s/lap). k2 captures cliff curvature.
    """

    k0: float  # fresh-tyre offset rel. medium (s)
    k1: float  # linear degradation (s/lap)
    k2: float = 0.0003  # quadratic degradation (s/lap^2)

    def penalty(self, age: float) -> float:
        return self.k0 + self.k1 * age + self.k2 * age * age


# Ground-effect-era defaults. Adjacent compound step ~0.5 s/lap (Austria-2025
# state-space), NOT the FIA target gaps (S-H 2.2s / M-H 1.2s) which aren't
# realised in practice. Slopes from TUM YasMarina-2019 fits (0.01-0.09 s/lap).
DEFAULT_COMPOUNDS: dict[Compound, CompoundParams] = {
    # Realistic racing-era priors: soft is fastest fresh but degrades hard (early
    # cliff), hard is slow fresh but durable. This makes 2-stops and the hard
    # tyre genuinely competitive — as on abrasive tracks — rather than always
    # favouring a soft 1-stop. All values are re-fit per event by
    # pitwall.data.calibrate; compound pace deltas (~0.3 s/lap) are kept modest
    # because real data shows the FIA target gaps are rarely realised.
    Compound.SOFT: CompoundParams(k0=-0.30, k1=0.100, k2=0.0011),
    Compound.MEDIUM: CompoundParams(k0=0.0, k1=0.060, k2=0.0006),
    Compound.HARD: CompoundParams(k0=0.30, k1=0.035, k2=0.0003),
    # Wets are coarse placeholders; the engine is a dry-strategy tool in v1.
    Compound.INTERMEDIATE: CompoundParams(k0=6.0, k1=0.040, k2=0.0003),
    Compound.WET: CompoundParams(k0=12.0, k1=0.050, k2=0.0003),
}


@dataclass(frozen=True)
class TyreModel:
    compounds: dict[Compound, CompoundParams] = field(
        default_factory=lambda: dict(DEFAULT_COMPOUNDS)
    )
    cold_lap_penalty: float = 1.0  # +1.0 s on the out-lap (age == 0), TUM t_add_coldtires
    deg_mult_sc: float = 0.25  # tyre ages only fractionally under SC
    deg_mult_vsc: float = 0.5  # ...and under VSC
    # Thermal coupling fitted by data.calibrate (s/lap of slope per °C). At
    # ``track_temp == temp_ref`` it is inert, so default models are unchanged.
    temp_coeff: float = 0.0
    temp_ref: float = 30.0       # reference track temperature of the fit (°C)
    track_temp: float = 30.0     # the temperature this model is evaluated at (°C)

    def slope(self, compound: Compound) -> float:
        """Effective degradation slope incl. the thermal term: hotter track ->
        steeper wear. ``k1_eff = k1 + temp_coeff*(track_temp - temp_ref)``."""
        k1 = self.compounds[compound].k1
        return k1 + self.temp_coeff * (self.track_temp - self.temp_ref)

    def penalty(self, compound: Compound, age: float) -> float:
        cp = self.compounds[compound]
        # Replace the compound's intrinsic linear slope with the thermal-aware one.
        thermal = self.temp_coeff * (self.track_temp - self.temp_ref)
        pen = cp.penalty(age) + thermal * age
        if age == 0:
            pen += self.cold_lap_penalty
        return pen


# --------------------------------------------------------------------------- #
# Fuel + base pace                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FuelModel:
    """Linear fuel-mass effect: penalty = k_fuel * remaining_fuel_kg.

    NOTE: the 0.107 s/kg figure circulating from an arXiv PDF extraction is ~3x
    the established value and is rejected — 0.030 s/kg is the calibrated default
    (track range 0.019 short / low-sensitivity .. 0.036 long)."""

    k_fuel: float = 0.030  # s/kg
    m_start: float = 110.0  # kg, F1 max tank
    burn_per_lap: float | None = None  # None -> derive m_start / n_laps, floored at 1.6

    def burn(self, n_laps: int) -> float:
        if self.burn_per_lap is not None:
            return self.burn_per_lap
        return max(1.6, self.m_start / max(1, n_laps))

    def remaining_kg(self, lap: int, n_laps: int) -> float:
        # `lap` is 1-indexed; fuel at the *start* of the lap.
        return max(0.0, self.m_start - self.burn(n_laps) * (lap - 1))

    def penalty(self, lap: int, n_laps: int) -> float:
        return self.k_fuel * self.remaining_kg(lap, n_laps)


@dataclass(frozen=True)
class PaceModel:
    """Base green race pace = pole reference + race-pace gap + car/driver delta."""

    t_quali_ref: float  # pole-reference lap time for the circuit (s)
    race_pace_gap: float = 4.0  # s above pole, low-fuel-corrected (TUM ~4.0)
    driver_delta: float = 0.0  # car+driver performance delta vs reference (s/lap)
    lap_noise_sigma: float = 0.6  # per-lap Gaussian noise (driver range 0.46-0.86)

    @property
    def base(self) -> float:
        return self.t_quali_ref + self.race_pace_gap + self.driver_delta


# --------------------------------------------------------------------------- #
# Pit loss                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PitModel:
    """Pit time loss vs staying out, plus stochastic stationary time and the
    'cheap stop' discounts available under neutralisations."""

    pit_loss_s: float = 22.0  # total green-flag delta vs staying out
    stationary_mean: float = 2.4  # s, mean tyre-change standstill
    stationary_sd: float = 0.4
    sc_saving_frac: float = 0.50  # fraction of pit loss saved if pitting under SC
    vsc_saving_frac: float = 0.35  # ...under VSC
    double_stack_penalty: float = 2.5  # queue cost for 2nd car stacked same lap
    # Real per-team variation in the pit-lane loss (s). Default 0 -> deterministic
    # stops (preserves native parity); calibrated models set this from real crews.
    pit_loss_sd: float = 0.0

    def effective_loss(self, regime: str = "green") -> float:
        if regime == "SC":
            return self.pit_loss_s * (1.0 - self.sc_saving_frac)
        if regime == "VSC":
            return self.pit_loss_s * (1.0 - self.vsc_saving_frac)
        return self.pit_loss_s

    def sample_loss(self, regime: str = "green", rng=None) -> float:
        """Effective pit loss with real pit-crew variance folded in. With
        ``pit_loss_sd == 0`` (default) this equals :meth:`effective_loss`, so a
        slow stop or a rival's fast crew shows up in the outcome spread only when
        the model has been calibrated. The neutralisation discount is applied to
        the mean, not the noise (a slow stop is slow under green or SC alike)."""
        base = self.effective_loss(regime)
        if self.pit_loss_sd <= 0.0 or rng is None:
            return base
        # Clamp at a fast-stop floor so noise can't produce a free or negative stop.
        floor = base - self.pit_loss_s * 0.18
        return float(max(floor, base + rng.normal(0.0, self.pit_loss_sd)))


# --------------------------------------------------------------------------- #
# Safety car / VSC (Heilmeier 2014-2019 categorical fit)                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SafetyCarModel:
    # P(exactly k SC phases) for k = 0,1,2,3  ->  P(>=1 SC) = 0.545, E = 0.71
    sc_count_probs: tuple[float, ...] = (0.455, 0.413, 0.099, 0.033)
    # SC start-lap bucket weights over [lap1, <20%, <40%, <60%, <80%, <100%]
    sc_start_weights: tuple[float, ...] = (0.364, 0.136, 0.136, 0.080, 0.193, 0.091)
    # SC duration (laps) pmf for durations 1..10
    sc_duration_probs: tuple[float, ...] = (
        0.0, 0.182, 0.25, 0.227, 0.193, 0.057, 0.068, 0.023, 0.0, 0.0,
    )
    # VSC duration (laps) pmf for durations 1..4
    vsc_duration_probs: tuple[float, ...] = (0.479, 0.396, 0.021, 0.104)
    p_vsc_per_failure: float = 0.227  # P(VSC | a car fails)
    team_failure_prob: float = 0.06  # per-car mechanical DNF probability / race
    red_flag_prob: float = 0.07  # per-race probability of a red flag (free tyre change)

    lap_mult_sc: float = 1.6  # SC lap time = green * 1.6
    lap_mult_vsc: float = 1.4  # VSC lap time = green * 1.4
    fuel_mult_sc: float = 0.25  # fuel burned fractionally under SC
    fuel_mult_vsc: float = 0.5

    # Per-circuit override of P(>=1 SC); None -> use sc_count_probs as-is.
    p_at_least_one_override: float | None = None


# --------------------------------------------------------------------------- #
# Wet weather / tyre crossover                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WetModel:
    """Wet-weather lap-time model on a continuous track *wetness* ``w`` in [0, 1]
    (0 = dry line, 1 = standing water).

    For a tyre at wetness ``w`` the lap-time *offset* added to the dry green base
    is a convex bowl with a minimum at that tyre's optimal wetness — slicks have
    no bowl, they just get catastrophically slow as the track wets (aquaplaning),
    which is why a dry-tyre gamble in the rain is a disaster. The crossover laps
    (slick<->inter<->wet) fall out of where two offsets cross, so the strategy
    optimiser can pick the fastest tyre at every wetness on a drying or wetting
    track.

    The absolute pace deficits ``inter_floor`` / ``wet_floor`` (s/lap a wet tyre
    is slower than a dry slick at its best) are CALIBRATED from real wet races
    (``data.calibrate.calibrate_wet``); the response *curvature/optima* are
    physically-grounded priors pending track-wetness-sensor data.
    """

    slick_aqua: float = 95.0     # slicks: offset = slick_aqua * w^2 (undriveable wet)
    inter_floor: float = 7.0     # best-case intermediate deficit vs dry slick (s/lap)
    inter_opt: float = 0.45      # wetness at which intermediates are happiest
    inter_curv: float = 40.0     # bowl curvature (s per unit w^2)
    wet_floor: float = 12.0      # best-case full-wet deficit vs dry slick (s/lap)
    wet_opt: float = 0.85
    wet_curv: float = 24.0
    wear_per_lap: float = 0.03   # gentle wear on the operative wet tyre (s/lap of age)

    def offset(self, compound: Compound, w: float) -> float:
        """Lap-time offset (s) added to the dry base for ``compound`` at wetness
        ``w``. ``w <= 0`` is dry (slicks 0; wets carry their floor)."""
        if w <= 0.0:
            if compound.is_slick:
                return 0.0
            # On a bone-dry track, wet tyres overheat badly off their bowl.
            return (self.inter_floor + self.inter_curv * self.inter_opt ** 2
                    if compound == Compound.INTERMEDIATE
                    else self.wet_floor + self.wet_curv * self.wet_opt ** 2)
        if compound.is_slick:
            return self.slick_aqua * w * w
        if compound == Compound.INTERMEDIATE:
            return self.inter_floor + self.inter_curv * (w - self.inter_opt) ** 2
        return self.wet_floor + self.wet_curv * (w - self.wet_opt) ** 2

    def best_tyre(self, w: float):
        """The fastest tyre at wetness ``w`` (the tyre a perfect strategist runs)."""
        from ..types import Compound as _C
        cands = (_C.MEDIUM, _C.INTERMEDIATE, _C.WET)
        return min(cands, key=lambda c: self.offset(c, w))

    def crossover(self, dry_tyre: Compound, wet_tyre: Compound,
                  lo: float = 0.0, hi: float = 1.0) -> float:
        """Wetness at which ``wet_tyre`` becomes faster than ``dry_tyre`` (bisection
        on the offset difference). Returns ``hi`` if they never cross in range."""
        f = lambda w: self.offset(wet_tyre, w) - self.offset(dry_tyre, w)
        a, b = lo, hi
        if f(a) * f(b) > 0:
            return hi if f((a + b) / 2) > 0 else lo
        for _ in range(40):
            m = (a + b) / 2
            if f(a) * f(m) <= 0:
                b = m
            else:
                a = m
        return round((a + b) / 2, 3)


# --------------------------------------------------------------------------- #
# Overtaking / dirty air                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OvertakeModel:
    """Lap-by-lap positional model on cumulative time (TUM). A trailing car can
    pass when its per-lap pace advantage exceeds the circuit threshold."""

    threshold_s: float = 1.8  # t_gap_overtake: pace advantage (s/lap) needed to pass
    velocity_mod: float = -0.045  # s per km/h of top-speed delta (lowers threshold)
    p_overtake: float = 0.25  # stochastic success prob once threshold cleared
    drs_bonus: float = -0.45  # s/lap when within DRS window
    drs_window: float = 1.0  # s gap to enable DRS
    drs_enable_lap: int = 3  # DRS allowed from this lap
    drs_sc_delay: int = 2  # DRS disabled this many laps after an SC restart
    min_follow_gap: float = 0.5  # held this far behind when unable to pass (track position)
    duel_penalty: float = 0.3  # s/lap to both cars while fighting
    loser_penalty: float = 0.3  # s to the passed car on the lap a pass completes
    dirty_air_deg_mult: float = 1.1  # tyre wear multiplier while within ~1 s
    # Lap-time lost to dirty air while running within ``dirty_air_gap`` of the car
    # ahead (not just when challenging). Default 0 keeps native parity; calibrated
    # models turn it on so the AI sees the real value of pitting to clear traffic.
    dirty_air_loss_s: float = 0.0
    dirty_air_gap: float = 1.0  # s gap under which the dirty-air penalty applies
