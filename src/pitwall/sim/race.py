"""Single-race, lap-discretized, multi-car simulator.

This is the **pure-Python reference implementation** — readable and the source of
truth that the Rust hot loop (``pitwall._native``) is validated bit-for-bit
against. Each lap, for every running car, we:

  1. compute a green lap time from the calibrated model (+ Gaussian noise),
  2. apply the neutralisation regime (SC/VSC slow-down + fractional tyre/fuel use),
  3. apply pit loss (cheap under SC/VSC) and switch compound on a stop lap,
  4. bunch the field behind a ghost SC car (SC only — VSC preserves gaps),
  5. resolve track position: a faster car is held ``min_follow_gap`` behind a
     slower one unless its pace advantage clears the circuit overtaking threshold
     (DRS- and top-speed-adjusted) and a stochastic pass succeeds.

Classification is by (laps completed desc, cumulative race time asc).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import GREEN, RED, SC, VSC, RaceControl, RaceModel, SafetyCarSampler
from ..types import Compound, Strategy


@dataclass
class CarEntry:
    """One car on the grid: its calibrated model, planned strategy, and start
    position. ``pace_rank`` (0 = fastest) seeds a small top-speed proxy used by
    the overtaking model."""

    car_id: int
    model: RaceModel
    strategy: Strategy
    grid: int = 1
    top_speed_delta: float = 0.0  # km/h vs reference (affects overtaking threshold)
    name: str = ""
    retire_lap: int | None = None  # force a retirement on this lap (e.g. real DNFs)


@dataclass
class RaceResult:
    order: list[int]                  # car_ids, winner first
    position: dict[int, int]          # car_id -> finishing position (1-indexed)
    total_time: dict[int, float]      # car_id -> cumulative race time (s)
    laps_completed: dict[int, int]
    retired: set[int]
    race_control: RaceControl
    # Optional per-lap cumulative-time history for a focal car (for plotting).
    focal_id: int | None = None
    focal_lap_time: np.ndarray | None = None
    focal_position: np.ndarray | None = None
    focal_pits: list | None = None        # realized (lap, compound) for an adaptive focal


class RaceSimulator:
    def __init__(self, entries: list[CarEntry], n_laps: int, *, grid_gap: float = 1.6):
        if not entries:
            raise ValueError("Need at least one car")
        self.entries = entries
        self.n_laps = n_laps
        self.n_cars = len(entries)
        # Seconds of initial cumulative-time offset per grid slot (track position).
        self.grid_gap = grid_gap
        # SC sampler keyed off the first car's model (race-level event).
        self._sc_sampler = SafetyCarSampler(entries[0].model.safety_car)
        # Precompute per-car pit schedules (lap -> new compound) and lap models.
        self._pit_plan: list[dict[int, Compound]] = []
        self._start_compound: list[Compound] = []
        for e in entries:
            plan: dict[int, Compound] = {}
            for pit_lap, comp in zip(e.strategy.pit_laps, e.strategy.compounds[1:]):
                plan[pit_lap] = comp
            self._pit_plan.append(plan)
            self._start_compound.append(e.strategy.compounds[0])

    # ------------------------------------------------------------------ #
    def run_once(
        self,
        rng: np.random.Generator,
        race_control: RaceControl | None = None,
        focal_id: int | None = None,
        policies: dict | None = None,
    ) -> RaceResult:
        """``policies`` maps a car_id to an adaptive pit policy
        ``fn(lap, compound, age, regime, used_compounds) -> Compound | None``.
        Such a car ignores its fixed plan and decides pits live each lap — used
        to follow the AI re-optimiser, which reacts to safety cars."""
        n, L = self.n_cars, self.n_laps
        rc = race_control or self._sc_sampler.sample(L, rng)
        pol = {i: policies[e.car_id] for i, e in enumerate(self.entries)
               if policies and e.car_id in policies}
        # Per-compound *set counts* used so far (incl. the starting set), so an
        # adaptive policy can respect the weekend's tyre allocation. Iterating the
        # dict yields the compounds, so policies treating it as a set still work.
        used_c = {i: {self._start_compound[i]: 1} for i in pol}
        pit_log: dict[int, list] = {i: [] for i in pol}

        # Stagger the start by grid slot so track position is real: a car must
        # actually fight past the cars ahead. Without this, grid is meaningless.
        cum = np.array([(e.grid - 1) * self.grid_gap for e in self.entries], dtype=float)
        age = np.zeros(n, dtype=float)
        compound = list(self._start_compound)
        retired = np.zeros(n, dtype=bool)
        laps_done = np.zeros(n, dtype=int)

        # Sample retirements up front (lap on which each failing car stops).
        retire_lap = np.full(n, L + 1, dtype=int)
        for i, e in enumerate(self.entries):
            if e.retire_lap is not None:                       # forced (real DNFs)
                retire_lap[i] = e.retire_lap
            elif rng.random() < e.model.safety_car.team_failure_prob:
                retire_lap[i] = int(rng.integers(1, L + 1))

        lt = [e.model.lap_time for e in self.entries]
        ov = self.entries[0].model.overtake  # circuit-level overtaking params
        sigma = [e.model.pace.lap_noise_sigma for e in self.entries]

        focal_lap_time = np.full(L + 1, np.nan) if focal_id is not None else None
        focal_pos = np.full(L + 1, 0, dtype=int) if focal_id is not None else None

        green_pace = np.zeros(n)  # this lap's deterministic green pace (for overtaking)
        last_restart = -10

        for lap in range(1, L + 1):
            regime = rc.at(lap)
            wet = rc.wet_at(lap)
            if regime == GREEN and rc.is_restart_lap(lap):
                last_restart = lap

            for i in range(n):
                if retired[i]:
                    continue
                if lap >= retire_lap[i]:
                    retired[i] = True
                    cum[i] = np.inf
                    continue

                # In the wet the crossover-tyre model replaces the dry deg curve.
                gp = (lt[i].wet_lap(compound[i], int(age[i]), lap, wet) if wet > 0.0
                      else lt[i].green_lap(compound[i], int(age[i]), lap))
                green_pace[i] = gp

                if regime == GREEN:
                    lap_t = gp + rng.normal(0.0, sigma[i])
                    age[i] += 1.0
                elif regime == SC:
                    lap_t = gp * self.entries[i].model.safety_car.lap_mult_sc
                    age[i] += self.entries[i].model.tyres.deg_mult_sc
                elif regime == VSC:
                    lap_t = gp * self.entries[i].model.safety_car.lap_mult_vsc
                    age[i] += self.entries[i].model.tyres.deg_mult_vsc
                else:  # RED flag: slow neutralised lap; tyres are reset FOR FREE.
                    lap_t = gp * self.entries[i].model.safety_car.lap_mult_sc

                # Pit stop at the end of this lap? Adaptive cars decide live.
                if i in pol:
                    new_comp = pol[i](lap, compound[i], int(age[i]), regime, used_c[i])
                else:
                    new_comp = self._pit_plan[i].get(lap)
                if new_comp is not None:
                    # A scheduled stop under a red flag is free (no pit loss).
                    # Otherwise pay the (optionally noisy) pit-lane loss — real
                    # pit-crew variance shows up in the outcome spread.
                    lap_t += 0.0 if regime == RED else self.entries[i].model.pit.sample_loss(regime, rng)
                    compound[i] = new_comp
                    age[i] = 0.0
                    if i in pol:
                        used_c[i][new_comp] = used_c[i].get(new_comp, 0) + 1
                        pit_log[i].append((lap, new_comp.value))
                elif regime == RED:
                    age[i] = 0.0  # free fresh tyres for everyone under a red flag

                cum[i] += lap_t
                laps_done[i] = lap
                if focal_id is not None and self.entries[i].car_id == focal_id:
                    focal_lap_time[lap] = lap_t

            # Field bunching under SC / red flag: compress to a train.
            if regime in (SC, RED):
                self._bunch_field(cum, retired)

            # Track-position / overtaking resolution (green racing only).
            if regime == GREEN:
                drs_live = lap >= ov.drs_enable_lap and (lap - last_restart) > ov.drs_sc_delay
                self._resolve_positions(cum, green_pace, retired, ov, drs_live, rng)

            if focal_id is not None:
                fi = next(k for k, e in enumerate(self.entries) if e.car_id == focal_id)
                focal_pos[lap] = self._position_of(cum, laps_done, retired, fi)

        res = self._classify(cum, laps_done, retired, rc, focal_id, focal_lap_time, focal_pos)
        if focal_id is not None:
            fi = next((k for k, e in enumerate(self.entries) if e.car_id == focal_id), None)
            if fi in pit_log:
                res.focal_pits = pit_log[fi]
        return res

    # ------------------------------------------------------------------ #
    def _bunch_field(self, cum: np.ndarray, retired: np.ndarray) -> None:
        """Pull running cars into a nose-to-tail train behind the leader."""
        running = np.where(~retired)[0]
        if running.size <= 1:
            return
        order = running[np.argsort(cum[running])]
        sc_gap = 1.0  # ~1 s between cars in the SC train
        lead_time = cum[order[0]]
        for k, c in enumerate(order):
            cum[c] = lead_time + k * sc_gap

    def _resolve_positions(
        self,
        cum: np.ndarray,
        green_pace: np.ndarray,
        retired: np.ndarray,
        ov,
        drs_live: bool,
        rng: np.random.Generator,
    ) -> None:
        """Hold a faster car behind a slower one unless it can pass. Processed
        front-to-back so clamps cascade through a train. Track position emerges
        from the min-gap clamp; the pace advantage needed scales with the circuit
        threshold, reduced by DRS and a top-speed proxy."""
        running = np.where(~retired)[0]
        if running.size <= 1:
            return
        order = list(running[np.argsort(cum[running])])
        dirty = ov.dirty_air_loss_s > 0.0
        for k in range(1, len(order)):
            ahead, cur = order[k - 1], order[k]
            gap = cum[cur] - cum[ahead]
            if gap >= ov.min_follow_gap:
                # Not challenging, but still in dirty air if close enough: a real
                # pace tax that gives pitting-to-clear-air genuine strategic value.
                if dirty and gap < ov.dirty_air_gap:
                    cum[cur] += ov.dirty_air_loss_s
                continue
            pace_adv = green_pace[ahead] - green_pace[cur]  # >0 means `cur` faster
            threshold = ov.threshold_s
            if drs_live and gap <= ov.drs_window:
                threshold += ov.drs_bonus  # drs_bonus is negative -> easier
            threshold += ov.velocity_mod * self.entries[cur].top_speed_delta
            passed = pace_adv >= threshold and rng.random() < ov.p_overtake
            if passed:
                cum[ahead] += ov.loser_penalty  # passed car loses a little
            else:
                # Held up: clamp behind and bleed surplus; both pay a duel cost,
                # and the trailing car eats the dirty-air pace loss.
                cum[cur] = cum[ahead] + ov.min_follow_gap + ov.duel_penalty
                cum[ahead] += ov.duel_penalty * 0.5
                if dirty:
                    cum[cur] += ov.dirty_air_loss_s

    @staticmethod
    def _position_of(cum, laps_done, retired, i) -> int:
        key_i = (laps_done[i], -cum[i])
        pos = 1
        for j in range(len(cum)):
            if j == i:
                continue
            key_j = (laps_done[j], -cum[j])
            if key_j > key_i:
                pos += 1
        return pos

    def _classify(self, cum, laps_done, retired, rc, focal_id, focal_lt, focal_pos) -> RaceResult:
        idx = list(range(self.n_cars))
        # More laps first; then less cumulative time.
        idx.sort(key=lambda i: (-laps_done[i], cum[i]))
        order = [self.entries[i].car_id for i in idx]
        position = {self.entries[i].car_id: p + 1 for p, i in enumerate(idx)}
        total_time = {self.entries[i].car_id: float(cum[i]) for i in idx}
        laps_map = {self.entries[i].car_id: int(laps_done[i]) for i in idx}
        retired_ids = {self.entries[i].car_id for i in idx if retired[i]}
        return RaceResult(
            order=order,
            position=position,
            total_time=total_time,
            laps_completed=laps_map,
            retired=retired_ids,
            race_control=rc,
            focal_id=focal_id,
            focal_lap_time=focal_lt,
            focal_position=focal_pos,
        )
