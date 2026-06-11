"""Always-on live strategy re-optimiser — "Google Maps for the pit wall".

Every lap (and the instant anything changes — a Safety Car, a yellow, a
degradation spike, a rival's stop) this sweeps **every remaining strategy** from
the car's *current* state and returns the best one for the rest of the race:

    remaining stops × compound for each × pit lap for each

The current stint is priced off the **live Kalman belief** (the car's measured
pace + degradation *right now*); fresh stints use the calibrated per-compound
offsets/slopes. Pit loss is **cheap inside a Safety-Car / VSC window**, which is
what makes the recommendation flip to "BOX NOW" the moment an SC deploys — the
single highest-value decision in F1 strategy.

O(1)-per-plan via prefix sums, so it evaluates thousands of remaining strategies
in a millisecond and can run continuously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Compound

SLICKS = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)


@dataclass
class RemainingPlan:
    stops: tuple[tuple[int, Compound], ...]   # (pit_lap, new_compound) for each remaining stop
    final_compound: Compound
    rem_time: float

    def label(self) -> str:
        if not self.stops:
            return f"no stop · run to flag on {self.final_compound.value[0]}"
        parts = [f"L{p}→{c.value[0]}" for p, c in self.stops]
        return " · ".join(parts)


def reoptimize(
    model, *, lap: int, compound: Compound, age: int,
    belief_a: float | None, belief_b: float | None,
    regime: str = "green", used_compounds: set | None = None,
    max_more_stops: int = 2, sc_window: int = 3,
    allocation: dict | None = None, set_counts: dict | None = None,
) -> dict:
    """Sweep every remaining strategy from the current state; return the best,
    the live decision, and how many outcomes were evaluated.

    ``allocation`` caps the sets of each compound the car may still use; combined
    with ``set_counts`` (sets already used, including the current tyre) it stops
    the live engine recommending a stint a team has no tyres for."""
    n = model.config.n_laps
    comps = model.tyres.compounds
    k0 = {c: comps[c].k0 for c in SLICKS}
    # Thermal-aware slope: hotter track -> steeper wear, so the AI shortens stints
    # on a hot day exactly as a real strategist would.
    k1 = {c: model.tyres.slope(c) for c in SLICKS}
    a = belief_a if belief_a is not None else model.pace.base
    b = belief_b if belief_b is not None else model.tyres.slope(compound)
    b = max(0.0, b)
    used = set(used_compounds or {compound})
    base_pit = model.pit.pit_loss_s
    saving = (model.pit.sc_saving_frac if regime == "SC"
              else model.pit.vsc_saving_frac if regime == "VSC" else 0.0)

    # Prefix sums over fuel and the current-tyre lap costs (1-indexed laps).
    fuel = np.array([0.0] + [model.fuel.penalty(L, n) for L in range(1, n + 1)])
    Fpre = np.cumsum(fuel)  # Fpre[L] = sum fuel[1..L]
    cur = np.zeros(n + 1)
    for L in range(lap + 1, n + 1):
        cur[L] = a + b * (age + (L - lap)) + fuel[L]
    CURpre = np.cumsum(cur)

    def fuel_sum(lo, hi):  # inclusive laps lo..hi
        return float(Fpre[hi] - Fpre[lo - 1]) if hi >= lo else 0.0

    def cur_sum(lo, hi):
        return float(CURpre[hi] - CURpre[lo - 1]) if hi >= lo else 0.0

    def fresh_sum(c, lo, hi):  # fresh compound c, laps lo..hi (stint ages 1..m)
        m = hi - lo + 1
        if m <= 0:
            return 0.0
        off = a + (k0[c] - k0[compound])
        tri = m * (m + 1) / 2.0
        return m * off + k1[c] * tri + fuel_sum(lo, hi)

    def pit_cost(plap):
        if regime in ("SC", "VSC") and plap <= lap + sc_window:
            return base_pit * (1.0 - saving)
        return base_pit

    def legal(extra: set) -> bool:
        return len({c for c in (used | extra) if c.is_slick}) >= 2

    alloc = allocation or {}
    counts0 = dict(set_counts or {})

    def alloc_ok(added: tuple) -> bool:
        if not alloc:
            return True
        c = dict(counts0)
        for comp in added:
            c[comp] = c.get(comp, 0) + 1
        return all(c.get(k, 0) <= alloc.get(k, 99) for k in c)

    plans: list[RemainingPlan] = []
    n_eval = 0

    # 0 more stops — only legal if the two-compound rule is already satisfied.
    if legal(set()) and alloc_ok(()):
        plans.append(RemainingPlan((), compound, cur_sum(lap + 1, n)))
    n_eval += 1

    # 1 more stop.
    if max_more_stops >= 1:
        for p1 in range(lap + 1, n):
            t_cur = cur_sum(lap + 1, p1)
            for c1 in SLICKS:
                n_eval += 1
                if not legal({c1}) or not alloc_ok((c1,)):
                    continue
                t = t_cur + pit_cost(p1) + fresh_sum(c1, p1 + 1, n)
                plans.append(RemainingPlan(((p1, c1),), c1, t))

    # 2 more stops.
    if max_more_stops >= 2:
        for p1 in range(lap + 1, n - 1):
            t_cur = cur_sum(lap + 1, p1)
            pc1 = pit_cost(p1)
            for c1 in SLICKS:
                for p2 in range(p1 + 1, n):
                    seg1 = fresh_sum(c1, p1 + 1, p2)
                    for c2 in SLICKS:
                        n_eval += 1
                        if not legal({c1, c2}) or not alloc_ok((c1, c2)):
                            continue
                        t = t_cur + pc1 + seg1 + pit_cost(p2) + fresh_sum(c2, p2 + 1, n)
                        plans.append(RemainingPlan(((p1, c1), (p2, c2)), c2, t))

    if not plans:
        return {"decision": "RUN TO FLAG", "first_pit": None, "first_comp": None,
                "label": "no legal plan", "rem_time": 0.0, "n_eval": n_eval,
                "second_gap": 0.0, "alternatives": []}

    plans.sort(key=lambda p: p.rem_time)
    best = plans[0]
    second_gap = (plans[1].rem_time - best.rem_time) if len(plans) > 1 else 0.0

    if not best.stops:
        decision, first_pit, first_comp = "RUN TO FLAG", None, None
    else:
        first_pit, first_comp = best.stops[0]
        decision = "BOX NOW" if first_pit <= lap + 1 else "STAY OUT"

    return {
        "decision": decision, "first_pit": first_pit,
        "first_comp": first_comp.value if first_comp else None,
        "label": best.label(), "rem_time": round(best.rem_time, 1),
        "n_eval": n_eval, "second_gap": round(second_gap, 1),
        "remaining_stops": len(best.stops),
        "alternatives": [{"label": p.label(), "delta": round(p.rem_time - best.rem_time, 1)}
                         for p in plans[1:4]],
    }
