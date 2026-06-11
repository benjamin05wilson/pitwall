"""Wet-weather tyre-crossover optimiser.

The defining decision of a wet race is *when* to swap between slicks,
intermediates and full wets as the track dries or wets. Given a per-lap track
wetness timeline (``models.RaceControl.wetness`` / ``data.wetness_timeline``) and
a calibrated :class:`WetModel`, this finds the tyre sequence that minimises total
race time by dynamic programming over (lap, tyre) states — every lap you either
stay out or take the pit-loss to switch compound.

Because the wet pace deficits (≈9 s/lap for inters, ≈18 s for wets) dwarf
in-stint degradation, the DP ignores tyre age and is dominated by getting the
*compound* right at every wetness — exactly the call a strategist sweats over a
drying track. The output is the optimal switch laps and a lap-by-lap "should be
on" trace to lay against what a team actually did.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Compound

# Slicks are represented by MEDIUM (the crossover is about slick-vs-wet, not the
# specific slick); INTERMEDIATE and WET are the wet-weather tyres.
WET_TYRES = (Compound.MEDIUM, Compound.INTERMEDIATE, Compound.WET)
_LABEL = {Compound.MEDIUM: "SLICK", Compound.INTERMEDIATE: "INTER", Compound.WET: "WET"}


@dataclass
class WetPlan:
    tyres: list[Compound]                 # chosen tyre per lap (1-indexed; [0] padding)
    switches: list[tuple[int, Compound]]  # (lap, new tyre) at each change
    total_time: float
    best_trace: list[Compound]            # the unconstrained fastest tyre each lap

    def label(self) -> str:
        if not self.switches:
            return f"stay on {_LABEL[self.tyres[1]]}"
        first = _LABEL[self.tyres[1]]
        rest = " → ".join(f"{_LABEL[c]}@L{lp}" for lp, c in self.switches)
        return f"{first} → {rest}"


def optimal_wet_strategy(model, wetness: list[float], *, start_tyre: Compound | None = None,
                         pit_loss: float | None = None) -> WetPlan:
    """DP for the minimum-time tyre sequence over a wetness timeline.

    ``wetness`` is 1-indexed (index 0 unused), length ``n_laps + 1``. ``start_tyre``
    fixes the grid tyre if given (else lap 1 is free). ``pit_loss`` defaults to the
    model's calibrated pit loss."""
    n = len(wetness) - 1
    if n < 1:
        return WetPlan([Compound.MEDIUM], [], 0.0, [])
    pit = pit_loss if pit_loss is not None else model.pit.pit_loss_s
    base = model.pace.base
    n_laps = model.config.n_laps

    def lap_cost(t: Compound, lap: int) -> float:
        return base + model.fuel.penalty(lap, n_laps) + model.wet.offset(t, wetness[lap])

    K = len(WET_TYRES)
    INF = float("inf")
    dp = [INF] * K
    back: list[list[int]] = [[-1] * K for _ in range(n + 1)]

    for ti, t in enumerate(WET_TYRES):
        if start_tyre is not None and t != start_tyre:
            continue
        dp[ti] = lap_cost(t, 1)

    for lap in range(2, n + 1):
        nxt = [INF] * K
        for ti, t in enumerate(WET_TYRES):
            lc = lap_cost(t, lap)
            best, arg = INF, -1
            for pj in range(K):
                if dp[pj] == INF:
                    continue
                cand = dp[pj] + (0.0 if pj == ti else pit) + lc
                if cand < best:
                    best, arg = cand, pj
            nxt[ti] = best
            back[lap][ti] = arg
        dp = nxt

    end = min(range(K), key=lambda i: dp[i])
    total = dp[end]
    seq_idx = [0] * (n + 1)
    cur = end
    for lap in range(n, 0, -1):
        seq_idx[lap] = cur
        cur = back[lap][cur] if lap > 1 else cur
    tyres = [Compound.MEDIUM] + [WET_TYRES[seq_idx[lap]] for lap in range(1, n + 1)]
    switches = [(lap, tyres[lap]) for lap in range(2, n + 1) if tyres[lap] != tyres[lap - 1]]
    best_trace = [Compound.MEDIUM] + [model.wet.best_tyre(wetness[lap]) for lap in range(1, n + 1)]
    return WetPlan(tyres, switches, round(float(total), 1), best_trace)


def analyze_wet_race(year: int, gp: str, driver: str | None = None) -> dict:
    """Reconstruct a real wet race's wetness timeline and the optimal tyre
    crossover, alongside what a driver actually ran — the wet-weather analogue of
    the dry 'ghost'. Returns the wetness trace, the AI's switch laps, and (if a
    driver is named) their real stops, so you can see who switched too late."""
    import warnings

    import fastf1

    from .data.fastf1_loader import _enable_cache, wetness_timeline
    from .data.priors import apply_to_model
    from .models import RaceModel
    from .replay import _GP_TO_CIRCUIT

    _enable_cache()
    wetness = wetness_timeline(year, gp)
    if not wetness or max(wetness) < 0.05:
        return {"error": "dry race — no crossover to optimise"}
    n_laps = len(wetness) - 1
    cid = _GP_TO_CIRCUIT.get(gp, "bahrain")
    model = apply_to_model(RaceModel.for_circuit(cid, n_laps=n_laps), circuit_id=cid)

    plan = optimal_wet_strategy(model, wetness)
    out = {
        "meta": {"year": year, "gp": gp, "circuit_id": cid, "n_laps": n_laps,
                 "peak_wetness": round(max(wetness), 3),
                 "peak_lap": int(max(range(len(wetness)), key=lambda i: wetness[i]))},
        "wetness": [round(w, 3) for w in wetness],
        "ai": {"label": plan.label(), "switches": [[lp, c.value] for lp, c in plan.switches],
               "tyres": [c.value for c in plan.tyres[1:]]},
        "crossovers": {
            "slick_to_inter": model.wet.crossover(Compound.MEDIUM, Compound.INTERMEDIATE),
            "inter_to_wet": model.wet.crossover(Compound.INTERMEDIATE, Compound.WET),
        },
    }
    if driver:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ses = fastf1.get_session(year, gp, "R")
            ses.load(telemetry=False, weather=False, messages=False)
        dl = ses.laps[ses.laps["Driver"] == driver].sort_values("LapNumber")
        real = []
        last = None
        for _, lp in dl.iterrows():
            c = str(lp["Compound"])
            if c != last and c in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"):
                real.append([int(lp["LapNumber"]), c])
                last = c
        out["real"] = {"driver": driver, "stints": real}
    return out
