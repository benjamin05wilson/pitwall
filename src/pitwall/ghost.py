"""Ghost counterfactual — "what if we'd followed the AI?"

Reconstructs the real race as a simulatable field (every driver on their *actual*
strategy and calibrated pace), then runs the chosen driver TWICE against that
field with common random numbers:

  * REAL  — their actual strategy (sanity-checks the sim reproduces reality)
  * GHOST — the strategy the AI engine recommends

and reports where each finishes plus a lap-by-lap position trace, so you can see
the ghost car's alternate race against what really happened.
"""

from __future__ import annotations

import warnings

import numpy as np

from .data.backtest import _real_strategies
from .data.calibrate import calibrate_race, calibrate_races
from .data.fastf1_loader import _enable_cache
from .models import RaceModel
from .replay import _GP_TO_CIRCUIT
from .sim import CarEntry, RaceSimulator, ScenarioSet


def _eval(focal_strategy, rivals, *, focal_id, focal_grid, focal_retire,
          focal_model, scen, policy=None):
    """Run the focal car against the fixed real field under the real race control.
    With ``policy`` set, the focal follows the AI re-optimiser live (adaptive —
    reacts to the actual safety cars) instead of a fixed plan. Returns the finish
    distribution, a representative position trace, and the focal's realized pits."""
    focal = CarEntry(car_id=focal_id, model=focal_model,
                     strategy=focal_strategy, grid=focal_grid, name="FOCAL", retire_lap=focal_retire)
    sim = RaceSimulator([focal, *rivals], scen.n_laps)
    policies = {focal_id: policy} if policy else None
    pos = np.empty(len(scen.scenarios))
    trace, pits, lap_times = None, None, None
    for k, s in enumerate(scen.scenarios):
        r = sim.run_once(np.random.default_rng(s.seed), race_control=s.race_control,
                         focal_id=focal_id, policies=policies)
        pos[k] = r.position[focal_id]
        if k == 0:
            trace = [int(x) for x in (r.focal_position[1:] if r.focal_position is not None else [])]
            pits = r.focal_pits
            # Per-lap lap times of the representative scenario (incl. pit loss),
            # so the cumulative real-vs-AI time gap can drive an on-track ghost.
            lap_times = (r.focal_lap_time.copy() if r.focal_lap_time is not None else None)
    return pos, trace, pits, lap_times


def _ai_policy(model, allocation: dict | None = None):
    """A live re-optimiser as a pit policy: box this lap iff the engine says BOX
    NOW — which a safety car triggers via its cheap-stop window. The car may only
    fit tyres it has sets for (``allocation``). Memoised because the decision is
    deterministic in (lap, compound, age, regime, used-set-counts)."""
    from .live_optimizer import reoptimize
    from .types import Compound

    memo: dict = {}
    SC_WINDOW = 3

    def policy(lap, compound, age, regime, used):
        # ``used`` is a {Compound: set-count} dict from the simulator.
        counts = {c: int(n) for c, n in used.items()}
        key = (lap, compound.value, age, regime,
               frozenset((c.value, n) for c, n in counts.items()))
        if key not in memo:
            r = reoptimize(model, lap=lap, compound=compound, age=age, belief_a=None,
                           belief_b=None, regime=regime, used_compounds=set(counts),
                           allocation=allocation, set_counts=counts)
            memo[key] = (r["decision"], r["first_pit"], r["first_comp"])
        dec, fp, fc = memo[key]
        if fc is None or fp is None:
            return None
        if regime in ("SC", "VSC"):
            # A neutralisation is out: if the plan wants to pit anytime in the
            # cheap window, grab it NOW — the window can close any lap.
            return Compound(fc) if fp <= lap + SC_WINDOW else None
        return Compound(fc) if dec == "BOX NOW" else None   # green: pit at the optimum
    return policy


def _infer_allocation(strategies: dict) -> dict:
    """Bound the AI to a realistic weekend tyre allocation: at most one set more
    of each compound than the most any car actually used that race, capped at the
    FIA totals (2 Hard / 3 Medium / 8 Soft). This keeps every real strategy
    feasible while stopping the ghost inventing sets a team never had."""
    from collections import Counter

    from .types import Compound
    fia = {Compound.SOFT: 8, Compound.MEDIUM: 3, Compound.HARD: 2}
    used_max = {Compound.SOFT: 1, Compound.MEDIUM: 1, Compound.HARD: 1}
    for strat in strategies.values():
        cnt = Counter(c for c in strat.compounds if c.is_slick)
        for c, k in cnt.items():
            used_max[c] = max(used_max.get(c, 1), k)
    return {c: min(used_max[c] + 1, fia[c]) for c in fia}


def _real_race_control(ses, n_laps: int):
    """Reconstruct the actual safety-car / VSC / red-flag timeline from the race."""
    from .models import RED, SC, VSC, RaceControl
    from .replay import _track_state
    rc = RaceControl(n_laps)
    laps = ses.laps
    for L in range(1, n_laps + 1):
        sub = laps[laps["LapNumber"] == L]
        if sub.empty:
            continue
        states = {_track_state(str(s)) for s in sub["TrackStatus"].dropna()}
        rc.regime[L] = (RED if "red" in states else SC if "SC" in states
                        else VSC if "VSC" in states else "green")
    return rc


def build_ghost(year: int, gp: str, driver: str, *, n_scenarios: int = 200,
                pool_years: list[int] | None = None) -> dict:
    import fastf1

    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, gp, "R")
        ses.load(telemetry=False, weather=True, messages=False)
    results = ses.results
    if results is None or len(results) == 0:
        return {"error": "no results"}
    n_laps = int(ses.laps["LapNumber"].max())
    cid = _GP_TO_CIRCUIT.get(gp, "bahrain")

    # Actual mean track temperature, so the (thermal-aware) tyre model is
    # evaluated at race conditions rather than the calibration's reference.
    race_tt = None
    try:
        w = ses.weather_data
        if w is not None and len(w) and "TrackTemp" in w.columns:
            race_tt = float(w["TrackTemp"].mean())
    except Exception:
        pass

    cal_pace = calibrate_race(year, gp, cid)
    cal_deg = calibrate_races([(y, gp) for y in pool_years], cid) if pool_years else cal_pace
    model = cal_deg.to_race_model(cid) if cal_deg else RaceModel.for_circuit(cid, n_laps=n_laps)
    if model.config.n_laps != n_laps:
        from dataclasses import replace
        model.config = replace(model.config, n_laps=n_laps)
    driver_pace = cal_pace.driver_pace if cal_pace else {}
    ref = min(driver_pace.values()) if driver_pace else 0.0

    strategies = _real_strategies(ses)
    # Real retirements: a DNF driver's cutoff is their last completed lap.
    last_lap = {c: int(g["LapNumber"].max()) for c, g in ses.laps.groupby("Driver")}

    def retire_of(abbr, status):
        st = str(status)
        return None if ("Finished" in st or "Lap" in st) else last_lap.get(abbr)

    from .data.priors import apply_to_model

    def car_model(delta, team_name):
        """Per-car calibrated model: circuit priors + this team's reliability,
        evaluated at the actual track temperature, then offset by driver pace."""
        m = apply_to_model(model, circuit_id=cid, constructor=team_name, track_temp=race_tt)
        return m.with_driver(delta)

    entries: list[CarEntry] = []
    focal_real = None
    focal_grid, focal_retire, focal_model = 10, None, None
    actual_finish = None
    team = ""
    for _, r in results.iterrows():
        abbr = r["Abbreviation"]
        if abbr not in strategies:
            continue
        delta = driver_pace.get(abbr, ref + 1.0) - ref
        grid = int(r["GridPosition"]) if r["GridPosition"] and r["GridPosition"] > 0 else 20
        rl = retire_of(abbr, r.get("Status", ""))
        team_name = r.get("TeamName", "")
        e = CarEntry(car_id=len(entries), model=car_model(delta, team_name),
                     strategy=strategies[abbr], grid=grid, name=abbr, retire_lap=rl)
        entries.append(e)
        if abbr == driver:
            focal_real, focal_grid, focal_retire = strategies[abbr], grid, rl
            focal_model = car_model(delta, team_name)
            team = team_name
            actual_finish = int(r["Position"]) if r["Position"] and not np.isnan(r["Position"]) else None
    if focal_real is None:
        return {"error": f"{driver} not classified"}

    rivals = [e for e in entries if e.name != driver]
    FID = 999
    allocation = _infer_allocation(strategies)
    # Replay the ACTUAL safety-car / VSC timeline; only lap noise varies across runs,
    # so the real-strategy sim should reproduce the real result.
    from .sim.monte_carlo import Scenario
    real_rc = _real_race_control(ses, n_laps)
    scen = ScenarioSet([Scenario(real_rc, 1000 + i) for i in range(n_scenarios)], n_laps)

    # Same starting tyre the team actually chose; the AI then makes every pit
    # call live during the race (re-deciding on safety cars).
    from .types import Stint, Strategy
    ai_start = focal_real.compounds[0]
    adaptive = Strategy([Stint(ai_start, n_laps)])  # placeholder; the policy decides pits
    policy = _ai_policy(focal_model, allocation=allocation)

    real_pos, real_trace, _, real_lt = _eval(focal_real, rivals, focal_id=FID, focal_grid=focal_grid,
                                             focal_retire=focal_retire, focal_model=focal_model, scen=scen)
    ghost_pos, ghost_trace, ghost_pits, ghost_lt = _eval(
        adaptive, rivals, focal_id=FID, focal_grid=focal_grid,
        focal_retire=focal_retire, focal_model=focal_model, scen=scen, policy=policy)

    def summary(pos, *, label, pit_laps, compounds, trace, adaptive=False):
        return {
            "strategy": label, "pit_laps": pit_laps, "compounds": compounds, "adaptive": adaptive,
            "median_finish": float(np.median(pos)), "mean_finish": round(float(np.mean(pos)), 2),
            "p_podium": round(float(np.mean(pos <= 3)), 3), "p_points": round(float(np.mean(pos <= 10)), 3),
            "trace": trace,
        }

    gp_pits = ghost_pits or []
    g_label = " → ".join([ai_start.value[0]] + [f"{cv[0]}@L{lp}" for lp, cv in gp_pits]) or ai_start.value[0]
    real = summary(real_pos, label=focal_real.label(), pit_laps=list(focal_real.pit_laps),
                   compounds=[c.value for c in focal_real.compounds], trace=real_trace)
    ghost = summary(ghost_pos, label=g_label, pit_laps=[lp for lp, _ in gp_pits],
                    compounds=[ai_start.value] + [cv for _, cv in gp_pits], trace=ghost_trace, adaptive=True)
    # Which AI stops were taken under a neutralisation (a live reaction)?
    sc_laps = {l for l in range(1, n_laps + 1) if real_rc.regime[l] != "green"}
    ghost["reacted_sc"] = [lp for lp, _ in gp_pits if lp in sc_laps]
    # Per-lap cumulative time gap (real − AI; +ve ⇒ the AI car is AHEAD on the
    # road). Built from the SAME scenario as the traces under common random
    # numbers, so it is pure strategy — this drives the on-track ghost car.
    def _cum(lt):
        if lt is None:
            return None
        a = np.nan_to_num(np.asarray(lt[1:], dtype=float), nan=0.0)
        return np.cumsum(a)
    cr, cg = _cum(real_lt), _cum(ghost_lt)
    if cr is not None and cg is not None:
        m = min(len(cr), len(cg))
        ghost["gap_s"] = [0.0] + [round(float(cr[i] - cg[i]), 2) for i in range(m)]
    else:
        ghost["gap_s"] = []
    # Tyre-allocation feasibility: the AI plan can only use sets the team had.
    from collections import Counter

    from .types import Compound as _C
    ai_seq = [ai_start] + [_C(cv) for _, cv in gp_pits]
    ai_counts = Counter(c for c in ai_seq if c.is_slick)
    ghost["sets_used"] = {c.value: n for c, n in ai_counts.items()}
    ghost["allocation"] = {c.value: allocation[c] for c in allocation}
    # Feasible = never uses more sets of a compound than the team had (the
    # credibility claim). The two-compound rule is reported separately because a
    # car that *retired* legitimately may not have completed its second compound.
    ghost["within_allocation"] = all(ai_counts[c] <= allocation[c] for c in ai_counts)
    ghost["two_compound"] = len(ai_counts) >= 2
    ghost["feasible"] = ghost["within_allocation"] and (
        ghost["two_compound"] or focal_retire is not None)
    delta = real["median_finish"] - ghost["median_finish"]   # +ve => AI finishes higher
    return {
        "meta": {"year": year, "gp": gp, "driver": driver, "team": team,
                 "n_laps": n_laps, "actual_finish": actual_finish},
        "real": real, "ghost": ghost, "delta": round(delta, 1),
        "verdict": ("AI gains" if delta > 0.5 else "AI loses" if delta < -0.5 else "line-ball"),
    }
