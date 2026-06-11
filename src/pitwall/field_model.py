"""Multi-agent field model + best-response strategy.

Instead of optimising the chosen car in isolation, this runs the live engine for
**every driver** — a Kalman tyre-degradation estimate and a predicted pit lap for
each — then optimises the chosen car as a *best response* to the predicted field:
undercut the car ahead before it stops, cover the threat from behind, exploit a
rival whose tyres are dying.

Pipeline:
  build_field_state(year, gp)   -> per-driver, per-lap {pace a, deg b, age, pos,
                                   gap basis t_s, PREDICTED next pit lap}
  build_field_strategy(...)     -> the chosen car's per-lap timeline enriched with
                                   undercut (car ahead) / overcut-threat (car behind)
"""

from __future__ import annotations

import warnings

from .live import KalmanTyreModel
from .live_optimizer import SLICKS, reoptimize
from .models import RaceModel
from .replay import (
    _COMPOUND_MAP, _GP_TO_CIRCUIT, _isnan, _load_session, _track_state, build_replay,
)
from .types import Compound


def build_field_state(year: int, gp: str) -> dict:
    """Run the live tyre estimator + pit predictor for every driver."""
    s = _load_session(year, gp)
    laps = s.laps.copy()
    laps["t_s"] = laps["Time"].dt.total_seconds()
    n = int(laps["LapNumber"].max())
    cid = _GP_TO_CIRCUIT.get(gp, "bahrain")
    model = RaceModel.for_circuit(cid, n_laps=n)

    drivers: dict[str, dict] = {}
    by_lap_pos: dict[int, dict] = {}
    for code, g in laps.groupby("Driver"):
        g = g.sort_values("LapNumber")
        prior = float(g["LapTime"].dropna().dt.total_seconds().median() or 95.0)
        kf = KalmanTyreModel(n_laps=n, prior_pace=prior, prior_deg=0.05)
        used: set = set()
        last_stint = None
        recs: dict[int, dict] = {}
        for _, lp in g.iterrows():
            L = int(lp["LapNumber"])
            comp = _COMPOUND_MAP.get(str(lp["Compound"]).upper())
            age = int(lp["TyreLife"]) if not _isnan(lp.get("TyreLife")) else 0
            pos = int(lp["Position"]) if not _isnan(lp.get("Position")) else None
            ts = float(lp["t_s"]) if not _isnan(lp.get("t_s")) else None
            track = _track_state(lp.get("TrackStatus", "1"))
            stint = int(lp["Stint"]) if not _isnan(lp.get("Stint")) else last_stint
            lt = lp["LapTime"].total_seconds() if not _isnan(lp.get("LapTime")) else None
            if comp is not None and comp.is_slick:
                used.add(comp)
            if last_stint is not None and stint != last_stint:
                kf.reset(prior_pace=kf.belief.fresh_pace, prior_deg=max(0.0, kf.belief.deg_rate * 0.8))
            last_stint = stint
            if (lt and comp and comp.is_slick and track == "green"
                    and _isnan(lp.get("PitInTime")) and _isnan(lp.get("PitOutTime"))):
                kf.update(L, age, float(lt))
            b = kf.belief
            pred = None
            if comp is not None and comp.is_slick and L < n - 1 and len(kf.history) >= 2:
                r = reoptimize(model, lap=L, compound=comp, age=age,
                               belief_a=b.fresh_pace, belief_b=b.deg_rate,
                               regime="green", used_compounds=set(used), max_more_stops=1)
                pred = r["first_pit"]
            recs[L] = {"t_s": ts, "pos": pos, "compound": comp.value if comp else None,
                       "age": age, "a": round(b.fresh_pace, 2), "b": round(b.deg_rate, 4),
                       "pred_pit": pred}
            if pos is not None:
                by_lap_pos.setdefault(L, {})[pos] = code
        drivers[code] = recs
    return {"drivers": drivers, "by_lap_pos": by_lap_pos, "n_laps": n, "circuit_id": cid}


def _k0(model):
    return {c: model.tyres.compounds[c].k0 for c in SLICKS}


def _battle(model, *, lap, n, gap, focal_a, focal_compound, fresh_compound,
            rival, attacking: bool):
    """Quantify an undercut (attacking the car ahead) or the overcut threat
    (the car behind attacking you). ``gap`` is the time gap between the cars (s)."""
    if rival is None or gap is None:
        return None
    k0 = _k0(model)
    fc = _COMPOUND_MAP.get((fresh_compound or "HARD").upper(), Compound.HARD)
    cc = _COMPOUND_MAP.get((focal_compound or "MEDIUM").upper(), Compound.MEDIUM)
    # An undercut/overcut only exists when we PREDICT the rival is about to pit
    # (their tyres are talking). No prediction → no window → no battle. The
    # advantage is credited over a short reaction window (HORIZON laps).
    HORIZON = 4
    if rival["pred_pit"] is None:
        return None
    pred = int(rival["pred_pit"])
    laps_out = min(max(0, pred - lap), HORIZON)
    rival_pace = rival["a"] + rival["b"] * rival["age"]  # their current (worn) lap pace
    if attacking:                                    # you pit now onto a fresh set
        my_fresh = focal_a + (k0[fc] - k0[cc])
        per_lap = max(0.15, rival_pace - my_fresh)   # seconds gained per lap vs them
        margin = per_lap * laps_out - gap            # >0 => you jump them
    else:                                            # the car behind pits to undercut you
        rc = _COMPOUND_MAP.get((rival["compound"] or "MEDIUM").upper(), Compound.MEDIUM)
        their_fresh = rival["a"] + (k0[fc] - k0[rc])
        my_pace = focal_a + 0.0                       # you stay out, degrading
        per_lap = max(0.15, my_pace - their_fresh)
        margin = per_lap * laps_out - gap            # >0 => they jump you if they box now
    return {
        "code": None,                                 # filled by caller
        "gap": round(gap, 1), "pred_pit": int(pred) if pred else None,
        "per_lap": round(per_lap, 2), "margin": round(margin, 1),
        "viable": bool(margin > 1.0 and laps_out >= 2),  # a meaningful position jump
        "tyre": rival["compound"], "age": rival["age"],
    }


def build_field_strategy(year: int, gp: str, driver: str, field_state: dict | None = None) -> dict:
    """The chosen car's live timeline, enriched each lap with undercut/overcut
    analysis vs the *predicted* field."""
    fs = field_state or build_field_state(year, gp)
    model = RaceModel.for_circuit(fs["circuit_id"], n_laps=fs["n_laps"])
    n = fs["n_laps"]
    rep = build_replay(year, gp, driver)
    if "error" in rep:
        return rep
    drivers, by_lap_pos = fs["drivers"], fs["by_lap_pos"]

    for t in rep["timeline"]:
        L = t["lap"]
        t["undercut"] = t["overcut"] = None
        foc = drivers.get(driver, {}).get(L)
        if not foc or foc["pos"] is None:
            continue
        pos = foc["pos"]
        fresh_comp = t["plan"]["first_comp"] if t.get("plan") and t["plan"].get("first_comp") else "HARD"
        lp = by_lap_pos.get(L, {})
        # Car ahead → undercut opportunity.
        ah_code = lp.get(pos - 1)
        ah = drivers.get(ah_code, {}).get(L) if ah_code else None
        if ah and ah["t_s"] and foc["t_s"]:
            res = _battle(model, lap=L, n=n, gap=foc["t_s"] - ah["t_s"], focal_a=foc["a"],
                          focal_compound=t["compound"], fresh_compound=fresh_comp,
                          rival=ah, attacking=True)
            if res:
                res["code"] = ah_code
                res["pos"] = pos - 1
                t["undercut"] = res
        # Car behind → overcut/undercut threat.
        bh_code = lp.get(pos + 1)
        bh = drivers.get(bh_code, {}).get(L) if bh_code else None
        if bh and bh["t_s"] and foc["t_s"]:
            res = _battle(model, lap=L, n=n, gap=bh["t_s"] - foc["t_s"], focal_a=foc["a"],
                          focal_compound=t["compound"], fresh_compound=bh["compound"],
                          rival=bh, attacking=False)
            if res:
                res["code"] = bh_code
                res["pos"] = pos + 1
                t["overcut"] = res
    return rep
