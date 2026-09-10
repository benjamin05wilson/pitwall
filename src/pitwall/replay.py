"""Live race-replay + live-feed engine.

The core is :class:`LiveStrategist` — a **stateful, per-driver** strategy engine
that ingests *one completed lap at a time* and returns the live call (Kalman
tyre-degradation estimate + credible band, pit-window recommendation, anomaly
flag). This is the seam a real live source plugs into: the *same* ``ingest()``
drives a historical replay, an OpenF1 live session, or a team's own telemetry —
nothing downstream changes.

* ``build_replay``        — feed a whole historical race through it at once.
* ``stream_events``       — feed it lap-by-lap at wall-clock pace (the live path);
                            the source is a genuinely-live OpenF1 session if one
                            is running, otherwise a historical race streamed in
                            real time to demonstrate the live pipeline.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .data.fastf1_loader import _enable_cache
from .live import KalmanTyreModel, advise
from .models.circuits import get_profile
from .models.params import FuelModel
from .types import Compound

_GP_TO_CIRCUIT = {
    "Bahrain": "bahrain", "Saudi Arabia": "jeddah", "Australia": "albert_park",
    "Japan": "suzuka", "China": "shanghai", "Miami": "miami", "Emilia Romagna": "imola",
    "Monaco": "monaco", "Spain": "catalunya", "Canada": "villeneuve", "Austria": "red_bull_ring",
    "Great Britain": "silverstone", "Hungary": "hungaroring", "Belgium": "spa",
    "Netherlands": "zandvoort", "Italy": "monza", "Azerbaijan": "baku", "Singapore": "marina_bay",
    "United States": "americas", "Mexico": "rodriguez", "Brazil": "interlagos",
    "Las Vegas": "vegas", "Qatar": "losail", "Abu Dhabi": "yas_marina",
}
_COMPOUND_MAP = {"SOFT": Compound.SOFT, "MEDIUM": Compound.MEDIUM, "HARD": Compound.HARD,
                 "INTERMEDIATE": Compound.INTERMEDIATE, "WET": Compound.WET}


def _track_state(status: str) -> str:
    s = str(status)
    if "5" in s:
        return "red"
    if "4" in s:
        return "SC"
    if "6" in s or "7" in s:
        return "VSC"
    if "2" in s:
        return "yellow"
    return "green"


def _isnan(x) -> bool:
    try:
        import pandas as pd
        return x is None or pd.isna(x)
    except Exception:
        return x is None


# --------------------------------------------------------------------------- #
# The stateful live engine                                                     #
# --------------------------------------------------------------------------- #
class LiveStrategist:
    """Ingest one completed lap; emit the live strategy state for that lap —
    including a continuously re-optimised plan over the whole remaining strategy
    space (``live_optimizer.reoptimize``), fed by the live Kalman belief and the
    current track status (so a Safety Car instantly reroutes it)."""

    def __init__(self, model, prior_pace: float = 95.0):
        self.model = model
        self.n_laps = model.config.n_laps
        self.pit_loss = model.pit.pit_loss_s
        self.kf = KalmanTyreModel(n_laps=self.n_laps, prior_pace=prior_pace, prior_deg=0.05)
        self._last_stint: int | None = None
        self._used: set = set()  # slick compounds used so far (FIA two-compound rule)

    def ingest(self, ev: dict) -> dict:
        """ev keys: lap, position, lap_time, compound(str), tyre_age, gap_ahead,
        gap_behind, track, is_pit, is_out, stint."""
        comp = _COMPOUND_MAP.get(str(ev.get("compound") or "").upper())
        L = int(ev["lap"])
        age = int(ev.get("tyre_age") or 0)
        lt = ev.get("lap_time")
        if comp is not None and comp.is_slick:
            self._used.add(comp)

        if self._last_stint is not None and ev.get("stint") != self._last_stint:
            self.kf.reset(prior_pace=self.kf.belief.fresh_pace,
                          prior_deg=max(0.0, self.kf.belief.deg_rate * 0.8))
        self._last_stint = ev.get("stint")

        updated = anomaly = False
        if (lt is not None and comp is not None and comp.is_slick
                and ev.get("track") == "green" and not ev.get("is_out") and not ev.get("is_pit")):
            u = self.kf.update(L, age, float(lt))
            anomaly = bool(u.anomaly)
            updated = True
        belief = self.kf.belief

        rec = None
        if comp is not None and L < self.n_laps - 1 and len(self.kf.history) >= 2:
            adv = advise(belief, current_lap=L, current_age=age, n_laps=self.n_laps, pit_loss=self.pit_loss)
            rec = {"pit_lap": adv.optimal_pit_lap, "lo": adv.window_lo, "hi": adv.window_hi}

        # Continuous re-optimisation over the whole remaining strategy space.
        plan = None
        if comp is not None and comp.is_slick and L < self.n_laps - 1:
            from .live_optimizer import reoptimize
            plan = reoptimize(
                self.model, lap=L, compound=comp, age=age,
                belief_a=belief.fresh_pace if self.kf.history else None,
                belief_b=belief.deg_rate if self.kf.history else None,
                regime=ev.get("track", "green"), used_compounds=set(self._used),
            )

        lo, hi = belief.credible_deg()
        return {
            "lap": L, "position": ev.get("position"),
            "lap_time": round(float(lt), 3) if lt is not None else None,
            "compound": comp.value if comp else None, "tyre_age": age,
            "gap_ahead": ev.get("gap_ahead"), "gap_behind": ev.get("gap_behind"),
            "track": ev.get("track", "green"), "is_pit": bool(ev.get("is_pit")),
            "deg": round(belief.deg_rate, 4) if self.kf.history else None,
            "deg_lo": round(lo, 4), "deg_hi": round(hi, 4),
            "fresh_pace": round(belief.fresh_pace, 2), "rec_pit": rec, "anomaly": anomaly,
            "plan": plan,
        }


# --------------------------------------------------------------------------- #
# Historical source (FastF1) -> raw per-lap events                            #
# --------------------------------------------------------------------------- #
@dataclass
class RaceData:
    meta: dict
    events: list[dict]
    actual_pit_laps: list[int]


def _load_session(year: int, gp: str):
    import fastf1
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, gp, "R")
        ses.load(telemetry=False, weather=False, messages=False)
    return ses


def list_drivers(year: int, gp: str) -> dict | None:
    ses = _load_session(year, gp)
    res = ses.results
    if res is None or len(res) == 0:
        return None
    drivers = [{
        "code": r["Abbreviation"], "team": r.get("TeamName", ""),
        "grid": int(r["GridPosition"]) if r["GridPosition"] else 0,
        "finish": int(r["Position"]) if r["Position"] and not np.isnan(r["Position"]) else None,
    } for _, r in res.iterrows()]
    return {"year": year, "gp": gp, "n_laps": int(ses.laps["LapNumber"].max()), "drivers": drivers}


def _historical(year: int, gp: str, driver: str) -> RaceData:
    """Pull a real race into ordered raw lap events (no engine yet)."""
    ses = _load_session(year, gp)
    laps = ses.laps.copy()
    laps["t_s"] = laps["Time"].dt.total_seconds()
    n_laps = int(laps["LapNumber"].max())
    cid = _GP_TO_CIRCUIT.get(gp, "bahrain")
    pit_loss = get_profile(cid).pit_loss_s
    foc = laps[laps["Driver"] == driver].sort_values("LapNumber")

    events, actual_pit_laps, last_stint = [], [], None
    for _, lp in foc.iterrows():
        L = int(lp["LapNumber"])
        lt = lp["LapTime"].total_seconds() if not _isnan(lp.get("LapTime")) else None
        age = int(lp["TyreLife"]) if not _isnan(lp.get("TyreLife")) else 0
        pos = int(lp["Position"]) if not _isnan(lp.get("Position")) else None
        track = _track_state(lp.get("TrackStatus", "1"))
        stint = int(lp["Stint"]) if not _isnan(lp.get("Stint")) else last_stint
        is_pit = bool(not _isnan(lp.get("PitInTime"))) or (last_stint is not None and stint != last_stint)
        if last_stint is not None and stint != last_stint:
            actual_pit_laps.append(L)
        last_stint = stint
        gap_ahead = gap_behind = None
        same = laps[laps["LapNumber"] == L].dropna(subset=["t_s", "Position"])
        if pos is not None and not same.empty:
            tf = same[same["Driver"] == driver]["t_s"]
            ah = same[same["Position"] == pos - 1]; bh = same[same["Position"] == pos + 1]
            if not tf.empty:
                if not ah.empty:
                    gap_ahead = round(float(tf.iloc[0] - ah["t_s"].iloc[0]), 1)
                if not bh.empty:
                    gap_behind = round(float(bh["t_s"].iloc[0] - tf.iloc[0]), 1)
        events.append({
            "lap": L, "position": pos, "lap_time": lt,
            "compound": str(lp["Compound"]).upper() if not _isnan(lp.get("Compound")) else None,
            "tyre_age": age, "gap_ahead": gap_ahead, "gap_behind": gap_behind,
            "track": track, "is_pit": is_pit, "is_out": bool(not _isnan(lp.get("PitOutTime"))),
            "stint": stint,
        })
    prior_pace = float(foc["LapTime"].dropna().dt.total_seconds().median() or 95.0)
    meta = {"year": year, "gp": gp, "circuit_id": cid, "driver": driver,
            "team": foc["Team"].iloc[0] if "Team" in foc and not foc.empty else "",
            "n_laps": n_laps, "pit_loss": pit_loss, "prior_pace": prior_pace}
    return RaceData(meta, events, actual_pit_laps)


# --------------------------------------------------------------------------- #
# Public: batch replay + scorecard                                             #
# --------------------------------------------------------------------------- #
def build_replay(year: int, gp: str, driver: str) -> dict:
    from .models import RaceModel
    rd = _historical(year, gp, driver)
    if not rd.events:
        return {"error": f"no laps for {driver}"}
    model = RaceModel.for_circuit(rd.meta["circuit_id"], n_laps=rd.meta["n_laps"])
    strat = LiveStrategist(model, rd.meta["prior_pace"])
    timeline = [strat.ingest(ev) for ev in rd.events]
    return {
        "meta": {k: rd.meta[k] for k in ("year", "gp", "circuit_id", "driver", "team", "n_laps", "pit_loss")},
        "actual_pit_laps": rd.actual_pit_laps,
        "timeline": timeline,
        "scorecard": _scorecard(timeline, rd.actual_pit_laps),
    }


def _scorecard(timeline: list[dict], actual_pit_laps: list[int]) -> dict:
    by_lap = {t["lap"]: t for t in timeline}
    stops, agreed = [], 0
    for P in actual_pit_laps:
        rec = None
        for back in range(3, 7):
            t = by_lap.get(P - back)
            if t and t.get("rec_pit"):
                rec = t["rec_pit"]
                break
        if rec is None:
            stops.append({"actual": P, "engine": None, "verdict": "no live read yet", "in_window": False})
            continue
        in_win = rec["lo"] <= P <= rec["hi"]
        if in_win:
            agreed += 1
            verdict = "agreed — stop was inside the engine's window"
        elif rec["pit_lap"] < P:
            verdict = f"engine would box {P - rec['pit_lap']} lap(s) earlier"
        else:
            verdict = f"engine would wait {rec['pit_lap'] - P} lap(s) longer"
        stops.append({"actual": P, "engine": rec["pit_lap"], "lo": rec["lo"],
                      "hi": rec["hi"], "verdict": verdict, "in_window": in_win})
    n = len(actual_pit_laps)
    return {"stops": stops, "agreed": agreed, "total": n,
            "headline": f"engine agreed with {agreed}/{n} of the team's stops" if n
            else "no green-flag stops to grade"}


# --------------------------------------------------------------------------- #
# Public: live stream (the real-feed path)                                     #
# --------------------------------------------------------------------------- #
def openf1_live_session() -> dict | None:
    """Detect a *currently-running* F1 race via OpenF1 (None if nothing is live
    right now). Only returns a session if wall-clock time is inside its window."""
    try:
        import requests
        from datetime import datetime, timezone
        r = requests.get("https://api.openf1.org/v1/sessions?session_key=latest", timeout=6)
        s = r.json()
        if not s or s[0].get("session_type") != "Race":
            return None
        start, end = s[0].get("date_start"), s[0].get("date_end")
        if not (start and end):
            return None
        now = datetime.now(timezone.utc)
        st = datetime.fromisoformat(start.replace("Z", "+00:00"))
        en = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return s[0] if st <= now <= en else None
    except Exception:
        return None


def stream_events(year: int, gp: str, driver: str, *, interval: float = 1.0) -> Iterator[dict]:
    """Stream the selected historical race; no current-session ingestion exists."""
    from .models import RaceModel
    source = "historical-replay"
    rd = _historical(year, gp, driver)
    model = RaceModel.for_circuit(rd.meta["circuit_id"], n_laps=rd.meta["n_laps"])
    strat = LiveStrategist(model, rd.meta["prior_pace"])
    yield {"type": "meta", "source": source,
           "meta": {k: rd.meta[k] for k in ("year", "gp", "circuit_id", "driver", "team", "n_laps", "pit_loss")},
           "actual_pit_laps": rd.actual_pit_laps}
    states = []
    for ev in rd.events:
        state = strat.ingest(ev)
        states.append(state)
        yield {"type": "lap", "state": state}
        time.sleep(interval)
    yield {"type": "end", "scorecard": _scorecard(states, rd.actual_pit_laps)}
