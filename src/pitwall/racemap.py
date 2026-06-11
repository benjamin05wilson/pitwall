"""Full-race track animation + events.

Turns a real race into something you can *watch*: the circuit shape, every car's
position on track over time (resampled onto a common clock), and the race-control
event stream (flags, safety cars, penalties, incidents, retirements). The
frontend plays this back on a track map; selecting a car layers the live strategy
engine (``replay.LiveStrategist``) on top.

Position data is downsampled onto a fixed time grid so a full race is a single,
modestly-sized payload; retired cars simply stop having positions (and a
retirement event fires), so a crash/DNF shows as a car dropping off the map.
"""

from __future__ import annotations

import warnings

import numpy as np

from .data.fastf1_loader import _enable_cache

# Race-control categories worth surfacing as headline events.
_KEEP_FLAGS = {"RED", "YELLOW", "DOUBLE YELLOW", "GREEN", "CHEQUERED"}


def _load(year: int, gp: str):
    import fastf1
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = fastf1.get_session(year, gp, "R")
        s.load(telemetry=True, weather=False, messages=True)
    return s


def track_outline(year: int, gp: str) -> dict:
    """Circuit outline (X,Y) from a representative fast lap, plus bounds."""
    s = _load(year, gp)
    pos = s.laps.pick_fastest().get_pos_data()
    x = pos["X"].to_numpy(dtype=float)
    y = pos["Y"].to_numpy(dtype=float)
    return {"x": x.tolist(), "y": y.tolist(),
            "bounds": [float(x.min()), float(x.max()), float(y.min()), float(y.max())]}


def _driver_meta(s) -> dict[str, dict]:
    out = {}
    for _, r in s.results.iterrows():
        out[str(r["DriverNumber"])] = {
            "code": r["Abbreviation"], "team": r.get("TeamName", ""),
            "color": "#" + str(r.get("TeamColor") or "888888").lstrip("#"),
            "finish": int(r["Position"]) if r["Position"] and not np.isnan(r["Position"]) else None,
            "status": r.get("Status", ""),
        }
    return out


def race_frames(year: int, gp: str, *, step: float = 1.5) -> dict:
    """Resample every car's on-track position onto a common time grid (one frame
    every ``step`` seconds of race time)."""
    s = _load(year, gp)
    out = track_outline(year, gp)  # reuse session via cache
    meta = _driver_meta(s)

    laps = s.laps
    t0 = float(laps[laps["LapNumber"] == 1]["LapStartTime"].dropna().dt.total_seconds().min())
    t1 = float(laps["Time"].dropna().dt.total_seconds().max())
    grid = np.arange(t0, t1, step)

    # When each driver last completed a lap (their cutoff — a retiree drops off).
    last_lap_t = {code: float(g["Time"].dropna().dt.total_seconds().max())
                  for code, g in laps.groupby("Driver") if g["Time"].notna().any()}

    # Per-driver position timeline (lap, race-clock time, classified position) —
    # drives the live leaderboard and syncs each car's strategy to the clock.
    pos_tl: dict[str, list] = {}
    for code, g in laps.groupby("Driver"):
        g = g.sort_values("LapNumber")
        tl = []
        for _, lp in g.iterrows():
            t = lp.get("Time")
            if _isna(t):
                continue
            tl.append({"lap": int(lp["LapNumber"]),
                       "t": round(float(t.total_seconds()) - t0, 1),
                       "pos": int(lp["Position"]) if not _isna(lp.get("Position")) else None})
        pos_tl[code] = tl

    cars = []
    for num, info in meta.items():
        pd_ = s.pos_data.get(num)
        if pd_ is None or len(pd_) == 0:
            continue
        ts = pd_["SessionTime"].dt.total_seconds().to_numpy()
        xs = pd_["X"].to_numpy(dtype=float)
        ys = pd_["Y"].to_numpy(dtype=float)
        keep = ts.argsort()
        ts, xs, ys = ts[keep], xs[keep], ys[keep]
        gx = np.interp(grid, ts, xs)
        gy = np.interp(grid, ts, ys)
        # Valid from first sample to the driver's last completed lap (retirees
        # vanish; finishers run to the flag).
        cutoff = min(ts[-1], last_lap_t.get(info["code"], ts[-1]) + step)
        valid = (grid >= ts[0]) & (grid <= cutoff)
        X = [int(v) if ok else None for v, ok in zip(gx, valid)]
        Y = [int(v) if ok else None for v, ok in zip(gy, valid)]
        retired = bool(info["finish"] is None or
                       ("Finished" not in str(info["status"]) and "Lap" not in str(info["status"])))
        cars.append({"num": num, "code": info["code"], "team": info["team"],
                     "color": info["color"], "x": X, "y": Y,
                     "finish": info["finish"], "status": info["status"],
                     "retired": retired, "retire_frame": int(valid.argmin()) if retired and (~valid).any() else None,
                     "pos_timeline": pos_tl.get(info["code"], [])})

    return {
        "track": {"x": out["x"], "y": out["y"], "bounds": out["bounds"]},
        "t0": t0, "step": step, "n_frames": len(grid),
        "n_laps": int(laps["LapNumber"].max()),
        "gp": gp, "year": year, "cars": cars,
        "events": race_events(year, gp, _session=s),
    }


def race_events(year: int, gp: str, *, _session=None) -> list[dict]:
    """Headline race-control events + retirements, on the race clock (seconds)."""
    s = _session or _load(year, gp)
    ev = []
    rcm = s.race_control_messages
    laps = s.laps
    t0 = float(laps[laps["LapNumber"] == 1]["LapStartTime"].dropna().dt.total_seconds().min())
    # Map lap number -> race-clock time the leader completed it (for event timing).
    lap_t = (laps.groupby("LapNumber")["Time"].min().dropna().dt.total_seconds() - t0).to_dict()

    def at(lap):
        return float(lap_t.get(int(lap), max(lap_t.values()) if lap_t else 0)) if not _isna(lap) else None

    if rcm is not None and len(rcm):
        for _, m in rcm.iterrows():
            cat, flag, msg = str(m.get("Category")), m.get("Flag"), str(m.get("Message", ""))
            up = msg.upper()
            kind = None
            if cat == "SafetyCar" or "SAFETY CAR" in up or "VIRTUAL SAFETY" in up:
                kind = "sc"
            elif cat == "Flag" and flag in {"RED", "CHEQUERED"}:
                kind = "flag"
            elif "PENALTY" in up:
                kind = "penalty"
            elif "INVESTIGAT" in up or "NOTED" in up:
                kind = "incident"
            if not kind:
                continue
            ev.append({"t": at(m.get("Lap")), "lap": int(m["Lap"]) if not _isna(m.get("Lap")) else None,
                       "kind": kind, "flag": flag, "message": msg[:90]})
    # Retirements from results (timed at the driver's last lap).
    last_lap = {code: int(g["LapNumber"].max()) for code, g in laps.groupby("Driver")}
    for _, r in s.results.iterrows():
        st = str(r.get("Status", ""))
        if st and "Finished" not in st and "Lap" not in st:
            L = last_lap.get(r["Abbreviation"])
            ev.append({"t": at(L), "lap": L, "kind": "retire", "flag": None,
                       "message": f"{r['Abbreviation']} OUT — {st}"})
    ev.sort(key=lambda e: (e["t"] is None, e["t"] or 0))
    return ev


def _isna(x) -> bool:
    try:
        import pandas as pd
        return pd.isna(x)
    except Exception:
        return x is None
