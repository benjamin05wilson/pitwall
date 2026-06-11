"""FastF1 calibration backend.

FastF1 wraps the official F1 live-timing feed and exposes the richest public
data: per-lap ``TrackStatus`` (so Safety-Car / yellow laps can be filtered
*exactly* rather than by a pace heuristic), ``IsAccurate``, true ``TyreLife``,
sector times, trap speeds, and weather. When reachable it is the preferred
backend over OpenF1; ``pitwall.data.calibrate`` falls back to OpenF1 if FastF1
is blocked (the official API is IP-restricted in some environments).

``load_clean_laps`` returns the same schema as ``calibrate.build_lap_dataset``
— columns ``[driver, lap, compound, tyre_age, lap_time]`` (+ per-lap
``track_temp`` / ``air_temp``, ``speed_trap``) — so the calibration code is
backend-agnostic. ``load_pit_durations`` reconstructs real pit-lane losses
(``PitOutTime - PitInTime``) segmented by track status, and ``load_quali_ref``
pulls the qualifying pace anchor — both feeding the calibrated priors store.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import Compound

_COMPOUND_MAP = {
    "SOFT": Compound.SOFT, "MEDIUM": Compound.MEDIUM, "HARD": Compound.HARD,
    "INTERMEDIATE": Compound.INTERMEDIATE, "WET": Compound.WET,
}
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "fastf1_cache"


def _enable_cache() -> None:
    import fastf1
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(DEFAULT_CACHE))


def fastf1_available() -> bool:
    try:
        import fastf1  # noqa: F401
        return True
    except Exception:
        return False


def _per_lap_weather(ses) -> pd.DataFrame | None:
    """Weather aligned to each lap by session time (nearest sample). FastF1's
    weather stream is ~1/min; merge_asof attaches the contemporaneous TrackTemp /
    AirTemp / Rainfall to every lap so degradation can be made temperature-aware
    rather than collapsed to a single session mean."""
    try:
        w = ses.weather_data
    except Exception:
        return None
    if w is None or len(w) == 0 or "Time" not in w.columns:
        return None
    cols = [c for c in ("Time", "TrackTemp", "AirTemp", "Rainfall", "Humidity",
                        "WindSpeed") if c in w.columns]
    return w[cols].sort_values("Time").reset_index(drop=True)


def load_clean_laps(year: int, gp: str, session: str = "R") -> pd.DataFrame:
    """Load and clean a session's green-flag racing laps for calibration.

    Filtering: keep ``IsAccurate`` laps on pure-green ``TrackStatus == '1'``,
    drop pit in/out laps and unknown compounds, then a robust per-(driver,
    compound) 1.07x median guard. ``TyreLife`` is the true tyre age. Each lap
    carries its *contemporaneous* ``track_temp`` / ``air_temp`` (merged by time),
    so the fit can separate thermal drift from mechanical wear."""
    import fastf1
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, gp, session)
        ses.load(telemetry=False, weather=True, messages=False)
    laps = ses.laps
    if laps is None or len(laps) == 0:
        return pd.DataFrame()

    df = laps.copy()
    # Pure-green, accurate, slick-compound, non-pit laps only.
    df = df[df["IsAccurate"] == True]  # noqa: E712 - pandas mask
    df = df[df["TrackStatus"].astype(str) == "1"]
    df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[df["LapTime"].notna()]
    if df.empty:
        return pd.DataFrame()

    # Per-lap weather (nearest sample by session Time) with a session-mean fallback.
    wx = _per_lap_weather(ses)
    sess_track = float(wx["TrackTemp"].mean()) if wx is not None and "TrackTemp" in wx else float("nan")
    sess_air = float(wx["AirTemp"].mean()) if wx is not None and "AirTemp" in wx else float("nan")
    lap_tt: dict[int, float] = {}
    lap_at: dict[int, float] = {}
    if wx is not None and "Time" in df.columns:
        try:
            m = pd.merge_asof(
                df[["LapNumber", "Time"]].dropna(subset=["Time"]).sort_values("Time"),
                wx, on="Time", direction="nearest",
            )
            for _, r in m.iterrows():
                ln = int(r["LapNumber"])
                lap_tt[ln] = float(r["TrackTemp"]) if "TrackTemp" in m and pd.notna(r.get("TrackTemp")) else sess_track
                lap_at[ln] = float(r["AirTemp"]) if "AirTemp" in m and pd.notna(r.get("AirTemp")) else sess_air
        except Exception:
            pass

    rows = []
    for _, lp in df.iterrows():
        comp = _COMPOUND_MAP.get(str(lp["Compound"]).upper())
        if comp is None:
            continue
        ln = int(lp["LapNumber"])
        rows.append({
            "driver": lp["Driver"],
            "lap": ln,
            "compound": comp,
            "tyre_age": int(lp["TyreLife"]) if pd.notna(lp["TyreLife"]) else 0,
            "lap_time": lp["LapTime"].total_seconds(),
            "track_temp": lap_tt.get(ln, sess_track),
            "air_temp": lap_at.get(ln, sess_air),
            "speed_trap": float(lp["SpeedST"]) if pd.notna(lp.get("SpeedST")) else float("nan"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    keep = []
    for _, grp in out.groupby(["driver", "compound"]):
        med = grp["lap_time"].median()
        keep.append(grp[grp["lap_time"] <= 1.07 * med])
    return pd.concat(keep).reset_index(drop=True)


def _gap_to_car_ahead(laps_all: pd.DataFrame) -> dict:
    """Per-(lap, driver) on-track gap to the car ahead (s), from the order at the
    line each lap. ``Time`` is the session time a car completes the lap, so within
    a lap the gap to the car one position ahead is the difference of their Times.
    Leaders (and any car with no valid car ahead) get +inf (clean air)."""
    gap: dict = {}
    if "Position" not in laps_all.columns or "Time" not in laps_all.columns:
        return gap
    for L, grp in laps_all.groupby("LapNumber"):
        g = grp.dropna(subset=["Position", "Time"]).sort_values("Position")
        if g.empty:
            continue
        times = g["Time"].to_numpy()
        drivers = g["Driver"].to_numpy()
        for i in range(len(g)):
            if i == 0:
                ga = np.inf
            else:
                dt = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
                ga = float(dt) if dt == dt and dt >= 0 else np.inf
            gap[(int(L), str(drivers[i]))] = ga
    return gap


def load_laps_with_traffic(year: int, gp: str) -> pd.DataFrame:
    """Clean green racing laps annotated with the gap to the car ahead and whether
    the car is *held station* (following at a roughly constant gap) vs *closing*
    (catching fast). Isolating held-station laps is what lets the dirty-air
    calibration measure the pace a car loses while *stuck* in a wake, rather than
    confounding it with a faster car reeling in a slower one.

    Columns: clean-lap schema (driver, lap, compound, tyre_age, lap_time,
    track_temp, ...) plus ``gap_ahead`` and ``held`` (bool)."""
    import fastf1
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, gp, "R")
        ses.load(telemetry=False, weather=True, messages=False)
    laps_all = ses.laps
    if laps_all is None or len(laps_all) == 0:
        return pd.DataFrame()
    gap = _gap_to_car_ahead(laps_all)

    base = load_clean_laps(year, gp)  # reuses the green/accurate/slick filtering
    if base.empty:
        return base
    g_now = base.apply(lambda r: gap.get((int(r["lap"]), str(r["driver"])), np.inf), axis=1)
    g_prev = base.apply(lambda r: gap.get((int(r["lap"]) - 1, str(r["driver"])), np.inf), axis=1)
    base = base.copy()
    base["gap_ahead"] = g_now.to_numpy()
    # Held station: within the dirty-air window and not reeling the car ahead in
    # quickly (gap not shrinking by more than 0.4 s/lap). Catching laps are excluded
    # from the proximity term so a fast car's pace doesn't mask the wake penalty.
    delta = g_now.to_numpy() - g_prev.to_numpy()
    base["held"] = (base["gap_ahead"].to_numpy() < 3.0) & (delta > -0.4)
    return base


_WET_OPT = {"INTERMEDIATE": 0.45, "WET": 0.85}  # field-consensus wetness anchors


def load_wet_laps(year: int, gp: str) -> pd.DataFrame:
    """Representative wet-tyre racing laps (INTERMEDIATE / WET), accurate and
    non-pit, for calibrating wet pace. Same schema as ``load_clean_laps`` minus
    the slick filter. Empty for a dry race."""
    import fastf1
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, gp, "R")
        ses.load(telemetry=False, weather=True, messages=False)
    laps = ses.laps
    if laps is None or len(laps) == 0:
        return pd.DataFrame()
    df = laps.copy()
    df = df[df["IsAccurate"] == True]  # noqa: E712
    df = df[df["TrackStatus"].astype(str) == "1"]
    df = df[df["Compound"].isin(["INTERMEDIATE", "WET"])]
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[df["LapTime"].notna()]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, lp in df.iterrows():
        comp = _COMPOUND_MAP.get(str(lp["Compound"]).upper())
        if comp is None:
            continue
        rows.append({
            "driver": lp["Driver"], "lap": int(lp["LapNumber"]), "compound": comp,
            "tyre_age": int(lp["TyreLife"]) if pd.notna(lp["TyreLife"]) else 0,
            "lap_time": lp["LapTime"].total_seconds(),
        })
    return pd.DataFrame(rows)


def wetness_timeline(year: int, gp: str) -> list[float]:
    """Reconstruct a per-lap track-wetness proxy in [0, 1] from the field's
    collective tyre choice — the best public observable of how wet the track is
    (teams fit inters/wets exactly when conditions call for them). Slicks count as
    dry; intermediates pull wetness toward ~0.45, full wets toward ~0.85. Blended
    upward when the weather feed reports rain. Index 0 unused; length n_laps+1."""
    import fastf1
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, gp, "R")
        ses.load(telemetry=False, weather=True, messages=False)
    laps = ses.laps
    if laps is None or len(laps) == 0:
        return []
    n_laps = int(laps["LapNumber"].max())
    rain_by_time = _per_lap_weather(ses)
    out = [0.0] * (n_laps + 1)
    for L in range(1, n_laps + 1):
        sub = laps[laps["LapNumber"] == L]
        comps = sub["Compound"].dropna().astype(str)
        if comps.empty:
            out[L] = out[L - 1] if L else 0.0
            continue
        n = len(comps)
        w = sum(_WET_OPT.get(c, 0.0) for c in comps) / n
        out[L] = round(float(w), 3)
    # Light smoothing so single-lap tyre noise doesn't create spikes.
    sm = out[:]
    for L in range(2, n_laps):
        sm[L] = round((out[L - 1] + 2 * out[L] + out[L + 1]) / 4.0, 3)
    return sm


def load_pit_durations(year: int, gp: str) -> pd.DataFrame:
    """Reconstruct real pit-lane losses from ``PitOutTime - PitInTime``.

    A car's in-lap carries ``PitInTime``; the following out-lap carries
    ``PitOutTime``. The difference is the full pit-lane traverse + stationary
    time (≈ the strategic "pit loss"). We tag each stop with the track status at
    the stop so green vs SC/VSC losses can be separated (the cheap-stop discount).
    Returns columns ``[driver, team, lap, pit_loss, regime]``."""
    import fastf1
    _enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, gp, "R")
        ses.load(telemetry=False, weather=False, messages=False)
    laps = ses.laps
    if laps is None or len(laps) == 0:
        return pd.DataFrame()
    rows = []
    for drv, dl in laps.groupby("Driver"):
        dl = dl.sort_values("LapNumber")
        team = str(dl["Team"].iloc[0]) if "Team" in dl.columns and len(dl) else ""
        in_time, in_lap = None, None
        for _, lp in dl.iterrows():
            if pd.notna(lp.get("PitInTime")):
                in_time = lp["PitInTime"]
                in_lap = int(lp["LapNumber"])
            if pd.notna(lp.get("PitOutTime")) and in_time is not None:
                loss = (lp["PitOutTime"] - in_time).total_seconds()
                regime = _status_regime(str(in_lap and dl[dl["LapNumber"] == in_lap]["TrackStatus"].iloc[0]))
                if 12.0 < loss < 60.0:  # guard against red-flag / data artefacts
                    rows.append({"driver": drv, "team": team, "lap": in_lap,
                                 "pit_loss": loss, "regime": regime})
                in_time, in_lap = None, None
    return pd.DataFrame(rows)


def _status_regime(status: str) -> str:
    """Collapse a FastF1 TrackStatus code string to green / SC / VSC."""
    s = set(str(status))
    if "6" in s or "7" in s:
        return "VSC"
    if "4" in s:
        return "SC"
    return "green"


def load_quali_ref(year: int, gp: str) -> dict[str, float]:
    """Best qualifying lap (s) per driver acronym — a race-condition-independent
    pace anchor used to calibrate the race-pace gap and seed early-race calls
    before enough green laps accumulate for the Kalman filter to converge."""
    import fastf1
    _enable_cache()
    out: dict[str, float] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = fastf1.get_session(year, gp, "Q")
            q.load(telemetry=False, weather=False, messages=False)
        res = q.results
        if res is None or len(res) == 0:
            return out
        for _, r in res.iterrows():
            best = np.nan
            for col in ("Q3", "Q2", "Q1"):
                v = r.get(col)
                if pd.notna(v) and getattr(v, "total_seconds", None):
                    s = v.total_seconds()
                    best = s if np.isnan(best) else min(best, s)
            if not np.isnan(best):
                out[str(r["Abbreviation"])] = float(best)
    except Exception:
        pass
    return out
