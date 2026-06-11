"""Calibrate the tyre/pace model on real race data.

This is the engine's evidence base. We pull compound-labelled stints and real lap
durations (OpenF1), fuel-correct them, then fit the lap-time model

    corrected_lap(d, c, age) = base_pace[d] + k0[c] + k1[c]*age + k2[c]*age^2

as a single linear regression with **per-driver fixed effects**. The driver
intercepts soak up car/driver pace so the degradation slopes ``k1``/``k2`` and
compound offsets ``k0`` are not confounded by *which* cars happened to run *which*
compound — the classic trap when pooling tyre data across a field.

Outputs calibrated :class:`CompoundParams`, real per-driver pace deltas (used to
build a realistic simulated field), and an honest per-compound RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.params import CompoundParams, FuelModel
from ..types import Compound
from .openf1 import OpenF1Client

_COMPOUND_MAP = {
    "SOFT": Compound.SOFT, "MEDIUM": Compound.MEDIUM, "HARD": Compound.HARD,
    "INTERMEDIATE": Compound.INTERMEDIATE, "WET": Compound.WET,
}


@dataclass
class CalibrationResult:
    circuit_id: str
    year: int
    n_laps: int
    compounds: dict[Compound, CompoundParams]
    driver_pace: dict[str, float]    # acronym -> fuel-corrected fresh MEDIUM pace (s)
    reference_pace: float            # fastest driver's base pace (s)
    rmse: float                      # overall fit RMSE (s)
    rmse_by_compound: dict[Compound, float]
    n_obs: int
    # Thermal coupling of degradation: extra s/lap of slope per °C above temp_ref.
    # k1_effective(c) = k1(c) + temp_coeff * (track_temp - temp_ref).
    temp_coeff: float = 0.0
    temp_ref: float = 30.0           # mean TrackTemp of the fit (°C)

    def driver_deltas(self) -> dict[str, float]:
        """Per-driver pace delta vs the fastest car (s/lap, >=0)."""
        return {d: p - self.reference_pace for d, p in self.driver_pace.items()}

    def to_race_model(self, circuit_id: str, *, use_pace: bool = True):
        """Build a calibrated :class:`RaceModel`: per-circuit priors with the
        fitted compound degradation (and optionally the fitted base pace) swapped
        in. Missing compounds (e.g. wets) fall back to defaults."""
        from ..models import RaceModel
        from ..models.params import DEFAULT_COMPOUNDS, PaceModel, TyreModel
        from dataclasses import replace as _replace

        model = RaceModel.for_circuit(circuit_id, n_laps=self.n_laps)
        merged = dict(DEFAULT_COMPOUNDS)
        merged.update(self.compounds)
        # Carry the fitted thermal coupling; evaluate at the fit's mean temp by
        # default so a calibrated model reproduces its own conditions, and let
        # the live layer override ``track_temp`` from the weather feed.
        model.tyres = _replace(
            model.tyres, compounds=merged,
            temp_coeff=self.temp_coeff, temp_ref=self.temp_ref, track_temp=self.temp_ref,
        )
        if use_pace:
            # Fitted reference is a fuel-corrected fresh-medium pace (~race base).
            model.pace = _replace(
                model.pace,
                t_quali_ref=self.reference_pace - model.pace.race_pace_gap,
            )
        # Fold in the calibrated per-circuit priors (real pit loss + variance, SC
        # probability, race-pace gap, dirty-air) without clobbering the fitted
        # tyre coupling. Degrades to a no-op if the priors store isn't built.
        from .priors import apply_to_model
        return apply_to_model(model, circuit_id=circuit_id)

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.summary(), indent=2))

    def summary(self) -> dict:
        return {
            "circuit": self.circuit_id, "year": self.year, "n_obs": self.n_obs,
            "rmse_s": round(self.rmse, 3),
            "ref_pace_s": round(self.reference_pace, 3),
            "compounds": {
                c.value: {"k0": round(p.k0, 3), "k1": round(p.k1, 4), "k2": round(p.k2, 5)}
                for c, p in self.compounds.items()
            },
        }


def build_lap_dataset(client: OpenF1Client, session_key: int) -> pd.DataFrame:
    """Join laps + stints into clean per-lap observations with compound + true
    tyre age, dropping out/in laps and slow (SC/traffic/outlier) laps."""
    laps = client.laps(session_key)
    stints = client.stints(session_key)
    if laps.empty or stints.empty:
        return pd.DataFrame()

    rows = []
    for _, st in stints.iterrows():
        dn = st["driver_number"]
        lap0, lap1 = int(st["lap_start"]), int(st["lap_end"])
        comp = _COMPOUND_MAP.get(str(st["compound"]).upper())
        if comp is None or not comp.is_slick:
            continue
        age0 = int(st.get("tyre_age_at_start") or 0)
        seg = laps[(laps["driver_number"] == dn) &
                   (laps["lap_number"] >= lap0) & (laps["lap_number"] <= lap1)].copy()
        for _, lp in seg.iterrows():
            ln = int(lp["lap_number"])
            dur = lp.get("lap_duration")
            if dur is None or pd.isna(dur):
                continue
            is_out = bool(lp.get("is_pit_out_lap"))
            is_in = ln == lap1 and lap1 < int(laps["lap_number"].max())
            rows.append({
                "driver": dn, "lap": ln, "compound": comp,
                "tyre_age": age0 + (ln - lap0), "lap_time": float(dur),
                "is_edge": is_out or is_in or ln == lap0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Drop out/in/first-of-stint laps, then robust per-(driver,compound) quantile.
    df = df[~df["is_edge"]].copy()
    keep = []
    for (_, _), grp in df.groupby(["driver", "compound"]):
        med = grp["lap_time"].median()
        keep.append(grp[grp["lap_time"] <= 1.07 * med])
    return pd.concat(keep).reset_index(drop=True) if keep else df


def _huber_weights(resid: np.ndarray, k: float = 1.345) -> np.ndarray:
    """Huber weights from a robust (MAD) scale estimate."""
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    z = np.abs(resid) / scale
    return np.where(z <= k, 1.0, k / z)


def fit_compounds(
    df: pd.DataFrame, *, n_laps: int, fuel: FuelModel | None = None,
    quadratic: bool = False, track_evolution: bool = True, robust: bool = True,
) -> CalibrationResult | None:
    """Fit base-pace (per driver) + per-compound offset & degradation via a single
    fixed-effects regression on fuel-corrected lap times.

    The model includes a **global track-evolution term** (linear in lap-of-race)
    so degradation is not confounded with the circuit rubbering in. Degradation
    is **linear by default** (robust with sparse per-compound age ranges); pass
    ``quadratic=True`` only with rich data. ``robust`` runs a few IRLS-Huber
    iterations to down-weight traffic/SC-tainted laps that survive filtering."""
    if df.empty or len(df) < 20:
        return None
    fuel = fuel or FuelModel()
    df = df.copy()
    df["fuel_corr"] = df["lap_time"] - df["lap"].map(lambda l: fuel.penalty(l, n_laps))

    drivers = sorted(df["driver"].unique())
    comps_present = [c for c in (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
                     if c in set(df["compound"])]
    base_c = Compound.MEDIUM if Compound.MEDIUM in comps_present else \
        df["compound"].value_counts().idxmax()

    d_index = {d: i for i, d in enumerate(drivers)}
    nD = len(drivers)
    off_comps = [c for c in comps_present if c != base_c]
    o_index = {c: i for i, c in enumerate(off_comps)}
    c_index = {c: i for i, c in enumerate(comps_present)}
    nO, nC = len(off_comps), len(comps_present)
    deg_order = 2 if quadratic else 1

    # Columns: [driver dummies | compound offsets | per-compound age (^deg) | track-evo]
    ncols = nD + nO + deg_order * nC + (1 if track_evolution else 0)
    evo_col = nD + nO + deg_order * nC
    lap_mid = float(df["lap"].median())

    comp_arr = df["compound"].to_numpy(dtype=object)
    age = df["tyre_age"].to_numpy(dtype=float)
    lap = df["lap"].to_numpy(dtype=float)
    X = np.zeros((len(df), ncols))
    y = df["fuel_corr"].to_numpy()
    for r in range(len(df)):
        X[r, d_index[df["driver"].iat[r]]] = 1.0
        c = comp_arr[r]
        if c in o_index:
            X[r, nD + o_index[c]] = 1.0
        X[r, nD + nO + c_index[c]] = age[r]
        if quadratic:
            X[r, nD + nO + nC + c_index[c]] = age[r] ** 2
        if track_evolution:
            X[r, evo_col] = lap[r] - lap_mid

    w = np.ones(len(df))
    beta = np.zeros(ncols)
    for _ in range(4 if robust else 1):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ beta
        if not robust:
            break
        w = _huber_weights(resid)

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    driver_base = {d: float(beta[d_index[d]]) for d in drivers}
    offsets = {base_c: 0.0, **{c: float(beta[nD + o_index[c]]) for c in off_comps}}
    k1 = {c: float(beta[nD + nO + c_index[c]]) for c in comps_present}
    k2 = ({c: float(beta[nD + nO + nC + c_index[c]]) for c in comps_present}
          if quadratic else {c: 0.0 for c in comps_present})

    compounds = {c: CompoundParams(k0=offsets[c], k1=k1[c], k2=k2[c]) for c in comps_present}
    rmse_c = {}
    for c in comps_present:
        mask = np.array([x == c for x in comp_arr], dtype=bool)
        rmse_c[c] = float(np.sqrt(np.mean(resid[mask] ** 2))) if mask.any() else float("nan")

    ref = min(driver_base.values())
    return CalibrationResult(
        circuit_id="", year=0, n_laps=n_laps, compounds=compounds,
        driver_pace=driver_base, reference_pace=ref, rmse=rmse,
        rmse_by_compound=rmse_c, n_obs=len(df),
    )


def fit_compounds_pooled(
    frames: list[tuple[int, pd.DataFrame]], *, fuel: FuelModel | None = None,
    robust: bool = True,
) -> CalibrationResult | None:
    """Pool several races into one fit with **shared** per-compound degradation
    but **per-race** driver fixed-effects and track evolution. Far more stable
    than a single race: more compound overlap across track phases pins down the
    offsets and slopes that one event leaves confounded."""
    if not frames:
        return None
    fuel = fuel or FuelModel()
    parts = []
    for ridx, (n_laps, df) in enumerate(frames):
        if df.empty:
            continue
        d = df.copy()
        d["race"] = ridx
        d["group"] = d["race"].astype(str) + "::" + d["driver"].astype(str)
        d["fuel_corr"] = d["lap_time"] - d["lap"].map(lambda l: fuel.penalty(l, n_laps))
        d["lap_c"] = d["lap"] - d["lap"].median()
        parts.append(d)
    if not parts:
        return None
    df = pd.concat(parts).reset_index(drop=True)

    # Thermal coupling is identified only if we have real per-lap temperatures
    # spanning a range across the pooled races; otherwise drop the term cleanly.
    have_temp = "track_temp" in df.columns and df["track_temp"].notna().any()
    temp_ref = float(df["track_temp"].mean()) if have_temp else 30.0
    use_temp = have_temp and float(df["track_temp"].std(skipna=True) or 0.0) >= 1.0

    groups = sorted(df["group"].unique())
    races = sorted(df["race"].unique())
    comps_present = [c for c in (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
                     if c in set(df["compound"])]
    base_c = Compound.MEDIUM if Compound.MEDIUM in comps_present else \
        df["compound"].value_counts().idxmax()
    off_comps = [c for c in comps_present if c != base_c]

    g_index = {g: i for i, g in enumerate(groups)}
    o_index = {c: i for i, c in enumerate(off_comps)}
    c_index = {c: i for i, c in enumerate(comps_present)}
    r_index = {r: i for i, r in enumerate(races)}
    nG, nO, nC, nR = len(groups), len(off_comps), len(comps_present), len(races)

    # Columns: [group dummies | compound offsets | per-compound age | per-race evo | temp*age]
    temp_col = nG + nO + nC + nR
    ncols = temp_col + (1 if use_temp else 0)
    X = np.zeros((len(df), ncols))
    y = df["fuel_corr"].to_numpy()
    comp_arr = df["compound"].to_numpy(dtype=object)
    age = df["tyre_age"].to_numpy(dtype=float)
    lap_c = df["lap_c"].to_numpy(dtype=float)
    tdev = ((df["track_temp"].to_numpy(dtype=float) - temp_ref) if use_temp
            else np.zeros(len(df)))
    grp = df["group"].to_numpy(dtype=object)
    rc = df["race"].to_numpy()
    for r in range(len(df)):
        X[r, g_index[grp[r]]] = 1.0
        c = comp_arr[r]
        if c in o_index:
            X[r, nG + o_index[c]] = 1.0
        X[r, nG + nO + c_index[c]] = age[r]
        X[r, nG + nO + nC + r_index[rc[r]]] = lap_c[r]
        if use_temp:
            X[r, temp_col] = tdev[r] * age[r]   # thermal slope: hotter -> steeper wear
    w = np.ones(len(df)); beta = np.zeros(ncols); resid = y.copy()
    for _ in range(4 if robust else 1):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ beta
        if not robust:
            break
        w = _huber_weights(resid)

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    offsets = {base_c: 0.0, **{c: float(beta[nG + o_index[c]]) for c in off_comps}}
    k1 = {c: float(beta[nG + nO + c_index[c]]) for c in comps_present}
    compounds = {c: CompoundParams(k0=offsets[c], k1=k1[c], k2=0.0) for c in comps_present}
    temp_coeff = float(beta[temp_col]) if use_temp else 0.0
    rmse_c = {}
    for c in comps_present:
        m = np.array([x == c for x in comp_arr], dtype=bool)
        rmse_c[c] = float(np.sqrt(np.mean(resid[m] ** 2))) if m.any() else float("nan")
    group_base = {g: float(beta[g_index[g]]) for g in groups}
    ref = min(group_base.values())
    return CalibrationResult(
        circuit_id="", year=0, n_laps=frames[-1][0], compounds=compounds,
        driver_pace=group_base, reference_pace=ref, rmse=rmse,
        rmse_by_compound=rmse_c, n_obs=len(df), temp_coeff=temp_coeff, temp_ref=temp_ref,
    )


def _load_race_laps(
    year: int, country: str, backend: str, client: OpenF1Client | None,
) -> pd.DataFrame:
    """Load one race's clean green laps via the chosen backend.

    ``backend='fastf1'`` (richest: exact TrackStatus filtering) falls back to
    OpenF1 automatically if FastF1 is unavailable/blocked."""
    if backend == "fastf1":
        try:
            from .fastf1_loader import fastf1_available, load_clean_laps
            if fastf1_available():
                df = load_clean_laps(year, country, "R")
                if not df.empty:
                    return df
        except Exception:
            pass  # fall through to OpenF1
    client = client or OpenF1Client()
    try:
        sk = client.session_key(year, country, "Race")
        return build_lap_dataset(client, sk)
    except (ValueError, RuntimeError):
        return pd.DataFrame()


def calibrate_races(
    races: list[tuple[int, str]], circuit_id: str, *,
    backend: str = "fastf1", client: OpenF1Client | None = None,
) -> CalibrationResult | None:
    """Calibrate per-compound degradation by pooling several real races (e.g.
    [(2023,'Bahrain'),(2024,'Bahrain'),(2025,'Bahrain')]). ``backend`` is
    'fastf1' (preferred) or 'openf1'."""
    frames = []
    for year, country in races:
        df = _load_race_laps(year, country, backend, client)
        if not df.empty:
            frames.append((int(df["lap"].max()), df))
    res = fit_compounds_pooled(frames)
    if res is not None:
        res.circuit_id = circuit_id
        res.year = races[-1][0]
    return res


# --------------------------------------------------------------------------- #
# Data-derived priors: SC probability, pit loss, reliability, race-pace gap     #
#                                                                               #
# These replace the hardcoded Heilmeier (2014-2019) constants with values fit   #
# from recent real races, and are persisted by ``data.priors`` so the engine    #
# is calibrated end-to-end rather than only in tyre degradation + driver pace.  #
# --------------------------------------------------------------------------- #
def _race_neutralisations(year: int, gp: str) -> tuple[bool, bool] | None:
    """(had_SC, had_VSC) for a real race, from FastF1 per-lap TrackStatus.
    Returns ``None`` if the session can't be loaded."""
    import fastf1
    from .fastf1_loader import _enable_cache
    _enable_cache()
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            ses = fastf1.get_session(year, gp, "R")
            ses.load(telemetry=False, weather=False, messages=False)
    except Exception:
        return None
    laps = ses.laps
    if laps is None or len(laps) == 0:
        return None
    codes = set()
    for s in laps["TrackStatus"].dropna().astype(str):
        codes |= set(s)
    return ("4" in codes, ("6" in codes or "7" in codes))


def calibrate_sc_probs(races: list[tuple[int, str]]) -> dict:
    """Pool several real races into empirical P(>=1 SC) and P(>=1 VSC).
    Pass races for one circuit to get that circuit's rate; the build script
    groups the full calendar. Returns ``{p_sc, p_vsc, n_races}``."""
    n = sc = vsc = 0
    for year, gp in races:
        r = _race_neutralisations(year, gp)
        if r is None:
            continue
        n += 1
        sc += int(r[0])
        vsc += int(r[1])
    if n == 0:
        return {"p_sc": None, "p_vsc": None, "n_races": 0}
    return {"p_sc": round(sc / n, 3), "p_vsc": round(vsc / n, 3), "n_races": n}


def calibrate_pit_loss(races: list[tuple[int, str]]) -> dict:
    """Real pit-lane loss (mean/sd) and the SC/VSC cheap-stop discount, pooled
    over races, plus a per-team mean/sd (pit-crew speed). Returns a dict ready
    for the priors store: ``{green_mean, green_sd, sc_frac, vsc_frac, by_team}``."""
    import pandas as _pd

    from .fastf1_loader import load_pit_durations
    frames = []
    for year, gp in races:
        try:
            d = load_pit_durations(year, gp)
        except Exception:
            continue
        if not d.empty:
            frames.append(d)
    if not frames:
        return {"green_mean": None, "green_sd": None, "sc_frac": None,
                "vsc_frac": None, "by_team": {}, "n": 0}
    df = _pd.concat(frames).reset_index(drop=True)
    green = df[df["regime"] == "green"]["pit_loss"]
    gmean = float(green.median()) if len(green) else float(df["pit_loss"].median())
    gsd = float(green.std()) if len(green) > 1 else 0.0
    out = {"green_mean": round(gmean, 2), "green_sd": round(gsd, 2),
           "n": int(len(df)), "by_team": {}}
    for regime, key in (("SC", "sc_frac"), ("VSC", "vsc_frac")):
        sub = df[df["regime"] == regime]["pit_loss"]
        out[key] = round(float(1.0 - sub.median() / gmean), 3) if len(sub) and gmean else None
    for team, g in df.groupby("team"):
        gl = g[g["regime"] == "green"]["pit_loss"]
        if len(gl) >= 2:
            out["by_team"][str(team)] = {"mean": round(float(gl.median()), 2),
                                         "sd": round(float(gl.std()), 2), "n": int(len(gl))}
    return out


# A classified finish: completed status, lapped, or "+N Lap(s)". Everything else
# is a retirement. Incident retirements (driver/contact) are split out from
# *mechanical* failures so per-team "reliability" reflects the car, not crashes.
def _is_finish(status: str) -> bool:
    s = status.strip()
    return s in ("Finished", "Lapped") or s.startswith("+")


_INCIDENT_STATUSES = {
    "Accident", "Collision", "Collision damage", "Spun off", "Damage",
    "Disqualified", "Withdrew", "Did not qualify", "Did not prequalify",
    "Did not start", "Driver unwell", "Injury",
}


def calibrate_reliability(seasons: list[int], client=None) -> dict:
    """Per-constructor *mechanical* DNF probability from Jolpica results across
    seasons. Classified finishers (incl. lapped / +N Laps) are finishes; of the
    retirements, accidents/contact are excluded so the rate reflects car
    reliability rather than crashes. Returns ``{constructor: {p_dnf, p_retire, n}}``
    plus field-wide ``_overall`` (mechanical) and ``_overall_retire``."""
    from .jolpica import JolpicaClient
    cl = client or JolpicaClient()
    starts: dict[str, int] = {}
    mech: dict[str, int] = {}
    retire: dict[str, int] = {}
    tot_s = tot_m = tot_r = 0
    for season in seasons:
        try:
            sched = cl.season_schedule(season)
        except Exception:
            continue
        for rnd in sched["round"].tolist() if not sched.empty else []:
            try:
                res = cl.race_results(season, int(rnd))
            except Exception:
                continue
            for _, r in res.iterrows():
                team = str(r["constructor"])
                status = str(r["status"])
                starts[team] = starts.get(team, 0) + 1
                tot_s += 1
                if _is_finish(status):
                    continue
                retire[team] = retire.get(team, 0) + 1
                tot_r += 1
                if status.strip() not in _INCIDENT_STATUSES:  # mechanical
                    mech[team] = mech.get(team, 0) + 1
                    tot_m += 1
    out: dict = {}
    for team, n in starts.items():
        out[team] = {"p_dnf": round(mech.get(team, 0) / n, 3),
                     "p_retire": round(retire.get(team, 0) / n, 3), "n": n}
    out["_overall"] = round(tot_m / tot_s, 3) if tot_s else 0.06
    out["_overall_retire"] = round(tot_r / tot_s, 3) if tot_s else 0.10
    return out


def calibrate_wet(races: list[tuple[int, str]]) -> dict:
    """Calibrate the absolute wet-tyre pace deficits from real wet races, using
    each race's *own* dry slick laps as the baseline (so circuit and conditions
    cancel). ``inter_floor`` / ``wet_floor`` are the best-case seconds per lap an
    intermediate / full wet is slower than a dry slick — the anchors of the
    crossover model. The wetness-response curvature stays a physical prior.
    Returns ``{inter_floor, wet_floor, n_inter, n_wet, n_races}``."""
    from .fastf1_loader import load_clean_laps, load_wet_laps
    fuel = FuelModel()

    def best_pace(df, n_laps):
        if df.empty:
            return None, 0
        fc = df["lap_time"] - df["lap"].map(lambda l: fuel.penalty(l, n_laps))
        return float(np.quantile(fc, 0.10)), len(df)

    inter_d, wet_d, ni, nw, nr = [], [], 0, 0, 0
    for year, gp in races:
        try:
            dry = load_clean_laps(year, gp)
            wet = load_wet_laps(year, gp)
        except Exception:
            continue
        if dry.empty or wet.empty:
            continue
        n_laps = int(max(dry["lap"].max(), wet["lap"].max()))
        dry_base, _ = best_pace(dry, n_laps)
        if dry_base is None:
            continue
        nr += 1
        im = wet[wet["compound"] == Compound.INTERMEDIATE]
        wm = wet[wet["compound"] == Compound.WET]
        ib, n_i = best_pace(im, n_laps)
        wb, n_w = best_pace(wm, n_laps)
        if ib is not None and n_i >= 15:
            inter_d.append((ib - dry_base, n_i)); ni += n_i
        if wb is not None and n_w >= 15:
            wet_d.append((wb - dry_base, n_w)); nw += n_w

    def wmean(pairs):
        if not pairs:
            return None
        v = np.array([p[0] for p in pairs]); w = np.array([p[1] for p in pairs], dtype=float)
        return round(float(np.average(v, weights=w)), 2)

    return {"inter_floor": wmean(inter_d), "wet_floor": wmean(wet_d),
            "n_inter": ni, "n_wet": nw, "n_races": nr}


def calibrate_dirty_air(races: list[tuple[int, str]], *, gap0: float = 2.5,
                        rep_gap: float = 0.7) -> dict:
    """Fit the lap-time penalty a car pays for running in another's wake, from
    real telemetry. Pools a circuit's races and regresses fuel-corrected lap time
    on a **proximity** term — ``max(0, gap0 - gap_ahead)`` seconds of closeness,
    counted only on *held-station* laps so a faster car catching a slower one
    doesn't mask the effect — alongside per-(race,driver) fixed effects, compound
    offsets, per-compound degradation and per-race track evolution.

    The coefficient ``theta`` is s lost per second of closeness; the reported
    ``dirty_air_loss_s`` is the penalty at a representative following gap
    (``rep_gap``). Returns the loss, theta, the held-lap count and an honest r."""
    from .fastf1_loader import load_laps_with_traffic
    frames = []
    for year, gp in races:
        try:
            df = load_laps_with_traffic(year, gp)
        except Exception:
            continue
        if not df.empty and "gap_ahead" in df.columns:
            frames.append((int(df["lap"].max()), df))
    if not frames:
        return {"dirty_air_loss_s": None, "theta": 0.0, "n_held": 0, "r": 0.0}

    fuel = FuelModel()
    parts = []
    for ridx, (n_laps, df) in enumerate(frames):
        d = df.copy()
        d["race"] = ridx
        d["group"] = d["race"].astype(str) + "::" + d["driver"].astype(str)
        d["fuel_corr"] = d["lap_time"] - d["lap"].map(lambda l: fuel.penalty(l, n_laps))
        d["lap_c"] = d["lap"] - d["lap"].median()
        ga = d["gap_ahead"].to_numpy(dtype=float)
        held = d["held"].to_numpy(dtype=bool) if "held" in d.columns else (ga < gap0)
        prox = np.where(held & np.isfinite(ga), np.maximum(0.0, gap0 - ga), 0.0)
        d["prox"] = prox
        parts.append(d)
    df = pd.concat(parts).reset_index(drop=True)
    n_held = int((df["prox"] > 0).sum())
    if n_held < 30:
        return {"dirty_air_loss_s": None, "theta": 0.0, "n_held": n_held, "r": 0.0}

    groups = sorted(df["group"].unique())
    races_ = sorted(df["race"].unique())
    comps_present = [c for c in (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
                     if c in set(df["compound"])]
    base_c = Compound.MEDIUM if Compound.MEDIUM in comps_present else \
        df["compound"].value_counts().idxmax()
    off_comps = [c for c in comps_present if c != base_c]

    g_index = {g: i for i, g in enumerate(groups)}
    o_index = {c: i for i, c in enumerate(off_comps)}
    c_index = {c: i for i, c in enumerate(comps_present)}
    r_index = {r: i for i, r in enumerate(races_)}
    nG, nO, nC, nR = len(groups), len(off_comps), len(comps_present), len(races_)
    prox_col = nG + nO + nC + nR
    ncols = prox_col + 1

    X = np.zeros((len(df), ncols))
    y = df["fuel_corr"].to_numpy()
    comp_arr = df["compound"].to_numpy(dtype=object)
    age = df["tyre_age"].to_numpy(dtype=float)
    lap_c = df["lap_c"].to_numpy(dtype=float)
    grp = df["group"].to_numpy(dtype=object)
    rc = df["race"].to_numpy()
    prox = df["prox"].to_numpy(dtype=float)
    for r in range(len(df)):
        X[r, g_index[grp[r]]] = 1.0
        c = comp_arr[r]
        if c in o_index:
            X[r, nG + o_index[c]] = 1.0
        X[r, nG + nO + c_index[c]] = age[r]
        X[r, nG + nO + nC + r_index[rc[r]]] = lap_c[r]
        X[r, prox_col] = prox[r]
    w = np.ones(len(df)); beta = np.zeros(ncols); resid = y.copy()
    for _ in range(4):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ beta
        w = _huber_weights(resid)
    theta = float(beta[prox_col])
    loss = max(0.0, theta * (gap0 - rep_gap))
    # r of the proximity column's partial contribution, sign-aware.
    pc = X[:, prox_col]
    denom = float(np.sqrt(np.sum((pc - pc.mean()) ** 2) * np.sum((y - y.mean()) ** 2)))
    r = float(np.sum((pc - pc.mean()) * (y - y.mean())) / denom) if denom > 0 else 0.0
    return {"dirty_air_loss_s": round(min(loss, 1.2), 3), "theta": round(theta, 4),
            "n_held": n_held, "r": round(r, 3)}


def calibrate_temp_coupling(frames: list[pd.DataFrame]) -> dict:
    """Two-stage estimate of how degradation steepens with track temperature.

    Stage 1: fit each race's per-compound slope ``k1`` independently (so circuit
    asphalt and track-evolution are absorbed per race). Stage 2: regress those
    slopes on race-mean TrackTemp *with compound fixed effects*, so the
    coefficient measures the within-compound thermal sensitivity rather than the
    (confounded) cross-compound or cross-circuit differences that biased a naive
    single-stage pooled fit. Returns ``{temp_coeff, temp_ref, r, n_points}``."""
    pts: list[tuple] = []  # (compound, k1, mean_temp, n_obs)
    for df in frames:
        if df.empty or "track_temp" not in df.columns or df["track_temp"].isna().all():
            continue
        res = fit_compounds(df, n_laps=int(df["lap"].max()))
        if res is None:
            continue
        t = float(df["track_temp"].mean())
        for c, cp in res.compounds.items():
            n = int((df["compound"] == c).sum())
            if n >= 8 and 0.0 < cp.k1 < 0.4:  # ignore degenerate/implausible slopes
                pts.append((c, cp.k1, t, n))
    if len(pts) < 6:
        return {"temp_coeff": 0.0, "temp_ref": 30.0, "r": 0.0, "n_points": len(pts)}

    comps = sorted({p[0] for p in pts}, key=lambda c: c.value)
    c_idx = {c: i for i, c in enumerate(comps)}
    nC = len(comps)
    w = np.array([p[3] for p in pts], dtype=float)
    temps = np.array([p[2] for p in pts], dtype=float)
    tref = float(np.average(temps, weights=w))
    y = np.array([p[1] for p in pts], dtype=float)
    X = np.zeros((len(pts), nC + 1))
    for i, (c, _k, t, _n) in enumerate(pts):
        X[i, c_idx[c]] = 1.0
        X[i, nC] = t - tref
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    coeff = float(beta[nC])
    # Weighted correlation of the thermal component vs slope, for an honest r.
    pred = X @ beta
    resid = y - pred
    ss_res = float(np.sum(w * resid ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"temp_coeff": round(coeff, 6), "temp_ref": round(tref, 2),
            "r": round(float(np.sign(coeff) * np.sqrt(max(0.0, r2))), 3),
            "n_points": len(pts)}


def calibrate_race_pace_gap(year: int, gp: str, circuit_id: str,
                            cal: "CalibrationResult | None" = None) -> float | None:
    """Race-pace gap = calibrated race base pace − pole lap (s). Anchors absolute
    pace to a circuit's real qualifying instead of the flat 4.0 s TUM prior."""
    from .fastf1_loader import load_quali_ref
    cal = cal or calibrate_race(year, gp, circuit_id)
    if cal is None:
        return None
    q = load_quali_ref(year, gp)
    if not q:
        return None
    pole = min(q.values())
    gap = cal.reference_pace - pole
    return round(float(gap), 3) if 1.0 < gap < 8.0 else None


def calibrate_race(
    year: int, country: str, circuit_id: str, *, n_laps: int | None = None,
    backend: str = "fastf1", client: OpenF1Client | None = None,
) -> CalibrationResult | None:
    """End-to-end: pull one real race and fit the tyre/pace model. ``backend`` is
    'fastf1' (preferred, exact green-lap filtering) or 'openf1'."""
    df = _load_race_laps(year, country, backend, client)
    if df.empty:
        return None
    laps_count = n_laps or int(df["lap"].max())
    res = fit_compounds(df, n_laps=laps_count)
    if res is None:
        return None
    res.circuit_id = circuit_id
    res.year = year
    # FastF1 driver columns are already acronyms; OpenF1 are numbers -> map them.
    if all(str(d).isdigit() for d in res.driver_pace):
        cl = client or OpenF1Client()
        try:
            sk = cl.session_key(year, country, "Race")
            drv = cl.drivers(sk)
            num2acr = dict(zip(drv["driver_number"], drv["name_acronym"])) if not drv.empty else {}
            res.driver_pace = {num2acr.get(int(d), str(d)): p for d, p in res.driver_pace.items()}
        except Exception:
            pass
    res.reference_pace = min(res.driver_pace.values())
    return res
