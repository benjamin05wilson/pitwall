"""Build the calibrated priors store from real data.

Sweeps recent real races and fits the constants that were previously hardcoded
from 2014-2019 published work:

  * per-circuit P(>=1 SC/VSC), pit-lane loss (mean+sd), cheap-stop discounts,
    and the race-pace gap (base - pole), from FastF1;
  * a shared tyre thermal coupling (s/lap per °C) from pooled, weather-joined
    laps spanning hot and cool venues;
  * per-constructor mechanical-DNF probability from Jolpica results.

Writes ``src/pitwall/data/calibrated/priors.json`` (read by ``data.priors``).
Re-runnable; uses only cached sessions where possible.

    python scripts/build_priors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitwall.data.calibrate import (  # noqa: E402
    calibrate_dirty_air,
    calibrate_pit_loss,
    calibrate_race_pace_gap,
    calibrate_reliability,
    calibrate_sc_probs,
    calibrate_temp_coupling,
    calibrate_wet,
)
from pitwall.data.fastf1_loader import load_clean_laps  # noqa: E402
from pitwall.data.priors import save_priors  # noqa: E402
from pitwall.models.circuits import get_profile  # noqa: E402

# Shrink a small-sample empirical rate toward a prior with K pseudo-observations.
SC_SHRINK_K = 4.0

# circuit_id -> (FastF1 GP name, years with cached/available data)
CIRCUITS = {
    "bahrain": ("Bahrain", [2023, 2024, 2025]),
    "jeddah": ("Saudi Arabia", [2023, 2024, 2025]),
    "villeneuve": ("Canada", [2023, 2024, 2025]),
    "suzuka": ("Japan", [2023, 2024, 2025]),
    "monza": ("Italy", [2023, 2024, 2025]),
    "monaco": ("Monaco", [2023, 2024, 2025]),
}
RELIABILITY_SEASONS = [2023, 2024, 2025]
# Cached wet races used to calibrate the intermediate / full-wet pace deficits.
WET_RACES = [(2023, "Netherlands"), (2023, "Monaco")]


def log(msg: str) -> None:
    print(msg, flush=True)


def build_circuit(cid: str, gp: str, years: list[int]) -> dict:
    races = [(y, gp) for y in years]
    out: dict = {"gp": gp, "years": years}
    # P(>=1 SC), shrunk toward the research prior (n is only ~3 races).
    sc = calibrate_sc_probs(races)
    prior_psc = get_profile(cid).p_safety_car
    n = sc["n_races"]
    if n and sc["p_sc"] is not None:
        out["p_sc"] = round((sc["p_sc"] * n + prior_psc * SC_SHRINK_K) / (n + SC_SHRINK_K), 3)
        out["p_sc_raw"] = sc["p_sc"]
    out["p_vsc"], out["n_races"] = sc["p_vsc"], n
    # Real pit-lane loss mean + crew variance (sc/vsc discount is NOT recoverable
    # from PitOut-PitIn — that measures pit-lane time, not time lost vs a slowed
    # field — so we keep the research SC/VSC saving fractions).
    pit = calibrate_pit_loss(races)
    out["pit_loss_mean"] = pit["green_mean"]
    out["pit_loss_sd"] = pit["green_sd"]
    out["pit_by_team"] = pit["by_team"]
    # race-pace gap from the most recent year with a qualifying session
    for y in reversed(years):
        gap = calibrate_race_pace_gap(y, gp, cid)
        if gap is not None:
            out["race_pace_gap"] = gap
            break
    # Dirty-air pace loss from real wake telemetry (ship only a positive fit).
    da = calibrate_dirty_air(races)
    if da["dirty_air_loss_s"] is not None and da["theta"] > 0 and da["n_held"] >= 200:
        out["dirty_air_loss_s"] = da["dirty_air_loss_s"]
        out["dirty_air_n"] = da["n_held"]
    log(f"  {cid:12s} p_sc={out.get('p_sc')} (raw {out.get('p_sc_raw')}) "
        f"pit={out['pit_loss_mean']}±{out['pit_loss_sd']} gap={out.get('race_pace_gap')} "
        f"dirty_air={out.get('dirty_air_loss_s')} (n={n})")
    return out


def build_temp_coupling() -> dict:
    """Two-stage thermal estimate across hot and cool venues: per-race compound
    slopes vs race-mean track temp, with compound fixed effects."""
    frames = []
    temps = []
    for cid, (gp, years) in CIRCUITS.items():
        for y in years:
            try:
                df = load_clean_laps(y, gp)
            except Exception:
                continue
            if not df.empty and "track_temp" in df.columns:
                frames.append(df)
                temps.append(float(df["track_temp"].mean()))
    if not frames:
        return {"temp_coeff": 0.0, "temp_ref": 30.0, "n_points": 0}
    res = calibrate_temp_coupling(frames)
    # Physical prior: a negative thermal slope (cooler->faster wear) is not
    # credible at the S/M/H level, so an indefinite/wrong-signed estimate is held
    # inert rather than shipped. Reported honestly either way.
    if res["temp_coeff"] <= 0.0:
        log(f"  temp_coeff={res['temp_coeff']:+.5f} (n={res['n_points']}, r={res['r']}) "
            f"-> no robust positive thermal signal at S/M/H level; held inert (0.0)")
        res = {**res, "temp_coeff": 0.0, "shipped": 0.0}
    else:
        log(f"  temp_coeff={res['temp_coeff']:+.5f} s/lap/°C  temp_ref={res['temp_ref']:.1f}°C  "
            f"(range {min(temps):.1f}..{max(temps):.1f}, {res['n_points']} pts, r={res['r']})")
        res["shipped"] = res["temp_coeff"]
    return res


def main() -> None:
    log("Building calibrated priors from real data...\n[1/3] per-circuit SC / pit / pace")
    circuits = {}
    for cid, (gp, years) in CIRCUITS.items():
        try:
            circuits[cid] = build_circuit(cid, gp, years)
        except Exception as exc:  # noqa: BLE001
            log(f"  {cid}: FAILED {exc}")

    log("[2/4] tyre thermal coupling (pooled across venues)")
    tyre_temp = build_temp_coupling()

    log("[3/4] wet-tyre pace deficits (from real wet races)")
    wet = calibrate_wet(WET_RACES)
    log(f"  inter_floor={wet['inter_floor']}s  wet_floor={wet['wet_floor']}s  "
        f"(n_inter={wet['n_inter']} n_wet={wet['n_wet']} over {wet['n_races']} races)")

    log("[4/4] per-constructor reliability (Jolpica)")
    reliability = calibrate_reliability(RELIABILITY_SEASONS)
    teams = {k: v for k, v in reliability.items() if not k.startswith("_")}
    log(f"  {len(teams)} constructors, overall p_dnf={reliability.get('_overall')}")
    for t, v in sorted(teams.items(), key=lambda kv: -kv[1]["p_dnf"])[:6]:
        log(f"    {t:14s} p_dnf={v['p_dnf']} (n={v['n']})")

    data = {
        "meta": {"source": "FastF1 + Jolpica", "seasons": RELIABILITY_SEASONS,
                 "note": "Regenerate with scripts/build_priors.py"},
        "circuits": circuits,
        "tyre_temp": tyre_temp,
        "wet": wet,
        "reliability": reliability,
    }
    save_priors(data)
    from pitwall.data.priors import PRIORS_PATH
    log(f"\nWrote {PRIORS_PATH}")


if __name__ == "__main__":
    main()
