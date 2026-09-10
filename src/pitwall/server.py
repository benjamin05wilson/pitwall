"""FastAPI backend exposing the pitwall engine to the web frontend.

Run (dev):   uvicorn pitwall.server:app --reload --port 8000
Run (prod):  pitwall-serve            # serves the built React app too

Models, rival fields, scenario banks and the surrogate are cached per circuit so
the interactive endpoints stay snappy. The surrogate powers the instant heatmap;
Monte-Carlo powers the robust recommendation.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import json
import math

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal


def _clean(o):
    """Recursively convert numpy/NaN to JSON-safe Python types."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


class CleanJSON(JSONResponse):
    """Response that sanitises numpy/NaN — returned directly so FastAPI skips its
    own (numpy-blind) serialisation."""

    def render(self, content) -> bytes:
        return json.dumps(_clean(content), allow_nan=False).encode("utf-8")

from .models import RaceModel
from .models.circuits import CIRCUITS
from .optimize import optimize as run_optimize
from .sim import ScenarioSet, build_field, evaluate, with_focal
from .types import Compound, Stint, Strategy

app = FastAPI(title="pitwall API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_CODE = {"S": Compound.SOFT, "M": Compound.MEDIUM, "H": Compound.HARD}
_HEATMAP_PAIRS = [
    (Compound.SOFT, Compound.HARD), (Compound.SOFT, Compound.MEDIUM),
    (Compound.MEDIUM, Compound.HARD), (Compound.MEDIUM, Compound.SOFT),
    (Compound.HARD, Compound.MEDIUM), (Compound.HARD, Compound.SOFT),
]


# --------------------------------------------------------------------------- #
# Cached engine pieces                                                         #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _model(circuit: str) -> RaceModel:
    return RaceModel.for_circuit(circuit)


@lru_cache(maxsize=64)
def _rivals(circuit: str):
    return build_field(circuit, n_cars=20, seed=1)


@lru_cache(maxsize=128)
def _scenarios(circuit: str, n: int) -> ScenarioSet:
    m = _model(circuit)
    return ScenarioSet.sample(m.safety_car, m.config.n_laps, n, seed=7)


@lru_cache(maxsize=1)
def _surrogate():
    try:
        from .surrogate import load
        return load()
    except Exception:
        return None


def _parse_strategy(spec: str, n_laps: int) -> Strategy:
    parts = spec.upper().split("-")
    comps = [_CODE[p[0]] for p in parts]
    n = len(comps)
    base = n_laps // n
    lengths = [base] * (n - 1) + [n_laps - base * (n - 1)]
    return Strategy([Stint(c, l) for c, l in zip(comps, lengths)])


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #
class OptimizeReq(BaseModel):
    circuit: str = "bahrain"
    grid: int = Field(default=3, ge=1, le=20)
    delta: float = Field(default=0.3, ge=-5, le=5)
    objective: Literal["podium", "win", "points", "expected", "robust"] = "podium"
    scenarios: int = Field(default=20, ge=1, le=100)


class HeatmapReq(BaseModel):
    circuit: str = "bahrain"
    grid: int = Field(default=3, ge=1, le=20)
    delta: float = Field(default=0.3, ge=-5, le=5)


class SimReq(BaseModel):
    circuit: str = "bahrain"
    strategy: str = "M-H"
    grid: int = Field(default=3, ge=1, le=20)
    delta: float = Field(default=0.3, ge=-5, le=5)
    scenarios: int = Field(default=20, ge=1, le=100)


class CalibrateReq(BaseModel):
    country: str = "Bahrain"
    circuit: str = "bahrain"
    years: list[int] = [2023, 2024, 2025]


class BacktestReq(BaseModel):
    year: int = 2023
    country: str = "Bahrain"
    circuit: str = "bahrain"
    pool_years: list[int] | None = [2023, 2024, 2025]
    scenarios: int = 250


def _ens_dict(ens, grid: int) -> dict:
    return {
        "mean_pos": round(ens.mean_position, 2),
        "median_pos": ens.median_position,
        "p_win": round(ens.p_win, 3),
        "p_podium": round(ens.p_podium, 3),
        "p_points": round(ens.p_points, 3),
        "cvar10": round(ens.cvar_position(), 2),
        "std_pos": round(ens.std_position, 2),
        "dnf": round(ens.dnf_rate, 3),
    }


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@app.get("/api/circuits")
def circuits() -> list[dict]:
    return [
        {"id": c.circuit_id, "name": c.name, "n_laps": c.n_laps,
         "pit_loss": c.pit_loss_s, "overtake": c.overtake_threshold_s,
         "p_sc": c.p_safety_car}
        for c in CIRCUITS.values()
    ]


@app.post("/api/optimize")
def optimize(req: OptimizeReq) -> dict:
    if req.circuit not in CIRCUITS:
        raise HTTPException(404, "unknown circuit")
    model = _model(req.circuit)
    scen = _scenarios(req.circuit, req.scenarios)
    t0 = time.time()
    from .sim.native import backend_for
    probe = with_focal(_rivals(req.circuit), _parse_strategy("M-H", model.config.n_laps),
                       circuit_id=req.circuit, focal_model=model, focal_delta=req.delta)
    backend = backend_for(probe, scen)
    res = run_optimize(model, _rivals(req.circuit), scen, circuit_id=req.circuit,
                       focal_grid=req.grid, focal_delta=req.delta,
                       objective=req.objective, shortlist=3, use_native=backend == "Rust")
    ms = (time.time() - t0) * 1000
    best, runner = res.best, res.ranked[1] if len(res.ranked) > 1 else res.best
    gap = runner.ensemble.mean_position - best.ensemble.mean_position
    return {
        "objective": req.objective, "scenarios": len(scen), "compute_ms": round(ms),
        "backend": backend, "focal_model": "circuit defaults + explicit driver offset",
        "rival_model": "generic circuit defaults; seeded pace offsets",
        "seed": 7, "shortlist": len(res.ranked), "n_laps": model.config.n_laps,
        "best": {"label": best.strategy.label(), "n_stops": best.strategy.n_stops,
                 "pit_laps": list(best.strategy.pit_laps),
                 "compounds": [c.value for c in best.strategy.compounds],
                 **_ens_dict(best.ensemble, req.grid)},
        "reason": {
            "runner_up": runner.strategy.label(),
            "gap": round(gap, 2),
            "pit_loss": model.pit.pit_loss_s,
            "n_scenarios": len(scen),
        },
        "ranked": [{"label": s.strategy.label(), "stops": s.strategy.n_stops,
                    "det_time": round(s.det_time, 1), **_ens_dict(s.ensemble, req.grid)}
                   for s in res.ranked],
        "frontier": [{"label": s.strategy.label(),
                      "mean_pos": round(s.ensemble.mean_position, 2),
                      "cvar10": round(s.ensemble.cvar_position(), 2)}
                     for s in res.ranked],
        "frontier_optimal": [s.strategy.label() for s in res.frontier()],
    }


@app.post("/api/heatmap")
def heatmap(req: HeatmapReq) -> dict:
    net = _surrogate()
    if net is None:
        raise HTTPException(503, "Optional heatmap unavailable: no usable surrogate checkpoint. Core optimisation remains available.")
    from .surrogate import encode
    import numpy as np
    model = _model(req.circuit)
    n_laps = model.config.n_laps
    laps = list(range(6, n_laps - 4))
    feats, meta = [], []
    for ci, (a, b) in enumerate(_HEATMAP_PAIRS):
        for p in laps:
            strat = Strategy([Stint(a, p), Stint(b, n_laps - p)])
            feats.append(encode(model, strat, focal_grid=req.grid,
                                focal_delta=req.delta, pace_spread=1.5))
            meta.append((ci, p))
    t0 = time.time()
    pred = net.predict(np.asarray(feats))
    ms = (time.time() - t0) * 1000
    z = [[None] * len(laps) for _ in _HEATMAP_PAIRS]
    for (ci, p), mp in zip(meta, pred["mean_pos"]):
        z[ci][laps.index(p)] = round(float(mp), 2)
    # Best cell.
    best_idx = int(np.argmin(pred["mean_pos"]))
    bci, bp = meta[best_idx]
    return {
        "laps": laps,
        "rows": [f"{a.value[0]}-{b.value[0]}" for a, b in _HEATMAP_PAIRS],
        "z": z, "n": len(feats), "compute_ms": round(ms, 1),
        "best": {"row": f"{_HEATMAP_PAIRS[bci][0].value[0]}-{_HEATMAP_PAIRS[bci][1].value[0]}",
                 "lap": bp, "mean_pos": round(float(pred["mean_pos"][best_idx]), 2)},
    }


@app.post("/api/simulate")
def simulate(req: SimReq) -> dict:
    model = _model(req.circuit)
    strat = _parse_strategy(req.strategy, model.config.n_laps)
    from .sim.native import HAS_NATIVE
    field = with_focal(_rivals(req.circuit), strat, circuit_id=req.circuit,
                       focal_grid=req.grid, focal_delta=req.delta, focal_model=model)
    ens = evaluate(field, _scenarios(req.circuit, req.scenarios), use_native=HAS_NATIVE)
    hist = [int(x) for x in ens.positions]
    import numpy as np
    counts = np.bincount(np.array(hist), minlength=len(field) + 1)[1:].tolist()
    return {"strategy": strat.label(), **_ens_dict(ens, req.grid), "histogram": counts}


@app.post("/api/sc-counterfactual")
def sc_counterfactual(req: HeatmapReq) -> dict:
    import numpy as np
    from .live import advise
    from .live.updater import TyreBelief
    model = _model(req.circuit)
    n_laps = model.config.n_laps
    mid = n_laps // 2
    belief = TyreBelief(model.pace.base, 0.08, np.diag([0.5, 0.0004]))

    def panel(regime: str) -> dict:
        if regime == "green":
            a = advise(belief, current_lap=mid, current_age=mid // 2, n_laps=n_laps,
                       pit_loss=model.pit.effective_loss("green"))
        else:
            a = advise(belief, current_lap=mid, current_age=mid // 2, n_laps=n_laps,
                       pit_loss=model.pit.effective_loss("green"),
                       cheap_until=mid + 3, cheap_pit_loss=model.pit.effective_loss(regime))
        return {"pit_lap": a.optimal_pit_lap, "lo": a.window_lo, "hi": a.window_hi,
                "pit_loss": round(model.pit.effective_loss(regime if regime != "green" else "green"), 1)}

    return {
        "current_lap": mid,
        "green": panel("green"),
        "vsc": panel("VSC"),
        "sc": panel("SC"),
        # A red flag is a free stop (pit loss ~0) — the biggest swing of all.
        "red_flag": {"pit_loss": 0.0, "note": "free tyre change"},
        "p_sc": model.safety_car.p_at_least_one_override or 0.545,
        "p_red": model.safety_car.red_flag_prob,
    }


@app.post("/api/calibrate")
def calibrate(req: CalibrateReq) -> dict:
    from .data.calibrate import calibrate_races
    cal = calibrate_races([(y, req.country) for y in req.years], req.circuit)
    if cal is None:
        raise HTTPException(502, "calibration failed (no data / offline)")
    # Compare prior vs calibrated optimum.
    from .optimize import enumerate_candidates
    prior = [c.strategy.label() for c in enumerate_candidates(_model(req.circuit), top_k=1)]
    post = [c.strategy.label() for c in enumerate_candidates(cal.to_race_model(req.circuit), top_k=1)]
    return {
        "n_obs": cal.n_obs, "rmse": round(cal.rmse, 3),
        "compounds": {c.value: {"k0": round(p.k0, 3), "k1": round(p.k1, 4)}
                      for c, p in cal.compounds.items()},
        "driver_deltas": {d: round(v, 3) for d, v in
                          sorted(cal.driver_deltas().items(), key=lambda x: x[1])[:10]},
        "prior_optimum": prior[0] if prior else None,
        "calibrated_optimum": post[0] if post else None,
    }


@app.post("/api/backtest")
def backtest(req: BacktestReq) -> dict:
    from .data.backtest import reconstruct_race
    res = reconstruct_race(req.year, req.country, req.circuit,
                           n_scenarios=req.scenarios, pool_years=req.pool_years)
    if res is None:
        raise HTTPException(502, "backtest failed (no data / offline)")
    return {**res.summary(),
            "predicted_order": res.predicted_order[:15],
            "actual_order": res.actual_order[:15]}


# --------------------------------------------------------------------------- #
# Live race replay                                                             #
# --------------------------------------------------------------------------- #
# Curated demo races (Williams-relevant + varied strategy). FastF1 has 2018+.
REPLAY_RACES = [
    {"year": 2023, "gp": "Bahrain", "label": "2023 Bahrain — Albon P10 (points)"},
    {"year": 2023, "gp": "Canada", "label": "2023 Canada — Albon P7"},
    {"year": 2023, "gp": "Italy", "label": "2023 Italy (Monza) — low-deg 1-stop"},
    {"year": 2023, "gp": "Monaco", "label": "2023 Monaco — track position"},
    {"year": 2024, "gp": "Bahrain", "label": "2024 Bahrain"},
    {"year": 2024, "gp": "Saudi Arabia", "label": "2024 Saudi Arabia"},
]


class RaceReq(BaseModel):
    year: int = 2023
    gp: str = "Bahrain"


class ReplayReq(BaseModel):
    year: int = 2023
    gp: str = "Bahrain"
    driver: str = "ALB"


@lru_cache(maxsize=16)
def _replay(year: int, gp: str, driver: str) -> dict:
    from .replay import build_replay
    return build_replay(year, gp, driver)


@lru_cache(maxsize=16)
def _drivers(year: int, gp: str) -> tuple:
    from .replay import list_drivers
    m = list_drivers(year, gp)
    return tuple(m["drivers"]) if m else tuple()


@app.get("/api/replay-options")
def replay_options() -> list[dict]:
    return REPLAY_RACES


@app.post("/api/replay-drivers")
def replay_drivers(req: RaceReq) -> dict:
    try:
        return {"drivers": list(_drivers(req.year, req.gp))}
    except Exception as exc:  # offline / data missing
        raise HTTPException(502, f"could not load drivers: {exc}")


@app.post("/api/replay")
def replay(req: ReplayReq):
    try:
        return CleanJSON(_replay(req.year, req.gp, req.driver))
    except Exception as exc:
        raise HTTPException(502, f"replay failed: {exc}")


@lru_cache(maxsize=8)
def _field_state(year: int, gp: str) -> dict:
    from .field_model import build_field_state
    return build_field_state(year, gp)


@lru_cache(maxsize=24)
def _field_strategy(year: int, gp: str, driver: str) -> dict:
    from .field_model import build_field_strategy
    return build_field_strategy(year, gp, driver, field_state=_field_state(year, gp))


@lru_cache(maxsize=16)
def _ghost(year: int, gp: str, driver: str) -> dict:
    from .ghost import build_ghost
    return build_ghost(year, gp, driver, n_scenarios=80, pool_years=[2023, 2024, 2025])


@app.post("/api/ghost")
def ghost(req: ReplayReq):
    """Counterfactual: what would have happened if the driver had followed the AI."""
    try:
        return CleanJSON(_ghost(req.year, req.gp, req.driver))
    except Exception as exc:
        raise HTTPException(502, f"ghost failed: {exc}")


@app.post("/api/field-strategy")
def field_strategy(req: ReplayReq):
    """The chosen car's historical timeline as a best response to the *predicted* field
    (undercut/overcut vs every rival)."""
    try:
        return CleanJSON(_field_strategy(req.year, req.gp, req.driver))
    except Exception as exc:
        raise HTTPException(502, f"field strategy failed: {exc}")


@lru_cache(maxsize=16)
def _wet_strategy(year: int, gp: str, driver: str) -> dict:
    from .wet_strategy import analyze_wet_race
    return analyze_wet_race(year, gp, driver or None)


@app.post("/api/wet-strategy")
def wet_strategy(req: ReplayReq):
    """Wet-race tyre-crossover analysis: the per-lap wetness, the AI's optimal
    slick/inter/wet switch laps, and (if a driver is given) their real stops."""
    try:
        return CleanJSON(_wet_strategy(req.year, req.gp, req.driver))
    except Exception as exc:
        raise HTTPException(502, f"wet strategy failed: {exc}")


@lru_cache(maxsize=6)
def _racemap(year: int, gp: str) -> dict:
    from .racemap import race_frames
    return race_frames(year, gp, step=2.0)


@app.post("/api/race-map")
def race_map(req: RaceReq):
    try:
        return CleanJSON(_racemap(req.year, req.gp))
    except Exception as exc:
        raise HTTPException(502, f"race map failed: {exc}")


@app.get("/api/live-stream")
def live_stream(year: int = 2023, gp: str = "Bahrain", driver: str = "ALB", interval: float = 1.0):
    """Server-Sent Events for historical replay only; never current telemetry."""
    from .replay import stream_events

    def gen():
        try:
            for msg in stream_events(year, gp, driver, interval=max(0.05, interval)):
                yield f"data: {json.dumps(_clean(msg), allow_nan=False)}\n\n"
        except Exception as exc:  # pragma: no cover
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health() -> dict:
    from .sim.native import HAS_NATIVE
    return {"ok": True, "surrogate": _surrogate() is not None,
            "native_available": HAS_NATIVE, "replay_source": "historical-replay"}


# --------------------------------------------------------------------------- #
# Serve the built React app (if present)                                       #
# --------------------------------------------------------------------------- #
_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="web")


def main() -> None:
    import uvicorn
    uvicorn.run("pitwall.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
