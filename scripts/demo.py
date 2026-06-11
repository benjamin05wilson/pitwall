"""End-to-end showcase — reproduces the headline results in one command.

    PYTHONPATH=src python scripts/demo.py

Network-dependent steps (calibration, live) degrade gracefully if offline.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from pitwall.models import RaceModel  # noqa: E402
from pitwall.optimize import enumerate_candidates, optimize  # noqa: E402
from pitwall.sim import ScenarioSet, build_field, evaluate, with_focal  # noqa: E402


def rule(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    circuit = "bahrain"
    model = RaceModel.for_circuit(circuit)

    rule("1. ROBUST STRATEGY OPTIMISATION (priors)")
    rivals = build_field(circuit, seed=1)
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, 400, seed=7)
    res = optimize(model, rivals, scen, circuit_id=circuit, focal_grid=3,
                   focal_delta=0.3, objective="podium")
    print(f"Recommended from P3: {res.best.strategy.label()}  {res.best.ensemble.summary()}")

    rule("2. CALIBRATION ON REAL DATA — does it change the answer?")
    try:
        from pitwall.data.calibrate import calibrate_races
        cal = calibrate_races([(2023, "Bahrain"), (2024, "Bahrain"), (2025, "Bahrain")], circuit)
        print(f"Fitted on {cal.n_obs} real green laps, RMSE {cal.rmse:.2f}s")
        for c, p in cal.compounds.items():
            print(f"   {c.value:7} deg = {p.k1:.4f} s/lap")
        pri = [c.strategy.label() for c in enumerate_candidates(model, top_k=3)]
        cal_model = cal.to_race_model(circuit)
        post = [c.strategy.label() for c in enumerate_candidates(cal_model, top_k=3)]
        print(f"   prior optimum:      {pri[0]}  (1-stop-ish)")
        print(f"   calibrated optimum: {post[0]}  (real-data)")
    except Exception as exc:  # offline
        print(f"   (skipped — needs network: {exc})")

    rule("2b. RACE RECONSTRUCTION — can the sim re-run history?")
    try:
        from pitwall.data.backtest import reconstruct_race
        bt = reconstruct_race(2023, "Bahrain", circuit, n_scenarios=250,
                              pool_years=[2023, 2024, 2025])
        if bt:
            print(f"   2023 Bahrain: Spearman rank corr {bt.spearman:.2f} vs real result, "
                  f"MAE {bt.mae_position:.1f} positions, podium {bt.podium_accuracy:.0%}")
            print(f"   predicted top 5: {bt.predicted_order[:5]}")
            print(f"   actual    top 5: {bt.actual_order[:5]}")
    except Exception as exc:
        print(f"   (skipped — needs FastF1/network: {exc})")

    rule("3. LEARNED SURROGATE — real-time what-if")
    try:
        from pitwall.surrogate import encode, load
        net = load()
        cands = enumerate_candidates(model, top_k=12)
        feats = np.array([encode(model, c.strategy, focal_grid=3, focal_delta=0.3,
                                 pace_spread=1.5) for c in cands])
        big = np.repeat(feats, 200, axis=0)
        t0 = time.time(); net.predict(big); dt = (time.time() - t0) * 1000
        errs = []
        for c in cands:
            mc = evaluate(with_focal(rivals, c.strategy, circuit_id=circuit, focal_grid=3,
                                     focal_delta=0.3), scen).mean_position
            sp = float(net.predict(encode(model, c.strategy, focal_grid=3, focal_delta=0.3,
                                          pace_spread=1.5))["mean_pos"])
            errs.append(abs(sp - mc))
        print(f"   scored {len(big)} strategies in {dt:.1f} ms ({1000*len(big)/dt:,.0f}/s)")
        print(f"   accuracy vs Monte-Carlo: {np.mean(errs):.2f} positions mean error")
    except Exception as exc:
        print(f"   (skipped — build surrogate first: {exc})")

    rule("4. LIVE BAYESIAN TYRE UPDATER vs baselines (real stint)")
    try:
        from pitwall.data.calibrate import build_lap_dataset
        from pitwall.data.openf1 import OpenF1Client
        from pitwall.live import online_prediction_benchmark
        cl = OpenF1Client(); sk = cl.session_key(2023, "Bahrain", "Race")
        df = build_lap_dataset(cl, sk); nl = int(df["lap"].max())
        df["k"] = df["driver"].astype(str) + "/" + df["compound"].astype(str)
        g = df.groupby("k"); key = max(g.groups, key=lambda k: len(g.get_group(k)))
        st = g.get_group(key).sort_values("lap")
        b = online_prediction_benchmark(st["lap"].values, st["tyre_age"].values,
                                        st["lap_time"].values, n_laps=nl)
        print(f"   one-step-ahead RMSPE — Kalman {b['kalman_rmspe']:.3f}s | "
              f"persistence {b['persistence_rmspe']:.3f}s | OLS {b['ols_rmspe']:.3f}s")
    except Exception as exc:
        print(f"   (skipped — needs network: {exc})")

    rule("5. RUST ACCELERATOR")
    try:
        from pitwall.sim.native import HAS_NATIVE
        if HAS_NATIVE:
            field = with_focal(rivals, res.best.strategy, circuit_id=circuit, focal_grid=3,
                               focal_delta=0.3)
            big_scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, 2000, seed=11)
            t0 = time.time(); evaluate(field, big_scen); py = time.time() - t0
            t0 = time.time(); evaluate(field, big_scen, use_native=True); rs = time.time() - t0
            print(f"   2000 races — Python {py:.2f}s | Rust {rs:.3f}s | speedup {py/rs:.0f}x")
        else:
            print("   (native module not built — run: cd rust && maturin build --release)")
    except Exception as exc:
        print(f"   (skipped: {exc})")

    print("\nDone.")


if __name__ == "__main__":
    main()
