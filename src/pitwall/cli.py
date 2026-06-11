"""``pitwall`` command-line interface — the offline strategist's toolkit.

    pitwall circuits                       list known circuits
    pitwall calibrate 2023 Bahrain bahrain fit the tyre model on a real race
    pitwall optimize bahrain --grid 3      recommend a strategy
    pitwall simulate bahrain --strategy M-H --grid 5   run one ensemble
    pitwall live 2023 Bahrain              live Kalman tyre demo on real data
"""

from __future__ import annotations

import argparse
import sys

from .models import RaceModel
from .models.circuits import CIRCUITS
from .sim import ScenarioSet, build_field
from .types import Compound, Stint, Strategy

_CODE = {"S": Compound.SOFT, "M": Compound.MEDIUM, "H": Compound.HARD}


def _parse_strategy(spec: str, n_laps: int) -> Strategy:
    """'M-H' (auto pit laps) or 'M25-H' (explicit) -> Strategy."""
    parts = spec.upper().split("-")
    comps, lengths = [], []
    explicit = any(any(ch.isdigit() for ch in p) for p in parts)
    if explicit:
        cum = 0
        for p in parts:
            code = p[0]
            comps.append(_CODE[code])
            if p[1:]:
                lengths.append(int(p[1:]) - cum)
                cum = int(p[1:])
            else:
                lengths.append(n_laps - cum)
        return Strategy([Stint(c, l) for c, l in zip(comps, lengths)])
    comps = [_CODE[p[0]] for p in parts]
    n = len(comps)
    base = n_laps // n
    lengths = [base] * (n - 1) + [n_laps - base * (n - 1)]
    return Strategy([Stint(c, l) for c, l in zip(comps, lengths)])


def cmd_circuits(_args) -> None:
    print(f"{'id':16}{'name':18}{'laps':>5}{'pit':>6}{'overtake':>10}{'P(SC)':>7}")
    for cid, p in CIRCUITS.items():
        print(f"{cid:16}{p.name:18}{p.n_laps:>5}{p.pit_loss_s:>6.0f}"
              f"{p.overtake_threshold_s:>10.2f}{p.p_safety_car:>7.2f}")


def cmd_optimize(args) -> None:
    model = RaceModel.for_circuit(args.circuit)
    if args.calibrate:
        from .data.calibrate import calibrate_race
        cal = calibrate_race(args.calibrate[0], args.calibrate[1], args.circuit)
        if cal:
            model = cal.to_race_model(args.circuit)
            print(f"(calibrated on {args.calibrate[1]} {args.calibrate[0]}, RMSE {cal.rmse:.2f}s)")
    from .optimize import optimize
    rivals = build_field(args.circuit, n_cars=20, seed=1)
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, args.scenarios, seed=7)
    res = optimize(model, rivals, scen, circuit_id=args.circuit, focal_grid=args.grid,
                   focal_delta=args.delta, objective=args.objective, shortlist=args.shortlist)
    print(f"\nObjective: {args.objective}  |  grid P{args.grid}  |  {len(scen)} scenarios\n")
    print(f"{'strategy':26}{'mean':>6}{'P(win)':>8}{'P(pod)':>8}{'P(pts)':>8}{'CVaR10':>8}")
    for s in res.ranked[: args.top]:
        e = s.ensemble
        print(f"{s.strategy.label():26}{e.mean_position:>6.2f}{e.p_win:>8.2f}"
              f"{e.p_podium:>8.2f}{e.p_points:>8.2f}{e.cvar_position():>8.1f}")
    print(f"\n>>> RECOMMENDATION: {res.best.strategy.label()}")


def cmd_simulate(args) -> None:
    model = RaceModel.for_circuit(args.circuit)
    strat = _parse_strategy(args.strategy, model.config.n_laps)
    from .sim import evaluate, with_focal
    rivals = build_field(args.circuit, n_cars=20, seed=1)
    scen = ScenarioSet.sample(model.safety_car, model.config.n_laps, args.scenarios, seed=7)
    field = with_focal(rivals, strat, circuit_id=args.circuit, focal_grid=args.grid,
                       focal_delta=args.delta)
    ens = evaluate(field, scen)
    print(f"Strategy {strat.label()} from P{args.grid} at {model.config.name} "
          f"({len(scen)} races):")
    for k, v in ens.summary().items():
        print(f"   {k}: {v}")


def cmd_calibrate(args) -> None:
    from .data.calibrate import calibrate_race, calibrate_races
    if args.years:
        races = [(y, args.country) for y in args.years]
        cal = calibrate_races(races, args.circuit)
    else:
        cal = calibrate_race(args.year, args.country, args.circuit)
    if cal is None:
        print("Calibration failed (no data).")
        return
    import json
    print(json.dumps(cal.summary(), indent=2))
    if args.save:
        cal.save(args.save)
        print(f"saved -> {args.save}")


def cmd_backtest(args) -> None:
    from .data.backtest import reconstruct_race
    res = reconstruct_race(args.year, args.country, args.circuit,
                           n_scenarios=args.scenarios,
                           pool_years=args.pool_years)
    if res is None:
        print("Backtest failed (no data).")
        return
    print(f"Race reconstruction — {args.country} {args.year}")
    print(f"  Spearman rank corr vs real result: {res.spearman:.3f}")
    print(f"  mean |position error|: {res.mae_position:.2f}   podium accuracy: {res.podium_accuracy:.0%}")
    print(f"\n  {'pos':<4}{'predicted':<11}{'actual'}")
    for i, (p, a) in enumerate(zip(res.predicted_order[:12], res.actual_order[:12]), 1):
        mark = "  <-- match" if p == a else ""
        print(f"  P{i:<3}{p:<11}{a}{mark}")


def cmd_live(args) -> None:
    from .data.calibrate import build_lap_dataset
    from .data.openf1 import OpenF1Client
    from .live import KalmanTyreModel, advise
    client = OpenF1Client()
    sk = client.session_key(args.year, args.country, "Race")
    df = build_lap_dataset(client, sk)
    if df.empty:
        print("no data")
        return
    nl = int(df["lap"].max())
    df["k"] = df["driver"].astype(str) + "/" + df["compound"].astype(str)
    grp = df.groupby("k")
    key = max(grp.groups, key=lambda k: len(grp.get_group(k)))
    stint = grp.get_group(key).sort_values("lap")
    kf = KalmanTyreModel(n_laps=nl, prior_pace=float(stint["lap_time"].iloc[0]))
    print(f"Live tyre estimate on real stint {key} ({len(stint)} laps):")
    for _, r in stint.iterrows():
        u = kf.update(int(r["lap"]), int(r["tyre_age"]), float(r["lap_time"]))
        flag = "  <-- ANOMALY" if u.anomaly else ""
        lo, hi = kf.belief.credible_deg()
        print(f"  lap {u.lap:>2} age {u.tyre_age:>2}: deg={kf.belief.deg_rate:+.4f} "
              f"s/lap (90% {lo:+.3f}..{hi:+.3f}){flag}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pitwall", description="F1 race-strategy engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("circuits", help="list known circuits").set_defaults(func=cmd_circuits)

    o = sub.add_parser("optimize", help="recommend a strategy")
    o.add_argument("circuit")
    o.add_argument("--grid", type=int, default=1)
    o.add_argument("--delta", type=float, default=0.0)
    o.add_argument("--objective", default="podium", choices=["expected", "podium", "win", "points", "robust"])
    o.add_argument("--scenarios", type=int, default=400)
    o.add_argument("--shortlist", type=int, default=12)
    o.add_argument("--top", type=int, default=8)
    o.add_argument("--calibrate", nargs=2, metavar=("YEAR", "COUNTRY"), default=None)
    o.set_defaults(func=cmd_optimize)

    s = sub.add_parser("simulate", help="evaluate one strategy")
    s.add_argument("circuit")
    s.add_argument("--strategy", required=True, help="e.g. M-H or M25-H")
    s.add_argument("--grid", type=int, default=1)
    s.add_argument("--delta", type=float, default=0.0)
    s.add_argument("--scenarios", type=int, default=400)
    s.set_defaults(func=cmd_simulate)

    c = sub.add_parser("calibrate", help="fit the tyre model on real data")
    c.add_argument("year", type=int)
    c.add_argument("country")
    c.add_argument("circuit")
    c.add_argument("--years", type=int, nargs="+", help="pool multiple years")
    c.add_argument("--save")
    c.set_defaults(func=cmd_calibrate)

    bt = sub.add_parser("backtest", help="reconstruct a real race & score the sim vs reality")
    bt.add_argument("year", type=int)
    bt.add_argument("country")
    bt.add_argument("circuit")
    bt.add_argument("--scenarios", type=int, default=300)
    bt.add_argument("--pool-years", type=int, nargs="+", default=None)
    bt.set_defaults(func=cmd_backtest)

    lv = sub.add_parser("live", help="live Kalman tyre demo on a real race")
    lv.add_argument("year", type=int)
    lv.add_argument("country")
    lv.set_defaults(func=cmd_live)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
