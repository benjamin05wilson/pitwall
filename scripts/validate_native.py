"""Validate + benchmark the Rust accelerator against the Python reference.

Builds a Bahrain field (focal car on grid 3), samples a bank of ~2000 scenarios,
then runs BOTH the pure-Python ``evaluate`` and the Rust ``simulate_batch_native``
over the SAME scenarios (common random numbers at the scenario level). Reports the
focal finishing-position distributions side by side and the wall-clock speedup.

Bit-parity is NOT expected (Python uses NumPy's PCG64 stream, Rust uses its own
``rand_pcg::Pcg64`` with a different draw order). The bar is statistical
agreement: mean position within ~0.25 and P(podium) within ~0.05.

Run:  PYTHONPATH=src python3 scripts/validate_native.py
"""

from __future__ import annotations

import time

import numpy as np

from pitwall.models import RaceModel
from pitwall.sim import build_field, evaluate, nominal_strategy, with_focal
from pitwall.sim.monte_carlo import ScenarioSet
from pitwall.sim.native import HAS_NATIVE, simulate_batch_native

N_SCENARIOS = 2000
CIRCUIT = "bahrain"
FOCAL_GRID = 3
FOCAL_ID = 99


def hist(positions: np.ndarray, grid_size: int) -> np.ndarray:
    """Frequency of each finishing position 1..grid_size."""
    counts = np.zeros(grid_size, dtype=float)
    for p in range(1, grid_size + 1):
        counts[p - 1] = np.mean(positions == p)
    return counts


def main() -> int:
    if not HAS_NATIVE:
        print("FAIL: native extension (_native) not available; build it first:")
        print("  cd rust && maturin build --release && "
              "pip install --force-reinstall --no-deps target/wheels/*.whl")
        return 1

    base = RaceModel.for_circuit(CIRCUIT)
    laps = base.config.n_laps

    rivals = build_field(CIRCUIT)
    focal_strat = nominal_strategy(laps)
    field = with_focal(rivals, focal_strat, circuit_id=CIRCUIT, focal_grid=FOCAL_GRID)
    grid_size = len(field)

    scenarios = ScenarioSet.sample(base.safety_car, laps, N_SCENARIOS, seed=0)

    print(f"circuit={CIRCUIT}  laps={laps}  grid={grid_size}  "
          f"focal_grid={FOCAL_GRID}  scenarios={len(scenarios)}")
    print(f"focal strategy: {focal_strat.label()}")
    print()

    # ---- Python reference ------------------------------------------------- #
    t0 = time.perf_counter()
    py_res = evaluate(field, scenarios, focal_id=FOCAL_ID)
    py_secs = time.perf_counter() - t0
    py_pos = py_res.positions

    # ---- Rust accelerator ------------------------------------------------- #
    # warm-up (build/flatten cost) excluded from the timed run
    simulate_batch_native(field, scenarios, focal_id=FOCAL_ID)
    t0 = time.perf_counter()
    rs_pos = simulate_batch_native(field, scenarios, focal_id=FOCAL_ID)
    rs_secs = time.perf_counter() - t0

    # ---- compare ---------------------------------------------------------- #
    py_mean = float(np.mean(py_pos))
    rs_mean = float(np.mean(rs_pos))
    py_podium = float(np.mean(py_pos <= 3))
    rs_podium = float(np.mean(rs_pos <= 3))
    py_win = float(np.mean(py_pos <= 1))
    rs_win = float(np.mean(rs_pos <= 1))
    py_points = float(np.mean(py_pos <= 10))
    rs_points = float(np.mean(rs_pos <= 10))

    py_h = hist(py_pos, grid_size)
    rs_h = hist(rs_pos, grid_size)
    max_freq_diff = float(np.max(np.abs(py_h - rs_h)))

    print(f"{'metric':<18}{'Python':>12}{'Rust':>12}{'|diff|':>12}")
    print("-" * 54)
    print(f"{'mean position':<18}{py_mean:>12.4f}{rs_mean:>12.4f}{abs(py_mean - rs_mean):>12.4f}")
    print(f"{'P(win)':<18}{py_win:>12.4f}{rs_win:>12.4f}{abs(py_win - rs_win):>12.4f}")
    print(f"{'P(podium)':<18}{py_podium:>12.4f}{rs_podium:>12.4f}{abs(py_podium - rs_podium):>12.4f}")
    print(f"{'P(points)':<18}{py_points:>12.4f}{rs_points:>12.4f}{abs(py_points - rs_points):>12.4f}")
    print()

    print("position histogram (frequency):")
    print(f"{'pos':>4}{'Python':>12}{'Rust':>12}{'|diff|':>12}")
    for p in range(grid_size):
        print(f"{p + 1:>4}{py_h[p]:>12.4f}{rs_h[p]:>12.4f}{abs(py_h[p] - rs_h[p]):>12.4f}")
    print()
    print(f"max per-position frequency diff: {max_freq_diff:.4f}")
    print()

    speedup = py_secs / rs_secs if rs_secs > 0 else float("inf")
    print(f"Python wall-clock: {py_secs:.4f} s  ({1e6 * py_secs / len(scenarios):.1f} us/scenario)")
    print(f"Rust   wall-clock: {rs_secs:.4f} s  ({1e6 * rs_secs / len(scenarios):.1f} us/scenario)")
    print(f"SPEEDUP: {speedup:.1f}x")
    print()

    mean_ok = abs(py_mean - rs_mean) <= 0.25
    podium_ok = abs(py_podium - rs_podium) <= 0.05
    speed_ok = speedup >= 10.0

    print("VALIDATION:")
    print(f"  mean position within 0.25 : {'PASS' if mean_ok else 'FAIL'} "
          f"(diff={abs(py_mean - rs_mean):.4f})")
    print(f"  P(podium) within 0.05     : {'PASS' if podium_ok else 'FAIL'} "
          f"(diff={abs(py_podium - rs_podium):.4f})")
    print(f"  speedup >= 10x            : {'PASS' if speed_ok else 'FAIL'} "
          f"({speedup:.1f}x)")

    ok = mean_ok and podium_ok
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
