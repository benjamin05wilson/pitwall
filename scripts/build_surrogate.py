"""Build the surrogate end-to-end: parallel data generation -> train -> evaluate.

Usage: python scripts/build_surrogate.py [n_samples] [n_scenarios]
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitwall.surrogate import generate_dataset, save, train  # noqa: E402
from pitwall.surrogate.train import DEFAULT_PATH  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "surrogate" / "dataset.npz"


def _worker(args):
    n_samples, n_scenarios, spc, seed = args
    return generate_dataset(n_samples, n_scenarios=n_scenarios,
                            strategies_per_context=spc, seed=seed)


def generate_parallel(n_samples: int, n_scenarios: int, n_jobs: int = 6, spc: int = 10):
    per = max(spc, n_samples // n_jobs)
    jobs = [(per, n_scenarios, spc, 1000 + i) for i in range(n_jobs)]
    Xs, Ys, labels = [], [], None
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for X, Y, labels in ex.map(_worker, jobs):
            Xs.append(X)
            Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys), labels


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    n_scenarios = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    t0 = time.time()
    print(f"Generating ~{n_samples} samples x {n_scenarios} scenarios (parallel)...")
    X, Y, labels = generate_parallel(n_samples, n_scenarios)
    print(f"  generated {len(X)} samples in {time.time()-t0:.0f}s  | feature dim {X.shape[1]}")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_PATH, X=X, Y=Y, labels=labels)

    print("Training surrogate...")
    net, metrics = train(X, Y, epochs=150)
    save(net, DEFAULT_PATH)
    print("\n=== SURROGATE METRICS (held-out validation) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nsaved model -> {DEFAULT_PATH}")
    print(f"total build time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
