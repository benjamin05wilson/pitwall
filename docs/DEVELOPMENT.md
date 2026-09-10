# Development, advanced paths and packaging

Use the README's clone-based offline route first. `requirements-offline.txt` pins the tested Python dependency set; `web/package-lock.json` pins frontend and Playwright dependencies. Pin updates should rerun the core and browser suites. Dependencies are installed before offline execution, not by launch scripts. Source syntax permits Python 3.10, but the demonstration pins and CI target 3.11–3.12. Node 22 is configured in CI; Node 24 was used locally.

`npm test --prefix web` uses `.venv/bin/python` on POSIX unless `PITWALL_PYTHON` is set (use `python` for an activated Windows environment). It starts its own loopback backend on port 18743 and fails rather than reusing someone else's service. Run after `npm run build --prefix web`. Browser requests to external hosts are rejected. Videos and screenshots land in `web/test-results`; see the evidence directory for retained copies. `scripts/dev.sh` is a separate optional hot-reload route with API port 8000 and Vite port 5173, both loopback; it cleans its own child API.

The offline workflow runs on pull requests and pushes to main. It tests Python on Linux/Windows and builds/tests the frontend on Linux. Torch, native and public-data tests are separate opt-in paths, not prerequisites for core collection:

```bash
# Optional, installs a substantial ML dependency; no training implied:
python -m pip install -e '.[ml]'
python -m pytest tests/test_surrogate.py
# Optional public-data downloads, with time/network costs:
python -m pip install -e '.[data]'
PITWALL_NETWORK_TESTS=1 python -m pytest tests/test_data_network.py
```

Rust needs a separate toolchain and maturin. Build the crate under `rust/` into a repository-local environment, then run `tests/test_native.py`. That parity test covers a limited dry default configuration; it does not validate all calibrated settings. No local Rust run was possible in the readiness environment because cargo was unavailable.

Historical reconstruction and calibration CLI commands may download public race data. Their results must be described as retrospective. `scripts/build_surrogate.py` performs an expensive dataset-generation/training workflow and is not an installation or demo step. No checkpoint or real-race performance claim accompanies it.

`python -m build --wheel` and `scripts/check_wheel.py` verify that calibrated priors are packaged. The Python core wheel is portable; the web distribution, historical caches and surrogate weights remain outside the wheel. Web hosting is supported from an editable clone, not promised from a wheel alone. The priors resource resolves relative to the installed package. Priors regeneration writes into that resource location, so do it in an editable research checkout, not a read-only system installation.

`data/cache` and surrogate defaults also resolve from the checkout. These caches are not required by the offline demo. `web/tsconfig.tsbuildinfo`, local environments, caches, browser downloads and generated test results are ignored. The code's existing MIT licence is unchanged.

`python scripts/gen_interview_docs.py WORKFLOW.json` requires external interview-workflow output with `result.dimensions` (or root `dimensions`). That input is not bundled; the script overwrites modelling/interview documents. It does not reproduce experiments, and generated claims require manual evidence review. The older `INTERVIEW_QA.md` and `MODELLING.md` are historical design/interview notes, not an executed-results ledger.
