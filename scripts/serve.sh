#!/usr/bin/env bash
# Install first using README; this launcher never installs dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
npm run build --prefix web
exec "${PITWALL_PYTHON:-.venv/bin/python}" -m uvicorn pitwall.server:app --host 127.0.0.1 --port "${PITWALL_PORT:-18743}"
