#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PITWALL_PYTHON:-.venv/bin/python}" -m uvicorn pitwall.server:app --host 127.0.0.1 --port 8000 --reload &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true; wait "$api_pid" 2>/dev/null || true' EXIT INT TERM
npm run dev --prefix web -- --host 127.0.0.1 --strictPort
