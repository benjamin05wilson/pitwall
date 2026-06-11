#!/usr/bin/env bash
# Dev mode: FastAPI on :8000, Vite dev server on :5173 (hot reload, proxies /api).
set -e
cd "$(dirname "$0")/.."
if [ ! -d web/node_modules ]; then npm install --prefix web; fi
PYTHONPATH=src python3 -m uvicorn pitwall.server:app --host 127.0.0.1 --port 8000 --reload &
npm run dev --prefix web
