#!/usr/bin/env bash
# Build the frontend and serve the whole app (API + UI) on http://localhost:8000
set -e
cd "$(dirname "$0")/.."
if [ ! -d web/node_modules ]; then npm install --prefix web; fi
npm run build --prefix web
echo "Starting pitwall on http://localhost:8000 ..."
PYTHONPATH=src python3 -m uvicorn pitwall.server:app --host 127.0.0.1 --port 8000
