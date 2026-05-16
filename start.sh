#!/usr/bin/env bash
# Boot the backend (port 8000) + frontend (port 3000) for local dev.
# Run from the repo root: `./start.sh`. Ctrl-C stops both.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

cd "$ROOT/backend"
if [ ! -d venv ]; then
  echo "backend/venv missing — run ./setup.sh first"
  exit 1
fi
./venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "frontend/node_modules missing — running npm install..."
  npm install
fi
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
