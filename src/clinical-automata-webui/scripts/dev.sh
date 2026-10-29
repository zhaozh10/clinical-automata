#!/usr/bin/env bash
# Dev launcher: backend (uvicorn, mock claude) + frontend (vite).
# Usage: scripts/dev.sh            # mock claude, no API cost
#        CLAUDE_MODE=real scripts/dev.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"

CLAUDE_MODE="${CLAUDE_MODE:-mock}"
ACCESS_CODE="${ACCESS_CODE:-demo-access}"

cd "$here/backend"
[ -d .venv ] || uv venv --python 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -q fastapi "uvicorn[standard]" pydantic pydantic-settings pyyaml itsdangerous python-multipart websockets

CAWEBUI_claude_mode="$CLAUDE_MODE" CAWEBUI_access_code="$ACCESS_CODE" \
  uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACK=$!
trap 'kill $BACK 2>/dev/null || true' EXIT

cd "$here/frontend"
[ -d node_modules ] || npm install
echo "backend :8000 (claude_mode=$CLAUDE_MODE)  |  frontend dev server below"
npm run dev
