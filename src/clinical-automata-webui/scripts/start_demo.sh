#!/usr/bin/env bash
# Start the clinical-automata web demo servers on the current slurm allocation.
#
#   ./scripts/start_demo.sh          # auto-detect the running slurm job
#   ./scripts/start_demo.sh 78673    # or pin a specific jobid
#
# Starts the uvicorn server INSIDE the slurm allocation (so it can see the GPU):
#   :8899  main demo    (real claude CLI, real GPU, emails on)
#
# The cloudflared tunnels are SEPARATE and long-lived (they run on the login node).
# As long as the servers come back on the same node:port, the public URLs stay the same.
# Only re-run scripts/start_tunnels.sh if the tunnels themselves died (URLs change then).
#
# This script never touches the claude CLI, cloudflared, or any data on disk.
set -uo pipefail

WEBUI="$(cd "$(dirname "$0")/.." && pwd)"
SITE_CONFIG="$WEBUI/../config.env"   # src/config.env — see src/config.env.example
[ -f "$SITE_CONFIG" ] || { echo "!! $SITE_CONFIG not found — copy config.env.example and fill it in." >&2; exit 1; }
set -a; source "$SITE_CONFIG"; set +a
BACKEND=$WEBUI/backend

FORCE=0
VISIBILITY=public          # datasets flagged internal are hidden by default
ARGS=()
for a in "$@"; do
  case "$a" in
    --force)    FORCE=1 ;;
    --internal) VISIBILITY=internal ;;
    *)          ARGS+=("$a") ;;
  esac
done
[ "$VISIBILITY" = internal ] && echo "!! --internal: in-house datasets ARE exposed. Do not use for external users."

JOBID=${ARGS[0]:-$(squeue -u "$USER" -h -t R -o %i | head -1)}
if [ -z "$JOBID" ]; then
  echo "!! no running slurm job. Request one first (salloc/sbatch), then re-run." >&2
  exit 1
fi
NODE=$(squeue -j "$JOBID" -h -o %N | tr -d ' ')
echo "==> slurm job $JOBID on node $NODE"
# Record which allocation the demo is on, so gpu_autorenew.sh knows what to watch.
mkdir -p "$BACKEND/_work"
echo "$JOBID" > "$BACKEND/_work/demo_jobid"

# SAFETY: killing uvicorn tears down the whole agent tree with it
# (claude -> bash -> uv -> train.py), losing a clinician's run mid-training.
# A live `claude -p` process means a session is in flight — refuse to restart.
LIVE=$(srun --jobid "$JOBID" --overlap pgrep -u "$USER" -f "claude -p --input-format stream-json" 2>/dev/null | tr '\n' ' ')
if [ -n "$LIVE" ] && [ "$FORCE" -eq 0 ]; then
  echo
  echo "!! REFUSING TO RESTART — a session is live (claude pid:$LIVE)."
  srun --jobid "$JOBID" --overlap bash -c \
    'ps -u '"$USER"' -o etime,cmd --no-headers | grep "[t]rain.py"' 2>/dev/null \
    | sed 's/^/   training in progress: /'
  echo
  echo "   Restarting now would kill the agent AND its training run."
  echo "   Either wait for it to finish / end the session in the UI,"
  echo "   or re-run with --force if you accept losing it."
  exit 2
fi
[ -n "$LIVE" ] && echo "!! --force: killing the live session (claude pid:$LIVE)"

# Stop only OUR previous uvicorn servers. Never kills cloudflared.
srun --jobid "$JOBID" --overlap pkill -u "$USER" -f "uvicorn app.main:app --host 0.0.0.0" 2>/dev/null
sleep 2

mkdir -p "$BACKEND/_work"

# Logs are APPENDED, never truncated. `>` would clear the file the moment the new
# server starts while the old one is still shutting down — and since the old
# process keeps writing at its original offset, its last words (which email went
# out, why the session ended) land past a hole of NULs and read as lost.
for f in uvicorn_real; do
  printf '\n===== restart %s =====\n' "$(date -Is)" >> "$BACKEND/_work/$f.log"
done

echo "==> starting main demo   :8899"
setsid nohup srun --jobid "$JOBID" --overlap bash -c "
  cd $BACKEND && source .venv/bin/activate &&
  CAWEBUI_claude_mode=real CAWEBUI_gpu_mode=real CAWEBUI_permission_mode=bypassPermissions CAWEBUI_visibility=$VISIBILITY \
  exec uvicorn app.main:app --host 0.0.0.0 --port 8899
" >> "$BACKEND/_work/uvicorn_real.log" 2>&1 < /dev/null &

sleep 12
echo
for p in 8899; do
  if curl -sf "http://$NODE:$p/api/health" >/dev/null 2>&1; then
    echo "  [ok]   :$p up"
  else
    echo "  [FAIL] :$p — check $BACKEND/_work/uvicorn_real.log"
  fi
done

# Point the NAMED tunnel at whatever node this allocation landed on. The public
# hostname never changes; only the origin node behind it does. If the config
# already names $NODE we leave the tunnel alone; otherwise we rewrite it and let
# the keepalive watchdog respawn cloudflared with the new origin.
CFG=$CF_CONFIG
if [ -f "$CFG" ] && ! grep -q "http://$NODE:8899" "$CFG"; then
  echo
  echo "==> tunnel origin was not $NODE — repointing and bouncing the tunnel"
  sed -i -E "s|http://[a-zA-Z0-9._-]+:8899|http://$NODE:8899|g" "$CFG"
  TPID=$(pgrep -u "$USER" -o -f "cloudflared tunnel run $CF_TUNNEL_NAME")
  [ -n "$TPID" ] && kill "$TPID" 2>/dev/null   # keepalive respawns it with the new config within ~30s
fi

echo
echo "public URL (stable — does NOT change across restarts or node moves):"
echo "  main   : https://$DEMO_HOSTNAME   -> $NODE:8899"
