#!/usr/bin/env bash
# Start the NAMED cloudflare tunnel (stable URL = DEMO_HOSTNAME in src/config.env).
#
#   ./scripts/start_tunnel.sh
#
# Runs on the LOGIN node — deliberately outside any slurm allocation, so it
# survives allocation expiry, node changes and redeploys of the demo servers.
#
# The hostnames are bound to DNS records, not to this process, so restarting
# this tunnel does NOT change the URL (that was the quick-tunnel problem):
#   https://$DEMO_HOSTNAME    -> <compute-node>:8899
#
# Tunnel config lives in $CF_CONFIG. If the demo ever moves to another node,
# update the `service:` lines there and restart this (start_demo.sh does it).
set -uo pipefail

WEBUI="$(cd "$(dirname "$0")/.." && pwd)"
SITE_CONFIG="$WEBUI/../config.env"   # src/config.env — see src/config.env.example
[ -f "$SITE_CONFIG" ] || { echo "!! $SITE_CONFIG not found — copy config.env.example and fill it in." >&2; exit 1; }
set -a; source "$SITE_CONFIG"; set +a
CF=$WEBUI/backend/_work/cloudflared
LOG=$WEBUI/backend/_work/cf_named.log

if pgrep -u "$USER" -f "cloudflared tunnel .*run" >/dev/null 2>&1; then
  echo "==> named tunnel already running:"
  pgrep -u "$USER" -af "cloudflared tunnel .*run" | sed 's/^/    /'
  exit 0
fi

echo "==> starting named tunnel (login node, detached)"
# APPEND, never truncate — a previous run's death reason must survive a restart.
echo "=== [start_tunnel] starting at $(date -u '+%F %T UTC') ===" >> "$LOG"
setsid nohup "$CF" tunnel run "$CF_TUNNEL_NAME" >> "$LOG" 2>&1 < /dev/null &
sleep 8

if pgrep -u "$USER" -f "cloudflared tunnel .*run" >/dev/null 2>&1; then
  grep -c "Registered tunnel connection" "$LOG" | sed 's/^/    edge connections: /'
  echo
  echo "  main   : https://$DEMO_HOSTNAME"
else
  echo "  [FAIL] tunnel did not start — see $LOG" >&2
  exit 1
fi
