#!/usr/bin/env bash
# Keepalive watchdog for the named cloudflare tunnel.
#
# The tunnel is a single long-lived process on the LOGIN node. If it dies
# (login-node reboot, admin cleanup of stray processes, a crash) the public URL
# goes dark (Cloudflare 1033) until someone notices by eye. This watchdog makes
# it self-heal: it checks every CHECK_INTERVAL seconds and respawns cloudflared
# if it is not running.
#
# Designed to be BOTH:
#   * run once in the foreground/background as a long-lived loop, AND
#   * safe to invoke repeatedly from cron (a flock guard means only one loop
#     ever runs, so `* * * * *` cron lines cannot stack up duplicates).
#
# It never touches slurm, uvicorn, the GPU, or any data. It only ensures one
# `cloudflared tunnel run $CF_TUNNEL_NAME` is alive.
set -uo pipefail

WEBUI="$(cd "$(dirname "$0")/.." && pwd)"
SITE_CONFIG="$WEBUI/../config.env"   # src/config.env — see src/config.env.example
[ -f "$SITE_CONFIG" ] || { echo "!! $SITE_CONFIG not found — copy config.env.example and fill it in." >&2; exit 1; }
set -a; source "$SITE_CONFIG"; set +a
CF=$WEBUI/backend/_work/cloudflared
LOG=$WEBUI/backend/_work/cf_named.log
WATCHLOG=$WEBUI/backend/_work/cf_keepalive.log
LOCK=$WEBUI/backend/_work/cf_keepalive.lock
TUNNEL=$CF_TUNNEL_NAME
CHECK_INTERVAL=30
MAX_LOG_BYTES=$((5 * 1024 * 1024))   # rotate cf_named.log past 5 MB so it can't grow forever

ts() { date -u '+%F %T UTC'; }

# Reconcile to EXACTLY one tunnel. Returns 0 if one already runs, 1 if none
# (caller should start one). If several are somehow running (a double-spawn
# race), keep the OLDEST and kill the rest — Cloudflare reroutes to the
# surviving replica, so this is seamless. This makes the watchdog converge to a
# single tunnel instead of merely guaranteeing "at least one".
reconcile_tunnels() {
  local pids
  pids=$(pgrep -u "$USER" -f "cloudflared tunnel run $TUNNEL" 2>/dev/null)
  [ -z "$pids" ] && return 1
  local oldest
  oldest=$(pgrep -u "$USER" -o -f "cloudflared tunnel run $TUNNEL" 2>/dev/null)
  for p in $pids; do
    if [ "$p" != "$oldest" ]; then
      kill "$p" 2>/dev/null
      echo "$(ts)  killed duplicate tunnel pid $p (kept $oldest)" >> "$WATCHLOG"
    fi
  done
  return 0
}

rotate_log() {
  # keep ONE previous log so a death is still diagnosable, but bounded in size
  if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt "$MAX_LOG_BYTES" ]; then
    mv -f "$LOG" "$LOG.1"
  fi
}

start_tunnel() {
  rotate_log
  # APPEND (>>), never truncate — the previous run's death reason is preserved.
  echo "=== [keepalive] (re)starting tunnel at $(ts) ===" >> "$LOG"
  setsid nohup "$CF" tunnel run "$TUNNEL" >> "$LOG" 2>&1 < /dev/null &
  echo "$(ts)  respawned cloudflared (pid $!)" >> "$WATCHLOG"
}

# --- single-instance guard: only one watchdog loop may run at a time ---------
exec 9>"$LOCK"
if ! flock -n 9; then
  # another keepalive loop already holds the lock — nothing to do
  exit 0
fi

echo "$(ts)  keepalive loop started (pid $$)" >> "$WATCHLOG"
while true; do
  if ! reconcile_tunnels; then
    echo "$(ts)  tunnel DOWN — respawning" >> "$WATCHLOG"
    start_tunnel
  fi
  sleep "$CHECK_INTERVAL"
done
