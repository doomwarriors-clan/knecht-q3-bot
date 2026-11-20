#!/bin/bash
# start_rl_env.sh -- RL training env on knecht (port 28134, home ~/.q3a_rl).
# Engine: rl-env-2026.11.20 = dwc tip 6bb9b17e (botctl) + sv_timescale (6f5c1edd),
# built USE_BOTCTL=1. FFA duel (g_gametype 1), sv_fps 40, sv_timescale 10.
# botctl socket: ~/.q3a_rl/botctl-28134.sock (0600) -- the laptop training
# process connects via SSH unix-socket forward. While sv_timescale > 1 the
# engine REJECTS all real (non-NA_BOT) clients: bots-only by design.
#
# MAP ROTATION (2026-11-20): 12-map list (the dl1 Duel list, all BSP+AAS
# verified on knecht). This engine build has NO map-list rotation in
# SV_Map_f; at each natural match end (timelimit 10 game-min = 60 s wall at
# TS 10) the OSP QVM executes the `nextmap` cvar as a command. Its default
# (reset on every spawn, sv_init.c) is "map_restart 0" = in-place restart;
# the feeder sets it to "map <m>" for a different-map change, which is a
# FULL SV_SpawnServer (FS_Startup, botctl: stopped/started) -> the env's
# sidecar connection EOFs (the env reconnects itself).
#
# Feeder design (drift-free, self-healing; each silent-failure bug found
# on 2026-11-20 is documented where it matters):
#   detection = the CURRENT map from the server itself (rcon status over
#               loopback UDP) -- NOT the stdout log. The log is block-
#               buffered (up to ~30 s flush lag); detection off the log
#               let a nextmap shot land AFTER the successor's spawn reset
#               (sv_init.c re-sets nextmap to "map_restart 0" on every
#               spawn) -> nextmap = current map -> one wasted same-map
#               full spawn per hit. rcon status is instant and exact; it
#               also accepts only maps from the known list (garbage /
#               "NO_REPLY" during FS_Startup -> empty -> no trigger).
#   successor = one past the OBSERVED map in the list, re-derived from
#               reality each cycle. No counter: a counter would point at
#               MAPS[0] on the very first detection (same map -> in-place
#               restart -> the trigger never fires again -> stuck forever),
#               and it would drift after any missed cycle.
#   shots     = 3 idempotent nextmap cvar sets at +0/+10/+20 s after
#               detection; each re-verifies the LIVE server map is still
#               the trigger map before firing (kills the late-shot race).
#               A shot is inert until the next natural end, so early is
#               safe; with instant detection all 3 land well inside the
#               60 s match.
#   watchdog  = if NO map change within 150 s of the last one, force
#               `map <successor>` (violent full spawn mid-match,
#               self-healing) -- covers the all-shots-missed case and the
#               pre-client-join spawn: the first client that joins a fresh
#               server triggers a join-spawn that resets nextmap, so the
#               first cycle always ends in-place and the watchdog starts
#               the rotation (~2.5 min after engine start).
#
# OPPONENT OWNERSHIP: the QVM opponent (dwcbot5/"NUDELpas") is owned by
# the ENV's _ensure_opponent guard (env_duel.py: snapshot-based presence,
# 10 s throttle, rcon addbot on knecht via rcon.py). The feeder must NOT
# addbot: rcon QVM bots SURVIVE same-process map changes (they auto-
# reconnect after a full spawn), so any re-add stacks DUPLICATES (seen
# 2026-11-20: 3 duplicates -> command-overflow storm + 4-way ganks that
# distort the 1on1 signal). Single owner = zero stacking risk; the guard
# re-adds after overflow kicks and cold engine restarts (verified).
#
# RCON PATH (found 2026-11-20, cost a silent no-op rotation): ALL rcon
# commands go over LOOPBACK UDP (rcon.py in this dir) -- NOT through the
# engine's stdin/console. The dedicated console's `rcon` command is the
# CLIENT command (CL_Rcon_f); in a dedicated build it runs but sends its
# packet to a bogus address: no "Unknown command", no "Rcon from" log
# line, no effect -- total silence. Plain console commands (exec, map)
# work fine via the FIFO; only rcon must use the UDP path. The quoted
# value MUST survive as literal quotes in the UDP payload (the server
# tokenizer then re-groups it): rcon.py nextmap "\"map <m>\"".
#
# Solo variant (q3dm17 only): start_rl_env.sh.solo (backup, 2026-11-20).
set -u
Q3=/home/hkuehnemund/q3e
ENGINE=/home/hkuehnemund/tmp/q3e_dwc/build/release-linux-x86_64/quake3e.ded.x64
HOME_RL=/home/hkuehnemund/.q3a_rl
PORT=28134
SOCK=$HOME_RL/botctl-$PORT.sock
LOGDIR=/home/hkuehnemund/tmp/rl
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR" "$HOME_RL/osp"

if ss -ltnu 2>/dev/null | awk "{print \$5}" | grep -qE ":$PORT\s*$"; then
  echo "port $PORT already in use -- abort (stop_rl_env.sh first)"; exit 1
fi
[ -x "$ENGINE" ] || { echo "engine missing: $ENGINE"; exit 1; }

# rotation list (dl1 Duel order); OPPBOT/OPPSKILL documented here only --
# the env guard (env_duel.py) adds the opponent, not the feeder.
MAPS=(pro-q3dm6 ztn3tourney1 pro-q3tourney4 q3tourney2 hub3tourney1 hub3aeroq3 cht3 q3tourney4 nodm9 pro-nodm9 cpm3 cpm1a)
N=${#MAPS[@]}
OPPBOT="dwcbot5"
OPPSKILL=5

# fresh gamedir home -- NEVER ~/.q3a or ~/.q3a_duel (user play data / botmem)
rm -f "$HOME_RL/osp/"*.cfg
cp "$Q3/baseq3/zzz_dwc_bots.pk3" "$HOME_RL/osp/"

cat > "$HOME_RL/osp/rl.cfg" <<RLCFG
rconPassword testrcon123
g_gametype 1
g_friendlyFire 1
g_doWarmup 0
g_spskill 3
g_maxGameClients 0
timelimit 10
fraglimit 0
capturelimit 8
g_teamAutoJoin 0
g_teamForceBalance 0
bot_enable 1
g_bots 1
bot_autoload "[none]"
bot_memory 0
sv_maxclients 16
RLCFG

cd "$Q3" || exit 1
FIFO=$LOGDIR/stdin-$TS.fifo
rm -f "$FIFO"; mkfifo "$FIFO"
LOG=$LOGDIR/server_$TS.log

(
  sleep 3
  printf "exec rl\n"
  sleep 1
  printf "map %s\n" "${MAPS[0]}"
  # --- rotation driver (see header) -------------------------------------
  lastmap=""
  last_change_ts=0
  now() { date +%s; }
  # rcon via loopback UDP (see the RCON PATH note in the header)
  rcon() { python3 "$LOGDIR/rcon.py" "$@" >/dev/null 2>&1 || true; }
  # CURRENT map from the server itself (instant, exact); empty when the
  # server is silent (FS_Startup) or answers with a map not in the list.
  curmap() {
    local m
    m=$(python3 "$LOGDIR/rcon.py" status 2>/dev/null | grep -a "^map:" | head -1 | awk '{print $2}')
    case " ${MAPS[*]} " in
      *" $m "*) printf '%s' "$m" ;;
    esac
  }
  succ() {  # successor of $1 in the list; fallback MAPS[0] when unknown
    local k
    for k in "${!MAPS[@]}"; do
      if [ "${MAPS[$k]}" = "$1" ]; then
        printf '%s' "${MAPS[$(( (k + 1) % N ))]}"
        return 0
      fi
    done
    printf '%s' "${MAPS[0]}"
  }
  while :; do
    live=$(curmap)
    if [ -n "$live" ] && [ "$live" != "$lastmap" ]; then
      lastmap="$live"
      last_change_ts=$(now)
      next=$(succ "$live")
      echo "[$(date +%H:%M:%S)] rotation: live=$live -> next=$next" >&2
      # 3 idempotent shots; each re-verifies the LIVE server map (a shot
      # processed after the successor's spawn would set nextmap to the
      # now-current map -> one wasted same-map full spawn)
      k=0
      while [ "$k" -lt 3 ]; do
        if [ "$(curmap)" = "$live" ]; then
          rcon nextmap "\"map $next\""
        fi
        k=$((k + 1))
        sleep 10
      done
      sleep 30
    else
      # watchdog: no map change for 150 s -> force the successor (normal
      # cycles are ~65-85 s apart, so this only fires on a missed cycle
      # or the pre-client-join spawn that resets nextmap)
      if [ "$last_change_ts" -gt 0 ] && [ $(( $(now) - last_change_ts )) -gt 150 ]; then
        last_change_ts=$(now)
        next=$(succ "$lastmap")
        echo "[$(date +%H:%M:%S)] watchdog: no rotation for 150 s -> force map $next" >&2
        rcon map "$next"
        sleep 20
      fi
    fi
    sleep 5
  done
) > "$FIFO" &
FEEDER_PID=$!

nohup "$ENGINE" \
   +set fs_game osp \
   +set dedicated 1 \
   +set net_port "$PORT" \
   +set com_hunkMegs 256 \
   +set fs_basepath "$Q3" \
   +set fs_homepath "$HOME_RL" \
   +set sv_pure 0 \
   +set sv_fps 40 \
   +set sv_timescale 10 \
   +set botctl_enable 1 \
   < "$FIFO" > "$LOG" 2>&1 &
SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID FEEDER_PID=$FEEDER_PID LOG=$LOG"
echo "$SERVER_PID" > "$LOGDIR/server.pid"
echo "$FEEDER_PID" > "$LOGDIR/feeder.pid"

for i in $(seq 1 60); do
  [ -S "$SOCK" ] && break
  sleep 1
done
if [ ! -S "$SOCK" ]; then
  echo "socket $SOCK never appeared -- server log tail:"
  tail -25 "$LOG"; exit 1
fi
sleep 12
echo "== socket up; server log tail:"
tail -18 "$LOG"
echo START_OK
