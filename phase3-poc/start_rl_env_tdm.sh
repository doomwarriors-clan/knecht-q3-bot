#!/bin/bash
# start_rl_env_tdm.sh -- RL TDM training env on knecht (port 28135, home ~/.q3a_rl_tdm).
# Engine: rl-env-2026.11.20 = dwc tip 6bb9b17e (botctl) + sv_timescale (6f5c1edd),
# built USE_BOTCTL=1. TDM 2v2 (g_gametype 3), sv_fps 40, sv_timescale 10.
# botctl socket: ~/.q3a_rl_tdm/botctl-28135.sock (0600). While sv_timescale > 1
# the engine REJECTS all real (non-NA_BOT) clients: bots-only by design.
#
# TDM (2026-11-20): 8-map rotation = the clan TDM map list (user constraint:
# TDM test matches run ONLY on these maps), all BSP+AAS verified on knecht
# (q3dm14tmp's AAS is NOT in any pak -- it lives in the engine's extracted
# home cache ~/.q3a/baseq3/q3dm14tmp.2dcfd82c.pk3, which this script copies
# into the fresh home's baseq3). Same feeder design as the duel env
# (start_rl_env.sh): live-map detection via rcon status over loopback UDP
# (NOT the block-buffered stdout log -- a log-based detection once let a
# late nextmap shot land after the successor's spawn reset), successor
# derived from the OBSERVED map (no counter -> no first-cycle same-map
# deadlock, no drift), 3 idempotent nextmap shots (each re-verifies the
# live server map), 150 s watchdog. All rcon over loopback UDP: the
# dedicated console's `rcon` is the CLIENT command (CL_Rcon_f) and is a
# total no-op (see the duel script header for the full write-up).
#
# OPPONENT OWNERSHIP: env_tdm.py's guard owns ALL 3 QVM bots (1 teammate
# red + 2 opponents blue): per-bot presence from rcon status funnames
# (the snapshot carries no names), 15 s per-bot absence grace, re-add
# ONLY the missing bots. QVM bots survive same-process map changes
# (auto-reconnect) -- unconditional re-adds stack duplicates. addbot
# carries the team as 3rd argument here (`addbot <file> <skill> <team>`,
# the 4v4 harness on 28131 runs that way); the AGENT's team is assigned
# by the env via the client TEXT command (`team red`) -- the botctl ADD
# team field is userinfo metadata only (botctl.c). The feeder does NO
# addbot.
set -u
Q3=/home/hkuehnemund/q3e
ENGINE=/home/hkuehnemund/tmp/q3e_dwc/build/release-linux-x86_64/quake3e.ded.x64
HOME_RL=/home/hkuehnemund/.q3a_rl_tdm
PORT=28135
SOCK=$HOME_RL/botctl-$PORT.sock
LOGDIR=/home/hkuehnemund/tmp/rl
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR" "$HOME_RL/osp" "$HOME_RL/baseq3"

if ss -ltnu 2>/dev/null | awk "{print \$5}" | grep -qE ":$PORT\s*$"; then
  echo "port $PORT already in use -- abort (stop_rl_env_tdm.sh first)"; exit 1
fi
[ -x "$ENGINE" ] || { echo "engine missing: $ENGINE"; exit 1; }

# rotation list (clan TDM maps, user constraint)
MAPS=(pro-q3dm6 q3dm7 ospdm5 ospdm6 cpm4 q3dm14tmp vq2dm1 ospdm15rc3)
N=${#MAPS[@]}

# q3dm14tmp AAS: extracted-cache pak (no other copy on this machine)
cp -f /home/hkuehnemund/.q3a/baseq3/q3dm14tmp.2dcfd82c.pk3 "$HOME_RL/baseq3/" 2>/dev/null \
  || echo "WARNING: q3dm14tmp AAS cache missing -- that map will fail to load" >&2

# fresh gamedir home -- NEVER ~/.q3a or ~/.q3a_duel (user play data / botmem)
rm -f "$HOME_RL/osp/"*.cfg
cp "$Q3/baseq3/zzz_dwc_bots.pk3" "$HOME_RL/osp/"

cat > "$HOME_RL/osp/rl_tdm.cfg" <<RLCFG
rconPassword testrcon123
g_gametype 3
g_friendlyFire 1
g_doWarmup 0
g_spskill 3
g_maxGameClients 0
timelimit 10
fraglimit 0
g_teamAutoJoin 0
g_teamForceBalance 0
bot_enable 1
g_bots 1
bot_autoload "[none]"
bot_memory 0
sv_maxclients 16
RLCFG

cd "$Q3" || exit 1
FIFO=$LOGDIR/stdin_tdm-$TS.fifo
rm -f "$FIFO"; mkfifo "$FIFO"
LOG=$LOGDIR/server_tdm_$TS.log

(
  sleep 3
  printf "exec rl_tdm\n"
  sleep 1
  printf "map %s\n" "${MAPS[0]}"
  # --- rotation driver (same design as start_rl_env.sh, port 28135) ----
  lastmap=""
  last_change_ts=0
  now() { date +%s; }
  # rcon via loopback UDP on the TDM port (see header)
  rcon() { python3 "$LOGDIR/rcon.py" $PORT "$@" >/dev/null 2>&1 || true; }
  # CURRENT map from the server itself (instant, exact); empty when the
  # server is silent (FS_Startup) or answers with a map not in the list.
  curmap() {
    local m
    m=$(python3 "$LOGDIR/rcon.py" $PORT status 2>/dev/null | grep -a "^map:" | head -1 | awk '{print $2}')
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
      # 3 idempotent shots; each re-verifies the LIVE server map
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
      # watchdog: no map change for 150 s -> force the successor (covers
      # all-shots-missed and the pre-client-join spawn that resets nextmap)
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
echo "$SERVER_PID" > "$LOGDIR/server_tdm.pid"
echo "$FEEDER_PID" > "$LOGDIR/feeder_tdm.pid"

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
