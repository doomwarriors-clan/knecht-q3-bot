#!/bin/bash
set -u
LOGDIR=/home/hkuehnemund/tmp/rl
for f in server_tdm.pid feeder_tdm.pid; do
  [ -f "$LOGDIR/$f" ] || continue
  pid=$(cat "$LOGDIR/$f")
  kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
  rm -f "$LOGDIR/$f"
done
sleep 6
pgrep -f "net_port 28135" >/dev/null && pkill -9 -f "net_port 28135"
rm -f $LOGDIR/stdin_tdm-*.fifo
echo STOPPED
