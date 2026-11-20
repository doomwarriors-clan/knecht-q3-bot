"""
diag_env.py -- 60 s diagnostic session on the live RL env.

Prints per-2s: wall t, snap.time, my (health, pm_type), enemy health,
restart_id -- plus summary of engine frame pace (snap.time deltas) so we
can see the real tick rate, game speed, and when/why our client is absent.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from botctl import Botctl

sock = os.path.expanduser("~/tmp/rl-lab/botctl-28134.sock")
bc = Botctl(sock)
print("ping:", bc.ping())
slot = int(bc.add("diagbot"))
print("slot:", slot)
bc.cmd(slot, 0, 0, 0, 0, 2, 0.0, 0.0)

t0 = time.time()
last_time = None
dtimes = []
dtwalls = []
missing = 0
restart_changes = []
last_restart = None
last_print = 0.0
t_end = t0 + 60.0
while time.time() < t_end:
    snap = bc.snap()
    now = time.time()
    if last_time is not None:
        dtimes.append(snap.time - last_time)
        dtwalls.append(now - last_wall)
    last_time = snap.time
    last_wall = now
    if snap.restart_id != last_restart:
        restart_changes.append((round(now - t0, 1), snap.restart_id))
        last_restart = snap.restart_id
    me = snap.players.get(slot)
    if me is None:
        missing += 1
    if now - last_print >= 2.0:
        last_print = now
        if me is None:
            print(f"t={now-t0:5.1f} snap.t={snap.time} ME=MISSING restart={snap.restart_id}")
        else:
            ehp = [p["health"] for n, p in snap.players.items() if n != slot]
            print(f"t={now-t0:5.1f} snap.t={snap.time} me=({me['health']},pm{me['pm_type']}) "
                  f"enemy_hp={ehp} restart={snap.restart_id}")
bc.remove(slot)
bc.close()

import statistics as st
dtimes = [d for d in dtimes if d > 0]
print(f"\nframes sampled: {len(dtimes)}")
print(f"snap.time delta: median={st.median(dtimes)} ms, p90={sorted(dtimes)[int(len(dtimes)*0.9)]}, max={max(dtimes)}")
print(f"wall delta: median={st.median(dtwalls)*1000:.1f} ms, mean={st.mean(dtwalls)*1000:.1f} ms")
if dtimes:
    game_speed = st.median(dtimes) / (st.median(dtwalls) * 1000.0)
    print(f"game speed (median dt_game/dt_wall): {game_speed:.2f}x realtime")
    print(f"effective env step rate: {1.0/st.median(dtwalls):.0f} steps/s")
print(f"frames with ME missing: {missing}")
print(f"restart_id changes: {restart_changes}")
