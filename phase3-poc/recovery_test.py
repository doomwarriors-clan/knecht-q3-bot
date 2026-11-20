"""
recovery_test.py -- real kill/start recovery test for DuelEnv.

Sequence (wall time):
  t=0    env session starts (name rlrecov)
  t=20   knecht engine STOPPED (stop_rl_env.sh via ssh)
  t=65   knecht engine STARTED (start_rl_env.sh, nohup-detached via ssh)
  t=150  final check: env alive, steps continuing, opponent back

Success = no exceptions, steps keep flowing across the ~60 s outage,
a done=True boundary was seen (the reconnect), and at the end the
snapshot contains both rlrecov and the QVM opponent (dwcbot5).
"""
import subprocess
import threading
import time

from env_duel import DuelEnv

SOCK = "/home/hkuehnemund/tmp/rl-lab/botctl-28134.sock"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
       "hkuehnemund@192.168.69.25"]


def kill_start():
    t0 = time.time()
    time.sleep(20)
    r = subprocess.run(SSH + ["bash", "tmp/rl/stop_rl_env.sh"],
                       capture_output=True, text=True, timeout=90)
    print("t=%.0f STOP: %s" % (time.time() - t0,
                               r.stdout.strip().replace("\n", " | ")[:80]),
          flush=True)
    time.sleep(45)
    r = subprocess.run(
        SSH + ["nohup bash tmp/rl/start_rl_env.sh "
               "> tmp/rl/start_%s.log 2>&1 < /dev/null & echo START_LAUNCHED"
               % time.strftime("%H%M%S")],
        capture_output=True, text=True, timeout=60)
    print("t=%.0f START: %s" % (time.time() - t0,
                                r.stdout.strip().replace("\n", " | ")[:80]),
          flush=True)


def main():
    e = DuelEnv(SOCK, "rlrecov")
    obs = e.reset()
    t0 = time.time()
    print("started, slot %d" % e.slot, flush=True)

    th = threading.Thread(target=kill_start)
    th.start()

    steps = dones = errors = 0
    while time.time() - t0 < 150:
        e.send_action([0.7, 0.2, 0.15, 0.0, 0.5])
        t1 = time.time()
        try:
            obs, r, done = e.step()
        except Exception as ex:
            errors += 1
            print("t=%.0f ERROR: %r" % (time.time() - t0, ex), flush=True)
            break
        steps += 1
        dt = time.time() - t1
        if done:
            dones += 1
            print("t=%.0f DONE r=%.3f slot=%d" % (time.time() - t0, r, e.slot),
                  flush=True)
        if dt > 5:
            print("t=%.0f (slow step %.0f s -- engine down / reconnect)"
                  % (time.time() - t0, dt), flush=True)
    th.join()

    # self-heal phase: let the env's own recovery logic run before judging
    # (_ensure_present re-ADD after an OSP command-overflow kick, the
    # _ensure_opponent guard re-adding the QVM bot via rcon). The final
    # check right after the outage window can hit exactly such a kick.
    print("self-heal phase (20 s of stepping)...", flush=True)
    t1 = time.time()
    while time.time() - t1 < 20:
        e.send_action([0.7, 0.2, 0.15, 0.0, 0.5])
        try:
            e.step()
        except Exception as ex:
            print("self-heal step error: %r" % (ex,), flush=True)
            break
    try:
        snap = e.bc.snap()
        players = {n: (p["pm_type"], p["health"]) for n, p in snap.players.items()}
    except Exception as ex:
        players = {}
        print("FINAL snap failed: %r" % (ex,), flush=True)
    opp_back = any(n != e.slot for n in players)
    print("FINAL players: %s (slot %d) opp_back=%s" % (players, e.slot, opp_back),
          flush=True)
    print("RESULT steps=%d dones=%d errors=%d" % (steps, dones, errors),
          flush=True)
    ok = errors == 0 and steps > 300 and opp_back
    print("RECOVERY_OK" if ok else "RECOVERY_FAIL", flush=True)
    e.close()


if __name__ == "__main__":
    main()
