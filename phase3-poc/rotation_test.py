"""
rotation_test.py -- verify the 12-map rotation (no engine kill).

The engine rotates maps on natural match ends (timelimit 10 game-min =
60 s wall at TS 10); every different-map change is a FULL spawn that
EOFs the botctl sidecar connection. This test keeps one env session
live for 250 s (>= 2 map transitions expected), steps through them,
and requires: 0 errors, steps keeping flow, and -- checked from the
knecht side in the main script -- >= 2 distinct maps in the server log
and both clients present at the end.
"""
import subprocess
import time

from env_duel import DuelEnv

SOCK = "/home/hkuehnemund/tmp/rl-lab/botctl-28134.sock"


def main():
    e = DuelEnv(SOCK, "rlrot")
    obs = e.reset()
    t0 = time.time()
    print("started, slot %d" % e.slot, flush=True)

    steps = dones = errors = 0
    while time.time() - t0 < 250:
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
            print("t=%.0f (slow step %.0f s -- full spawn / reconnect)"
                  % (time.time() - t0, dt), flush=True)

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
    ok = errors == 0 and steps > 500 and opp_back
    print("ROTATION_OK" if ok else "ROTATION_FAIL", flush=True)
    e.close()


if __name__ == "__main__":
    main()
