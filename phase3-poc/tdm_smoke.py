"""
tdm_smoke.py -- TDM env smoke test (~4 min, 28135 via the SSH-forwarded
local socket). Verifies:
  - the 3 QVM bots join their teams (guard re-add, ~20-30 s after start)
  - the agent's team stays RED across full-spawn map changes
    (TEXT-command re-assert; a full spawn brings it back FREE)
  - 0 errors across >= 2 rotation cycles
  - no STALE SNAPSHOTS: the report's snap.time must advance between
    15 s-apart reports. A full spawn freezes snap.time for <= ~20 s
    (FS_Startup + QVM recompile), so ONE no-advance is a warning and
    THREE consecutive no-advances (>= 45 s) are a hard fail -- the
    2026-11-20 stall froze it for 100+ s on a live connection.
  - no DONE storm: >= 3 episode boundaries in 50 s is a warning
    (healthy rotation = one DONE per ~65-85 s; the stall produced one
    every ~15 s).
Team tolerance: a SINGLE TEAM_FREE report is a legit spawn window
(the client re-enters FREE for ~1-2 frames before the immediate
re-assert lands); THREE+ consecutive FREE reports (the re-assert never
landed) or any BLUE report are hard fails. SPECTATOR (3) reports are
legit while the agent waits for the next post-spawn join window (the
env boots mid-match; see the env_tdm module docstring) -- but the
agent MUST end up on RED within the run (joined), or the smoke fails.
The actual map change is verified separately from the knecht server log
(>= 2 distinct mapnames in InitGame lines + feeder nextmap entries).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_tdm import TDMEnv, TEAM_RED, TEAM_FREE, TEAM_BLUE  # noqa: E402

SOCK = "/home/hkuehnemund/tmp/rl-lab/botctl-28135.sock"
DUR = 240  # s: >= 2 rotation cycles (~65-85 s) + bot-join time


def main():
    env = TDMEnv(SOCK, name="rltdm", team="red")
    t0 = time.time()
    steps = 0
    errors = 0
    last_report = 0.0
    team_hist = []
    saw_full_compo = False
    # stale-snapshot canary: snap.time of the previous 15 s report
    last_rep_time = None
    stale_strikes = 0
    stale_fail = False
    # DONE-storm detector: episode-boundary wall timestamps
    done_times = []
    done_storm = False
    try:
        while time.time() - t0 < DUR:
            try:
                o, r, done = env.step()
                steps += 1
                if o.shape != (54,):
                    errors += 1
                    print("t=%3.0f BAD OBS SHAPE %s" % (time.time() - t0, o.shape))
                if done:
                    now = time.time()
                    done_times = [t for t in done_times if now - t <= 50.0]
                    done_times.append(now)
                    if len(done_times) >= 3:
                        done_storm = True
                        print("t=%3.0f DONE-STORM WARNING: %d boundaries "
                              "in 50 s" % (now - t0, len(done_times)))
                    print("t=%3.0f DONE (episode boundary) r=%.3f" % (now - t0, r))
            except Exception as e:
                errors += 1
                print("t=%3.0f ERROR: %r" % (time.time() - t0, e))
                time.sleep(5)
            now = time.time()
            if now - last_report < 15.0:
                continue
            last_report = now
            try:
                snap = env.bc.snap()
                counts = {TEAM_FREE: 0, TEAM_RED: 0, TEAM_BLUE: 0}
                for p in snap.players.values():
                    counts[p["team"]] = counts.get(p["team"], 0) + 1
                me = env._me(snap)
                myteam = me["team"] if me is not None else -1
                team_hist.append(myteam)
                if counts[TEAM_RED] >= 2 and counts[TEAM_BLUE] >= 2 \
                        and myteam == TEAM_RED:
                    saw_full_compo = True
                # stale canary (see module docstring): snap.time must
                # advance between reports; one no-advance can be a full
                # spawn (<= ~20 s freeze), three is a dead stream
                if last_rep_time is not None:
                    if snap.time <= last_rep_time:
                        stale_strikes += 1
                        if stale_strikes == 1:
                            print("t=%3.0f STALE-SNAP watch: snap.time frozen "
                                  "at %d (1st, may be a full spawn)"
                                  % (now - t0, snap.time))
                        else:
                            print("t=%3.0f STALE-SNAP: snap.time frozen at %d "
                                  "across %d reports" % (now - t0, snap.time,
                                                         stale_strikes + 1))
                            if stale_strikes >= 3:
                                stale_fail = True
                    else:
                        stale_strikes = 0
                last_rep_time = snap.time
                # alive-only + sanity-bounded, same as the env's obs
                # (dead blobs AND fresh-reconnect frames carry garbage
                # u32 scores -- see SCORE_MAX_SANITY)
                red_score = sum(p["score"] for p in snap.players.values()
                                if p["team"] == TEAM_RED and env._alive(p)
                                and 0 <= p["score"] <= 10000)
                blue_score = sum(p["score"] for p in snap.players.values()
                                 if p["team"] == TEAM_BLUE
                                 and env._alive(p)
                                 and 0 <= p["score"] <= 10000)
                print("t=%3.0f players=%d F/R/B=%d/%d/%d my_team=%s "
                      "scores R:%d B:%d dmg:%.0f kills:%d" % (
                          now - t0, len(snap.players),
                          counts[TEAM_FREE], counts[TEAM_RED], counts[TEAM_BLUE],
                          myteam, red_score, blue_score,
                          env.dmg, env.kills))
            except Exception as e:
                print("t=%3.0f report error: %r" % (now - t0, e))
    finally:
        env.close()

    # team tolerance (see module docstring): BLUE or 3+ consecutive
    # FREE are fails; a single FREE is a legit spawn window
    bad_team = [t for t in team_hist if t == TEAM_BLUE]
    max_free_run = 0
    cur = 0
    for t in team_hist:
        if t == TEAM_FREE:
            cur += 1
            max_free_run = max(max_free_run, cur)
        else:
            cur = 0
    team_fail = bool(bad_team) or max_free_run >= 3

    print("RESULT steps=%d errors=%d team_hist=%s full_compo=%s joined=%s "
          "max_free_run=%d stale_fail=%s done_storm=%s"
          % (steps, errors, team_hist, saw_full_compo,
             any(t == TEAM_RED for t in team_hist), max_free_run,
             stale_fail, done_storm))
    if (errors == 0 and not team_fail and not stale_fail
            and saw_full_compo and any(t == TEAM_RED for t in team_hist)
            and steps > 1000):
        print("TDM_SMOKE_OK")
    else:
        print("TDM_SMOKE_FAIL")


if __name__ == "__main__":
    main()
