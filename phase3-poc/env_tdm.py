"""
env_tdm.py -- Quake3e TDM 2v2 env over the botctl UDS socket.

Game layer on DuelEnv (env_duel.py): same transport, pacing, action
repeater, reconnection and presence machinery; the differences are
TEAM AWARENESS:

- The agent joins TEAM RED via the client TEXT command (`team red`).
  The botctl ADD "team" field is userinfo metadata only (botctl.c:
  "team assignment itself is done by the game module; the client
  should pick a team with a TEXT command").
- TEAM JOIN WINDOW (2026-11-21, proven empirically against the running
  zz-osp-server4 QVM -- the 1.03a xq3osp source does NOT match it):
  "team red" is only accepted BEFORE the match has started (fewer than
  match_minplayers=2 players in the game; rl_tdm.cfg sets g_doWarmup 0,
  so there is no warmup -- the match starts as soon as 2 players are
  in). While the match is running the join is rejected with a
  SINGLE-CLIENT print that the dedicated build never logs
  (SV_SendServerCommand only echoes "broadcast:" prints) -- completely
  silent in the server log. The working moment is the post-spawn
  window: after InitGame the managed client auto-reconnects faster
  than the 3 QVM bots, so _step re-asserts "team red" every
  TEAM_TEXT_FAST_S (2 s) for TEAM_FIX_WINDOW_S (30 s) after a restart
  (tracked on _last_restart_id, NOT self.prev -- the done branch
  resets prev to the first new-map frame, so a prev-based restart check
  is dead code). A failed join is invisible; success is the snapshot
  team flipping to TEAM_RED (1) + the "entered the game (RED)"
  broadcast. Mid-match re-asserts fall back to the 10 s guard period
  (harmless silent no-ops that keep working if the team is ever lost).
- INTERMISSION EXIT (the TDM stall fix, 2026-11-21): OSP's CheckExit
  (g_main.c) exits the intermission only when at least one HUMAN-like
  client is ready -- SVF_BOT is skipped in the ready count, and a
  botctl managed client is human-like, so without a ready press the
  intermission waits forever (the 2026-11-20 smoke stall, 100 s).
  The ready flag toggles on a RISING button edge in
  ClientIntermissionThink (g_active.c: ATTACK/USE_HOLDABLE, 300 ms
  debounce); the TEXT command "ready" does NOT work (OSP_Cmd_Ready_f
  bails "Match is already in progress!" outside warmup). While
  self._intermission is set, the base env's repeater
  (env_duel._repeat_loop) presses ATTACK exactly once per
  intermission -- one edge, readyToExit 0 -> 1, exit after the 5 s
  intermission minimum (BeginIntermission resets readyToExit to 0, so
  the edge always toggles the right way).
- 2v2: agent (red) + 1 QVM teammate (red) vs 2 QVM opponents (blue).
  QVM bots get their team from the rcon addbot 3rd argument
  (`addbot <file> <skill> <team>` -- the 4v4 harness on port 28131
  runs exactly that way).
- ALL 3 QVM bots are owned by this env's guard: per-bot presence from
  the rcon status funnames (the snapshot carries no names), 15 s
  per-bot absence grace (bots auto-reconnect after kicks; the spawn
  window is 5-10 s), re-add ONLY the missing bot, with its team.
  (Bots survive same-process map changes; unconditional re-adds stack
  duplicates -- the duel-env lesson, 2026-11-20.)

OBSERVATION (54 floats):
  0..20   self: pos(3)/1000 vel(3)/1000 hp/100 armor/100 yaw pitch
          weapon one-hot(10) clip/50   (same layout as duel)
  21..29  enemy 1 (nearest blue): rel x/1000 rel h/100 rel z/1000
          horiz dist/1000 sin/cos(bearing - my yaw) hp/100
          weaponstate/3 present
  30..38  enemy 2 (second nearest blue): same
  39..47  teammate (nearest red, not self): same 9-float block
  48      my team score / 100 (sum of my-team player scores)
  49      enemy team score / 100
  50      dead (pm_type == PM_DEAD)
  51      intermission (PM_INTERMISSION / PM_SPINTERMISSION)
  52..53  reserved zeros

REWARD (same placeholder shaping as duel, team-attributed):
  +0.0005 / step while alive, +0.05 on hp increase, +0.002 per hp
  point lost by ANY tracked blue player, +0.5 per tracked blue kill,
  -0.5 on own death (penalty, not terminal); clamped [-1, 1].
  Attribution is TEAM-level: damage/kills by the QVM teammate also
  credit the reward (acceptable for the placeholder; a per-agent
  attribution would need server-side hit data).

Server: port 28135, home ~/.q3a_rl_tdm, g_gametype 3, g_friendlyFire 1
(matches the production TDM server config), 8-map rotation on the clan
TDM map list (start_rl_env_tdm.sh). Laptop side: SSH-forwarded socket
~/tmp/rl-lab/botctl-28135.sock.
"""

import math
import re
import threading
import time

import numpy as np

from botctl import (
    Botctl,
    WP_MACHINEGUN,
    BUTTON_ATTACK,
    PM_NORMAL,
    PM_DEAD,
    PM_INTERMISSION,
    PM_SPINTERMISSION,
)
from env_duel import DuelEnv, _clamp

OBS_DIM_TDM = 54
ACT_DIM = 5

TEAM_FREE = 0
TEAM_RED = 1
TEAM_BLUE = 2
TEAM_SPECTATOR = 3

# Score sanity bound for the team-score obs features (2026-11-21):
# freshly RECONNECTED clients carry garbage ps.persistant[PERS_SCORE]
# for a few frames after the map change (measured 4294967295 =
# 0xFFFFFFFF and 4294967297 = 0x100000001 -- engine-mirror garbage,
# NOT dead-blob garbage: the blobs are alive, PM_NORMAL, sane health,
# so the alive-only filter does not catch them). Real TDM scores are
# < 100 per 60 s match, so anything above this bound is garbage and
# skipped for that frame (minor undercount << a 4e7 obs spike).
SCORE_MAX_SANITY = 10000

# TEAM JOIN WINDOW (2026-11-21, proven empirically against the running
# zz-osp-server4 QVM): "team red" is only accepted BEFORE the match has
# started (fewer than match_minplayers=2 players in the game). Once the
# match is running, the QVM rejects the join with a single-client print
# ("The red team is locked!"-family) that the dedicated build does NOT
# log (SV_SendServerCommand only echoes broadcast: prints) -- the
# rejection is completely silent in the server log. The working moment
# is the post-spawn window: after InitGame the managed client
# auto-reconnects faster than the 3 QVM bots, so a "team red" sent as
# soon as the client is visible again lands while <2 players are in.
# => re-assert every TEAM_TEXT_FAST_S for TEAM_FIX_WINDOW_S after a
# restart, then fall back to the guard's 10 s period (mid-match
# re-asserts are harmless silent no-ops).
TEAM_FIX_WINDOW_S = 30.0
TEAM_TEXT_FAST_S = 2.0


def _opp_team(my_team):
    return TEAM_BLUE if my_team == TEAM_RED else TEAM_RED


# QVM bots: (rcon addbot file, color-stripped funname from
# scripts/bots.txt in zzz_dwc_bots.pk3, team).
BOTS = [
    ("dwcbot2", "DISCO", "red"),          # teammate
    ("dwcbot7", "[BOT]TrapDou", "blue"),  # opponent
    ("dwcbot9", "[BOT]RamMan", "blue"),   # opponent
]
BOT_SKILL = 5


class TDMEnv(DuelEnv):
    def __init__(self, sockpath: str, name: str = "rltdm", team: str = "red"):
        assert team in ("red", "blue")
        self.name = name
        self.team = team
        self._team_num = TEAM_RED if team == "red" else TEAM_BLUE
        # rcon helper on knecht (ssh + loopback), TDM port
        self.rcon_cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "hkuehnemund@192.168.69.25", "python3", "tmp/rl/rcon.py",
            "28135",
        ]
        self.bc = Botctl(sockpath)
        self.bc.ping()
        self.slot = self._add_self()
        if self.slot < 0:
            raise RuntimeError("botctl ADD failed (slot -1)")
        self.last_yaw = 0.0
        self.prev = None
        self.dmg = 0.0
        self.kills = 0
        self._last_opp_check = 0.0
        self._bot_absent_since = {}
        # team join window (see module docstring + TEAM_FIX_WINDOW_S):
        # restart detection lives on a SEPARATE tracker -- self.prev is
        # reset to the first NEW-map frame by the done branch, so a
        # restart_id check against self.prev never fires (dead code,
        # found 2026-11-21)
        self._last_restart_id = None
        self._restart_at = None
        self._last_team_text = 0.0
        # intermission ready-button (the base env's repeater reads these;
        # see env_duel._repeat_loop + the module docstring)
        self._intermission = False
        self._inter_ticks = 0
        # snapshot freshness (stale watchdog in _snap_until_new)
        self._last_snap_time = None
        self._stale_since = None
        self._action = np.zeros(ACT_DIM, dtype=np.float32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._repeat_loop, daemon=True)
        self._thread.start()
        snap0 = self._snap_until_new(first=True)
        self._last_restart_id = snap0.restart_id

    # ------------------------------------------------------------- identity
    def _add_self(self):
        self.slot = int(self.bc.add(self.name, self.team))
        if self.slot >= 0:
            try:
                self.bc.text(self.slot, "team %s" % self.team)
            except Exception:
                pass  # the 10 s guard re-asserts from the snapshot
        return self.slot

    # ----------------------------------------------------------------- teams
    def _enemies(self, snap, count=2):
        """Up to `count` nearest alive players of the OPPOSING team,
        judged against the opposing team constant (NOT against my
        current team field) -- a TEAM_FREE window (e.g. right after a
        full spawn, before the re-assert lands) must not turn the
        teammate into an 'enemy'."""
        me = self._me(snap)
        if me is None:
            return []
        et = _opp_team(self._team_num)
        out = []
        for num, p in snap.players.items():
            if num == self.slot or not self._alive(p):
                continue
            if p["team"] != et:
                continue
            d = ((p["origin"][0] - me["origin"][0]) ** 2
                 + (p["origin"][2] - me["origin"][2]) ** 2)
            out.append((d, p))
        out.sort(key=lambda t: t[0])
        return [p for _, p in out[:count]]

    def _teammate(self, snap):
        me = self._me(snap)
        if me is None:
            return None
        best, bd = None, None
        for num, p in snap.players.items():
            if num == self.slot or not self._alive(p):
                continue
            if p["team"] != self._team_num:
                continue
            d = ((p["origin"][0] - me["origin"][0]) ** 2
                 + (p["origin"][2] - me["origin"][2]) ** 2)
            if bd is None or d < bd:
                best, bd = p, d
        return best

    def _team_scores(self, snap, my_team):
        """ALIVE players only, SANITY-BOUNDED scores:
        (1) dead player blobs carry garbage u32 fields (measured score
        = 4294967297 = 2^32+1 on a dead blue, 2026-11-21 smoke t=46 --
        the same family as the dead-blob health garbage from the
        spike). (2) freshly RECONNECTED clients carry garbage
        PERS_SCORE for a few frames even while alive (measured
        4294967295 = 0xFFFFFFFF on an alive blue at the first frames
        of a load, smoke #3 t=151) -- the alive filter alone does NOT
        catch that. Summing them would drive obs[48]/[49] to ~4e7 for
        a few frames. Real TDM scores are < 100 per 60 s match, so
        scores above SCORE_MAX_SANITY are skipped for that frame
        (minor transient undercount << a 4e7 spike). (The game's own
        scoreboard counts dead players' persistent session scores, but
        the botctl snapshot only sees the ps blob -- alive-only +
        sanity-bounded is the consistent, bounded signal.)"""
        et = _opp_team(my_team)
        a = sum(p["score"] for p in snap.players.values()
                if p["team"] == my_team and self._alive(p)
                and 0 <= p["score"] <= SCORE_MAX_SANITY)
        b = sum(p["score"] for p in snap.players.values()
                if p["team"] == et and self._alive(p)
                and 0 <= p["score"] <= SCORE_MAX_SANITY)
        return a, b

    # ------------------------------------------------- guard (team + bots)
    def _ensure_opponent(self):
        """10 s throttled. Two jobs:
        1) re-assert the agent's team: a full spawn brings the client
           back as TEAM_FREE and the game team only changes through the
           client TEXT command.
        2) re-add missing QVM bots: per-bot presence from the rcon
           status funnames (the snapshot has no names; the managed
           agent name cannot collide with a bot funname), 15 s per-bot
           absence grace, re-add ONLY what is missing, WITH its team.
           First appearance: bots are added ~20-30 s after env start
           (two guard cycles), same latency as the duel guard."""
        now = time.time()
        if now - self._last_opp_check < 10.0:
            return
        self._last_opp_check = now
        try:
            snap = self.bc.snap()
        except Exception:
            return
        me = self._me(snap)
        if me is not None and me["team"] != self._team_num:
            try:
                self.bc.text(self.slot, "team %s" % self.team)
            except Exception:
                pass
        try:
            import subprocess
            out = subprocess.run(self.rcon_cmd + ["status"],
                                 capture_output=True, text=True,
                                 timeout=20).stdout
        except Exception:
            return
        clean = re.sub(r"\^[0-9a-fA-F]", "", out)
        for fname, tag, team in BOTS:
            if tag in clean:
                self._bot_absent_since[fname] = None
                continue
            t0 = self._bot_absent_since.get(fname)
            if t0 is None:
                self._bot_absent_since[fname] = now
                continue
            if now - t0 < 15.0:
                continue
            self._bot_absent_since[fname] = None
            try:
                import subprocess
                subprocess.run(self.rcon_cmd + ["addbot", fname,
                                                str(BOT_SKILL), team],
                               capture_output=True, text=True, timeout=20)
            except Exception:
                pass

    # ------------------------------------------------------------- build obs
    @staticmethod
    def _rel_block(o, base, me, p):
        """9-float relative block: rel x/h/z, horiz dist, sin/cos of
        (bearing - my yaw), hp/100, weaponstate/3, present."""
        px, py, pz = me["origin"]
        ex, ey, ez = p["origin"]
        rx, rh, rz = ex - px, ey - py, ez - pz
        dist = math.hypot(rx, rz)
        yaw = me["viewangles"][0]
        bearing = math.atan2(rz, rx) * 180.0 / math.pi
        rel = bearing - yaw
        o[base] = rx / 1000.0
        o[base + 1] = rh / 100.0
        o[base + 2] = rz / 1000.0
        o[base + 3] = dist / 1000.0
        o[base + 4] = math.sin(math.radians(rel))
        o[base + 5] = math.cos(math.radians(rel))
        o[base + 6] = p["health"] / 100.0
        o[base + 7] = p["weaponstate"] / 3.0
        o[base + 8] = 1.0

    def _build_obs(self, snap, prev_snap):
        me = self._me(snap)
        o = np.zeros(OBS_DIM_TDM, dtype=np.float32)
        if me is None:
            return o
        px, py, pz = me["origin"]
        o[0] = px / 1000.0
        o[1] = py / 1000.0
        o[2] = pz / 1000.0
        if prev_snap is not None:
            pm = self._me(prev_snap)
            if (self._alive(pm) and self._alive(me)
                    and snap.time > prev_snap.time):
                dt = (snap.time - prev_snap.time) / 1000.0
                o[3] = (px - pm["origin"][0]) / dt / 1000.0
                o[4] = (py - pm["origin"][1]) / dt / 1000.0
                o[5] = (pz - pm["origin"][2]) / dt / 1000.0
        o[6] = (me["health"] / 100.0) if self._alive(me) else 0.0
        o[7] = _clamp(me["armor"] / 100.0, 0.0, 3.0)
        yaw = me["viewangles"][0]
        pitch = me["viewangles"][1]
        self.last_yaw = yaw
        o[8] = ((yaw + 180.0) % 360.0 - 180.0) / 180.0
        o[9] = _clamp(pitch / 90.0, -1.0, 1.0)
        wp = me["weapon"]
        if 1 <= wp <= 10:
            o[9 + wp] = 1.0
        clip = me["ammo"][wp] if 0 <= wp < len(me["ammo"]) else 0
        o[20] = _clamp(clip / 50.0, 0.0, 1.0) if clip >= 0 else 1.0

        es = self._enemies(snap, 2)
        if es:
            self._rel_block(o, 21, me, es[0])
        if len(es) > 1:
            self._rel_block(o, 30, me, es[1])
        tm = self._teammate(snap)
        if tm is not None:
            self._rel_block(o, 39, me, tm)

        a, b = self._team_scores(snap, self._team_num)
        o[48] = a / 100.0
        o[49] = b / 100.0
        pm_type = me["pm_type"]
        o[50] = 1.0 if pm_type == PM_DEAD else 0.0
        o[51] = 1.0 if pm_type in (PM_INTERMISSION, PM_SPINTERMISSION) else 0.0
        return o

    # ------------------------------------------------- done-branch respawn wait
    def _wait_alive(self, timeout=15.0, ref_snap=None):
        """TDM override (2026-11-21, smoke #2 forensics): wait out the
        intermission on the OLD map, then return as soon as the client
        is visible again on the NEW map -- any pm_type, NOT PM_NORMAL.

        The base DuelEnv._wait_alive blocks until the agent respawns as
        PM_NORMAL. In TDM that is never true in the pre-match window: a
        cold boot that lands mid-match re-enters as TEAM_SPECTATOR and
        only becomes PM_NORMAL after the post-spawn `team red` lands
        (TEAM JOIN WINDOW). So on every rotation the full 15 s timeout
        ran in the done branch while the step loop -- and with it the 2 s
        fast re-asserts -- was frozen through the entire post-spawn
        window (the client reconnects +4-8 s after G_Init, the QVM bots
        +6-10 s, the late-join restart +12-15 s) -> the first `team red`
        landed mid-match = silently rejected (measured: 15 s block
        01:23:07->:22 on load #A, the text only at rel 95, after the
        bots). The smoke-#2 join at t=91 survived only because one 2 s
        tick won a narrow window -- fragile.

        The wait is anchored on restart_id: keep polling while the
        snapshot is still on the old map. That also COLLAPSES the ~1-2 s
        intermission into the single episode boundary -- the base
        wait's storm-prevention role. A plain visibility check would NOT
        do that: the client IS visible in the intermission, so the wait
        would return immediately and re-fire done=True on every step
        until the map reloads. Once the new map is up, return at first
        client visibility: RED agents spawn PM_NORMAL (as before),
        spectator agents return as TEAM_SPECTATOR and the unblocked
        step loop lands the fast re-assert within one 2 s period of the
        window. The 15 s cap stays; on timeout the done branch
        continues as in the base (obs flags carry the state).
        """
        snap = ref_snap if ref_snap is not None else self._snap_until_new()
        rid = snap.restart_id
        t0 = time.time()
        while time.time() - t0 < timeout:
            snap = self._snap_until_new()
            if snap.restart_id == rid:
                continue  # still the old map (intermission / reload)
            if self._me(snap) is not None:
                return  # new map, client back: any pm_type
        # timeout: continue anyway (obs flags carry the state)

    # ------------------------------------------------------------------- step
    def _step(self):
        """Same frame walk as DuelEnv._step, but reward/damage/kills are
        team-attributed (all tracked opposing-team players)."""
        snap = self._snap_until_new()

        # RESTART DETECTION (2026-11-21): tracked on _last_restart_id,
        # NOT on self.prev -- the done branch below resets self.prev to
        # the first new-map frame, so a restart check against self.prev
        # never fires (that was the dead immediate re-assert).
        restart_changed = (self._last_restart_id is not None
                           and snap.restart_id != self._last_restart_id)
        if self._last_restart_id is None:
            self._last_restart_id = snap.restart_id
        if restart_changed:
            self._last_restart_id = snap.restart_id
            self._restart_at = time.time()

        # map (re)start: positions/velocities across the boundary are
        # meaningless and reward continuity must break
        if self.prev is not None and snap.restart_id != self.prev.restart_id:
            self.prev = None

        me = self._me(snap)
        if me is None:
            snap = self._ensure_present(snap)
            me = self._me(snap)
            self.prev = None

        # INTERMISSION READY BUTTON (see module docstring): arm the base
        # repeater's one-shot ATTACK edge while we are in the
        # intermission; without it the intermission waits forever when
        # no action is sent (the 2026-11-20 smoke stall, 100 s).
        if me is not None:
            self._intermission = (me["pm_type"] in (PM_INTERMISSION,
                                                    PM_SPINTERMISSION))
            # TEAM JOIN WINDOW (see TEAM_FIX_WINDOW_S): a full spawn
            # brings the client back as TEAM_SPECTATOR and the match has
            # not started yet (the QVM bots are still reconnecting) --
            # that is the only window where "team red" is accepted.
            # Re-assert fast while the window is open; once the match is
            # running the same command is a silent no-op, so fall back
            # to the slow guard period (which keeps the bots alive too).
            #
            # GATE DECOUPLED FROM PERS_TEAM (2026-11-21, smoke #3): the
            # running QVM can claim PERS_TEAM=RED for a client that is
            # NOT really in the team -- a fresh ADD with team metadata
            # showed the agent "RED" in the snapshot on the first
            # post-restart load while the server had it as SPECTATOR
            # (no "entered the game" broadcast, no userinfo t\1, no
            # sessionTeam; the NEXT load reverted it to 3). Gating the
            # re-assert on me["team"] alone would then skip the one
            # window where the text is needed. During the restart
            # window the text therefore goes out regardless: for a REAL
            # red join SetTeam is a clean no-op (1.03a g_cmds.c:
            # team == sessionTeam -> return, no state change), for the
            # phantom it performs the join.
            in_window = (self._restart_at is not None
                         and time.time() - self._restart_at
                         <= TEAM_FIX_WINDOW_S)
            if (me["team"] != self._team_num or in_window
                    or restart_changed):
                period = (TEAM_TEXT_FAST_S if (restart_changed or in_window)
                          else 10.0)
                if time.time() - self._last_team_text >= period:
                    self._last_team_text = time.time()
                    try:
                        self.bc.text(self.slot, "team %s" % self.team)
                    except Exception:
                        pass  # the guard re-asserts on its own cycle

        # keep the team + the QVM bots alive (throttled, ~10 s)
        self._ensure_opponent()

        r = 0.0
        done = False
        if me is not None:
            pm = me["pm_type"]
            if self.prev is not None:
                pm_prev = self._me(self.prev)
                alive_now = self._alive(me)
                alive_prev = self._alive(pm_prev)
                if alive_prev and alive_now:
                    r += 0.0005
                    if me["health"] > pm_prev["health"]:
                        r += 0.05
                elif alive_prev and (self._dead(me) or me is None):
                    # death is a PENALTY, not a terminal state (as in duel)
                    r -= 0.5
                if alive_now:
                    et = _opp_team(self._team_num)
                    e_prev = {n: p for n, p in self.prev.players.items()
                              if p["team"] == et and self._alive(p)}
                    e_now = {n: p for n, p in snap.players.items()
                             if p["team"] == et and self._alive(p)}
                    for n, p_now in e_now.items():
                        p_prev = e_prev.get(n)
                        if (p_prev is not None
                                and p_prev["health"] > p_now["health"]):
                            d = p_prev["health"] - p_now["health"]
                            r += 0.002 * d
                            self.dmg += d
                    for n in e_prev:
                        if n not in e_now:
                            # a tracked alive enemy is now dead/gone:
                            # team kill (teammate kills credit too)
                            r += 0.5
                            self.kills += 1
            if pm in (PM_INTERMISSION, PM_SPINTERMISSION):
                done = True

        r = max(-1.0, min(1.0, r))

        prev_snap = self.prev
        self.prev = snap
        if done:
            self._wait_alive(ref_snap=snap)
            snap = self._snap_until_new()
            self.prev = None
            o = self._build_obs(snap, None)
            self.prev = snap
            return o, r, True
        return self._build_obs(snap, prev_snap), r, False
