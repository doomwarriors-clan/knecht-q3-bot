"""
env_duel.py -- Quake3e FFA-duel env over the botctl UDS socket.

The engine runs on knecht (port 28134, home ~/.q3a_rl, sv_fps 40,
sv_timescale 10); on the laptop this module connects to an SSH-forwarded
local unix socket, so it is transport-agnostic.

Measured pacing (2026-11-20, diag_env.py): the engine runs at 10.15x
realtime -> ~400 ticks/s wall (25 ms game-time ticks); this env SAMPLES
new frames at ~50 steps/s wall (5 ms poll sleep + ~15 ms SSH RTT), so
one env step covers ~200 ms of game time. Movement is NOT paced by the
sampler: a background thread re-sends the current action every 2.5 ms
wall (= one engine tick) so every tick sees a real usercmd (the engine
drains the staged queue per tick and sends an idle cmd otherwise --
1 cmd per observed frame would mean a 1/8 duty cycle).

ACTION (5-dim continuous, tanh-squashed to [-1, 1]):
  a0 forward  -> usercmd fwd   i16 (a0 * 127)
  a1 strafe   -> usercmd right i16 (a1 * 127)
  a2 yaw      -> yaw += a2 * 45 deg per step (absolute sent to engine)
  a3 pitch    -> pitch = a3 * 90 deg, clamped [-90, 90]
  a4 fire     -> BUTTON_ATTACK if a4 > 0
Weapon is pinned to MACHINEGUN (weapon selection is Phase-3 proper work).

OBSERVATION (44 floats):
  0..2   px, py(height), pz  / 1000
  3..5   vx, vy, vz (per game-second) / 1000
  6      health / 100 (0 when dead -- see DEAD DETECTION)
  7      armor / 100
  8      yaw wrapped to [-1, 1]
  9      pitch / 90
  10..19 weapon one-hot (10)
  20     selected-weapon clip / 50, clamped
  21     enemy rel x / 1000
  22     enemy rel height / 100
  23     enemy rel z / 1000
  24     enemy horizontal dist / 1000
  25,26  sin, cos of (bearing - my yaw)
  27     enemy health / 100
  28     enemy weaponstate / 3
  29     enemy present
  30..33 item 1: rel x/1000, rel h/100, rel z/1000, modelindex/250
  34..37 item 2: same
  38     dead (pm_type == PM_DEAD)
  39     intermission (PM_INTERMISSION / PM_SPINTERMISSION)
  40..43 reserved zeros

DEAD DETECTION (hard-won, 2026-11-20 diag): a DEAD player stays in the
snapshot (pm_type == PM_DEAD) but its health field carries GARBAGE u32
values (~4.29e9, i.e. ~0xFFFFFFFF), never 0. Death must be detected via
pm_type, and every health value is treated as valid only in [0, 300]
(real duel max = 100 + 100 suit). The garbage is what corrupted run-1
rewards (rew 8.4e3, loss 3e11).

ROBUSTNESS: on a map restart the restart_id changes -> prev state is
dropped (no cross-map velocity/reward). If the client is ABSENT from
the snapshots for >2 s (OSP can kick clients on "Server command
overflow" during intermission; the normal map-restart disconnect/
reconnect resolves within 2 s), the env adopts an existing same-name
client or re-ADDs itself.

REWARD (PLACEHOLDER for the workload PoC -- not tuned for learning):
  +0.0005 / step while alive
  +0.05 on health increase (pickup)
  +0.002 per enemy hp point lost (while alive)
  +0.5 on kill, -0.5 on own death (pm_type-based)
  total clamped to [-1, 1] per step
"""

import math
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

OBS_DIM = 44
ACT_DIM = 5
WP_ONEHOT = 10

HP_MAX = 300            # real duel max health (100 + 100 suit)
CMD_PERIOD = 0.0025     # s wall per engine tick (25 ms game / 10x timescale)
ABSENT_GRACE = 2.0      # s a client may be absent (map-restart cycle) before re-ADD


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class DuelEnv:
    def __init__(self, sockpath: str, name: str = "rlbot"):
        self.name = name
        self.bc = Botctl(sockpath)
        self.bc.ping()
        self.slot = int(self.bc.add(name))
        if self.slot < 0:
            raise RuntimeError("botctl ADD failed (slot -1)")
        self.last_yaw = 0.0
        self.prev = None
        self._action = np.zeros(ACT_DIM, dtype=np.float32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._repeat_loop, daemon=True)
        self._thread.start()
        self._snap_until_new(first=True)

    # ------------------------------------------------------------------ util
    def _me(self, snap):
        return snap.players.get(self.slot)

    @staticmethod
    def _alive(p):
        return (p is not None and p["pm_type"] == PM_NORMAL
                and 0 <= p["health"] <= HP_MAX)

    @staticmethod
    def _dead(p):
        return p is not None and p["pm_type"] == PM_DEAD

    def _snap_until_new(self, first=False):
        """Blocking snap request that only returns on a NEW engine frame
        (snap.time change). Sleeps 5 ms between polls so the loop costs
        almost no CPU while the engine paces at ~400 frames/s wall."""
        last = None
        if first:
            return self.bc.snap()
        while True:
            snap = self.bc.snap()
            if last is None or snap.time != last:
                return snap
            last = snap.time
            time.sleep(0.005)

    def _enemy(self, snap):
        me = self._me(snap)
        if me is None:
            return None
        best, bd = None, None
        for num, p in snap.players.items():
            # nearest alive enemy with a sane health value (dead blobs
            # carry garbage u32 health and must not be "acquired")
            if num == self.slot or not self._alive(p):
                continue
            dx = p["origin"][0] - me["origin"][0]
            dz = p["origin"][2] - me["origin"][2]
            d = dx * dx + dz * dz
            if bd is None or d < bd:
                best, bd = p, d
        return best

    # ------------------------------------------------------- action repeater
    def _repeat_loop(self):
        """Stage the current action once per engine tick (2.5 ms wall at
        sv_timescale 10). The engine consumes staged inputs per tick and
        sends an idle cmd when the queue is empty, so this is what keeps
        the bot at a full movement duty cycle. Fire-and-forget; the
        engine's 128-deep per-client queue absorbs jitter."""
        while not self._stop.is_set():
            a = self._action
            try:
                fwd = int(_clamp(round(a[0] * 127.0), -127, 127))
                side = int(_clamp(round(a[1] * 127.0), -127, 127))
                yaw = (self.last_yaw + a[2] * 45.0) % 360.0
                pitch = _clamp(a[3] * 90.0, -90.0, 90.0)
                buttons = BUTTON_ATTACK if a[4] > 0.0 else 0
                self.bc.cmd(self.slot, fwd, side, 0, buttons,
                            WP_MACHINEGUN, pitch, yaw)
            except Exception:
                break
            self._stop.wait(CMD_PERIOD)

    # ------------------------------------------------------------- build obs
    def _build_obs(self, snap, prev_snap):
        me = self._me(snap)
        o = np.zeros(OBS_DIM, dtype=np.float32)
        if me is None:
            return o
        px, py, pz = me["origin"]
        o[0] = px / 1000.0
        o[1] = py / 1000.0
        o[2] = pz / 1000.0

        # velocity from the previous frame (game units per game-second)
        if prev_snap is not None:
            pm = self._me(prev_snap)
            if pm is not None and snap.time > prev_snap.time:
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
        if 1 <= wp <= WP_ONEHOT:
            o[9 + wp] = 1.0
        clip = me["ammo"][wp] if 0 <= wp < len(me["ammo"]) else 0
        o[20] = _clamp(clip / 50.0, 0.0, 1.0) if clip >= 0 else 1.0

        e = self._enemy(snap)
        if e is not None:
            ex, ey, ez = e["origin"]
            rx, rh, rz = ex - px, ey - py, ez - pz
            dist = math.hypot(rx, rz)
            bearing = math.atan2(rz, rx) * 180.0 / math.pi
            o[21] = rx / 1000.0
            o[22] = rh / 100.0
            o[23] = rz / 1000.0
            o[24] = dist / 1000.0
            rel = bearing - yaw
            o[25] = math.sin(math.radians(rel))
            o[26] = math.cos(math.radians(rel))
            o[27] = e["health"] / 100.0
            o[28] = e["weaponstate"] / 3.0
            o[29] = 1.0

        # two nearest ET_ITEM entities
        items = sorted(
            (it for it in snap.ents if it["eType"] == 2),
            key=lambda it: (it["origin"][0] - px) ** 2 + (it["origin"][2] - pz) ** 2,
        )[:2]
        for k, it in enumerate(items):
            base = 30 + 4 * k
            o[base] = (it["origin"][0] - px) / 1000.0
            o[base + 1] = (it["origin"][1] - py) / 100.0
            o[base + 2] = (it["origin"][2] - pz) / 1000.0
            o[base + 3] = it["modelindex"] / 250.0

        pm_type = me["pm_type"]
        o[38] = 1.0 if pm_type == PM_DEAD else 0.0
        o[39] = 1.0 if pm_type in (PM_INTERMISSION, PM_SPINTERMISSION) else 0.0
        return o

    # ------------------------------------------------------------------- api
    def reset(self):
        """Fresh observation (e.g. at run start)."""
        snap = self._snap_until_new(first=True)
        self.prev = None
        o = self._build_obs(snap, None)
        self.prev = snap
        return o

    def send_action(self, a):
        """Set the action the repeater thread stages on every engine tick."""
        self._action = np.asarray(a, dtype=np.float32)

    def _wait_alive(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            snap = self._snap_until_new()
            me = self._me(snap)
            if me is not None and me["pm_type"] == PM_NORMAL:
                return
        # timeout: continue anyway (obs flags carry the state)

    def _ensure_present(self, snap):
        """Return a snapshot in which our client is present. If it is
        absent, wait out the map-restart disconnect/reconnect cycle
        (ABSENT_GRACE s); if it is still gone (e.g. kicked by OSP
        'Server command overflow'), adopt an existing same-name client
        or re-ADD ourselves."""
        if self._me(snap) is not None:
            return snap
        t0 = time.time()
        while time.time() - t0 < ABSENT_GRACE:
            snap = self._snap_until_new()
            if self._me(snap) is not None:
                return snap
        try:
            for num, name in self.bc.list():
                if name == self.name:
                    self.slot = int(num)
                    break
            else:
                try:
                    self.bc.remove(self.slot)
                except Exception:
                    pass
                self.slot = int(self.bc.add(self.name))
        except Exception:
            pass
        t0 = time.time()
        while time.time() - t0 < 10.0:
            snap = self._snap_until_new()
            if self._me(snap) is not None:
                return snap
        return snap

    def step(self):
        """Advance to the next new engine frame.

        Returns (obs, reward, done). The reward covers the transition
        prev frame -> this frame; `done` marks death (pm_type-based) or
        intermission -- after `done` the env has already waited for the
        respawn, so the returned obs is the first state of the next
        episode (GRU hidden state must be reset by the caller)."""
        snap = self._snap_until_new()

        # map (re)start: positions/velocities across the boundary are
        # meaningless and reward continuity must break
        if self.prev is not None and snap.restart_id != self.prev.restart_id:
            self.prev = None

        me = self._me(snap)
        if me is None:
            snap = self._ensure_present(snap)
            me = self._me(snap)
            self.prev = None

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
                    r -= 0.5
                    done = True
                if alive_now:
                    e_prev = self._enemy(self.prev)
                    e_now = self._enemy(snap)
                    if e_prev is not None:
                        if e_now is not None and e_prev["health"] > e_now["health"]:
                            r += 0.002 * (e_prev["health"] - e_now["health"])
                        elif e_now is None or self._dead(e_now):
                            r += 0.5  # kill
            if pm in (PM_INTERMISSION, PM_SPINTERMISSION):
                done = True

        r = max(-1.0, min(1.0, r))

        prev_snap = self.prev
        self.prev = snap
        if done:
            self._wait_alive()
            snap = self._snap_until_new()
            self.prev = None
            o = self._build_obs(snap, None)
            self.prev = snap
            return o, r, True
        return self._build_obs(snap, prev_snap), r, False

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.bc.remove(self.slot)
        except Exception:
            pass
        self.bc.close()
