# rl-lab -- Phase-3 PoC: PPO training on the laptop GPU (RTX 2080)

Workload test (2026-11-20): can the laptop RTX 2080 Max-Q (8 GB) train the
RL duel bot while the game engine runs on knecht? Answer so far: **yes** --
the net is 1.86 M params (fits in 8 GB with ~50x headroom), the env on knecht
paces everything (46 steps/s at sv_fps 40 + sv_timescale 10), and the GPU
work per step is one small forward pass.

## Topology

```
knecht (192.168.69.25)                      laptop (this machine)
  quake3e.ded.x64                            .venv (torch 2.15 cu132)
  rl-env-2026.11.20 build                    train.py (PPO, GPU 2080)
  port 28134, home ~/.q3a_rl                 env_duel.py (botctl client)
  sv_fps 40, sv_timescale 10                 |
  botctl socket:                             |  ssh -N -L unix:unix forward
  ~/.q3a_rl/botctl-28134.sock  <------------+  ~/tmp/rl-lab/botctl-28134.sock
```

- Engine branch `rl-env-2026.11.20` (knecht `~/tmp/q3e_dwc`) = `dwc` tip
  `6bb9b17e` (botctl, PR #13) + cherry-picked `64c06541` (sv_timescale).
  Binary md5 `09dd742e043912dbc84b860270b27221` (USE_BOTCTL=1).
- Server: FFA duel (g_gametype 1) with a **12-map rotation** (the dl1 Duel
  list: pro-q3dm6 ztn3tourney1 pro-q3tourney4 q3tourney2 hub3tourney1
  hub3aeroq3 cht3 q3tourney4 nodm9 pro-nodm9 cpm3 cpm1a — all BSP+AAS on
  knecht), opponent = `dwcbot5` (skill 5, "NUDELpas") from
  `zzz_dwc_bots.pk3`. While `sv_timescale > 1` the engine rejects all real
  clients (bots-only). Rotation: this engine build has no map-list
  rotation; at each natural match end (60 s wall) the OSP QVM executes
  the `nextmap` cvar (default `map_restart 0` = in-place, reset on every
  spawn). The feeder (in `start_rl_env.sh`) polls the LIVE server map
  (`rcon status` over loopback — NOT the log, which is block-buffered up
  to ~30 s and once let a late shot land after the successor's spawn
  reset -> nextmap = current map -> wasted same-map full spawns),
  successor = one past the OBSERVED map in the list (re-derived each
  cycle, no counter — a counter points at the starting map on the first
  detection and then never fires again), 3 idempotent `nextmap` shots at
  +0/+10/+20 s (each re-verifies the live server map first), and a 150 s
  watchdog that forces `map <successor>` if a whole cycle was missed
  (also covers the join-spawn of the first client, which resets
  nextmap). The OPPONENT is owned by the env's `_ensure_opponent` guard
  (snapshot presence + 15 s absence grace — rcon QVM bots auto-reconnect
  after same-process full spawns, so an immediate re-add would stack
  duplicates), NOT by the feeder. Start/stop: `~/tmp/rl/` on knecht
  (copies in this dir).

  **Grep-escaping trap (cost one silent rotation attempt):** the PCRE
  pattern `mapname\\\K[^\\]+` must reach `grep -P` exactly so — in the
  first feeder it was double-quoted, the backslashes collapsed to
  `mapname\\\K[^\]+`, PCRE failed with "missing terminating ]", the
  subshell's `|| true` swallowed the error, `live` was always empty, and
  the trigger NEVER fired with no error in any log. The pattern is
  single-quoted in the script and must be verified standalone
  (`tail -n 400 <server log> | grep -a "^InitGame:.*mapname" | tail -1 |
  grep -aoP 'mapname\\\K[^\\]+'` must print a map name) before
  (re)starting the engine.

  **Console-`rcon` trap (cost another silent rotation attempt):** in the
  dedicated build the console's `rcon` command is the CLIENT command
  (`CL_Rcon_f`) — it runs but ships its packet to a bogus address: no
  "Unknown command", no "Rcon from" log line, no effect, total silence.
  All feeder rcon therefore goes over loopback UDP
  (`python3 ~/tmp/rl/rcon.py <cmd...>`), where a quoted cvar value must
  keep LITERAL quotes in the payload (`rcon.py nextmap "\"map <m>\""` —
  the server tokenizer re-groups it; without quotes only `map` would be
  stored).
- Tunnel (laptop): `ssh -fN -o ServerAliveInterval=10 -o ExitOnForwardFailure=yes
  -L ~/tmp/rl-lab/botctl-28134.sock:~/.q3a_rl/botctl-28134.sock hkuehnemund@192.168.69.25`

## Files

- `botctl.py` -- protocol client (copied from the spike suite, stdlib only)
- `env_duel.py` -- duel env: 44-dim obs, 5-dim tanh action
  [fwd, side, yaw-delta, pitch-abs, fire], placeholder reward
- `net.py` -- policy: MLP(44->512->512) + GRU(512) + Gaussian head, 1.86 M params
- `ppo.py` -- minimal PPO (GAE, clipped surrogate, squashed Gaussians,
  hidden state stored per step, no BPTT through the chain)
- `train.py` -- rollout 2048 steps + 4 epochs; stats_<name>.jsonl,
  <name>_policy.pt, <name>.pid
- `impact.py` -- sampler: nvidia-smi + CPU + RAM + coretemp -> CSV
- `start_rl_env.sh` / `stop_rl_env.sh` -- knecht `~/tmp/rl/` copies
  (rotation feeder + engine start/stop; see the Topology note)
- `recovery_test.py` -- kill/start recovery test (stop engine mid-session,
  cold start, env must reconnect + re-ADD itself + opponent comes back)
- `rotation_test.py` -- map-rotation survival test (250 s across >= 2
  full-spawn map changes; env side: 0 errors + both clients back = pass;
  the actual map change must be verified from the knecht server log:
  >= 2 distinct `mapname` values in InitGame lines + the feeder's
  `Rcon from ...: nextmap "map <next>"` entries)
- `rcon.py` -- one-shot rcon over loopback on knecht, `[port] <cmd...>`
  (port defaults to 28134; the TDM env uses 28135). Quoted cvar values
  keep LITERAL quotes in the payload (server tokenizer re-groups).
- `env_tdm.py` -- TDM 2v2 env, game layer on `DuelEnv`: 54-dim
  team-aware obs (2 enemies + teammate + team scores), agent joins TEAM
  RED via the client TEXT command (`team red`; re-asserted from the
  snapshot after every full spawn), guard owns the 3 QVM bots
  (1 red teammate + 2 blue opponents) via per-bot rcon-status funname
  checks + 15 s absence grace. See the "TDM 2v2 env" section.
- `start_rl_env_tdm.sh` / `stop_rl_env_tdm.sh` -- TDM engine on knecht
  port 28135 (home `~/.q3a_rl_tdm`), same feeder design, 8-map rotation
  on the clan TDM map list. Copies in knecht `~/tmp/rl/`.
- `tdm_smoke.py` -- TDM smoke test (~4 min): 0 errors + full team
  composition (F/R/B = 0/2/2 incl. the agent) + agent team stays RED
  across map changes (a single FREE report is a legit spawn window;
  3+ consecutive FREE or any BLUE fails) + no STALE-SNAP (the report's
  snap.time must advance between 15 s-apart reports; 3 consecutive
  no-advances fail) + no DONE storm (>= 3 episode boundaries in 50 s
  warns) = TDM_SMOKE_OK (map changes verified from the knecht log
  separately).

## Run

```
.venv/bin/python train.py --minutes 5 --name run1        # (start impact.py first/parallel)
.venv/bin/python impact.py run1.pid impact_run1.csv 2
```

## Scope (PoC, NOT the real Phase-3 net)

Placeholder reward (survival/kill/death/pickup), weapon pinned to MG, no
BPTT, obs is a hand-picked subset of the botctl snapshot. Purpose: measure
the laptop impact (GPU/CPU/RAM/thermals) of a real PPO loop, and prove the
knecht-engine + laptop-GPU topology end to end. The real observation/action
spaces come from the Phase-1 sim-parity work.

## Diagnostics (diag_env.py, 2026-11-20) — findings + env fixes

- **Real pacing:** engine runs 10.15x realtime (~400 ticks/s wall at
  sv_fps 40 + sv_timescale 10). The env SAMPLES new frames at ~50-77
  steps/s wall (5 ms poll sleep + ~15 ms SSH RTT), so one env step covers
  ~130-200 ms of game time. The variable step rates seen in run-1 stats
  (40-148/s) are just this sampling jitter.
- **Dead-player blob:** a DEAD player stays in the snapshot (pm_type 3)
  but its `health` field carries GARBAGE u32 (~4.29e9), never 0. Death is
  detected via `pm_type == PM_DEAD`; health is only trusted in [0, 300].
  This garbage corrupted run-1's it 1-2 (rew 8.4e3, loss 3.1e11 via the
  damage term).
- **OSP "Server command overflow":** during intermission OSP queues 65+
  pending `cs`/`cp` commands + `map_restart`; the queue overflowed and the
  engine KICKED both clients (NUDELpas + rlbot) → empty server → run-1's
  zero-reward phase (it 4-8: `me` absent, no reward branch can fire).
  The env now waits 2 s (map-restart reconnect cycle) and then adopts /
  re-ADDs its client if it is still absent.
- **Movement duty cycle:** the engine drains the staged input queue per
  tick and sends an idle cmd when empty — 1 cmd per observed frame would
  be 1/8 duty. `env_duel.py` now runs a background thread that re-sends
  the current action every 2.5 ms wall (one engine tick), fire-and-forget
  (the engine's 128-deep queue absorbs jitter).
- **Phase-3 notes:** full 400 ticks/s observation is not reachable over
  SSH RTT (~15 ms) — a co-located sidecar on knecht (or a multi-snapshot
  protocol op) would be needed if per-tick obs becomes important; reward
  per env step then covers ~200 ms of game time and must be scaled to
  match.

## Reconnect + long-run robustness (2026-11-20, after run3 crashed)

run3 (2 h, 1on1) died at ~8.7 min with `ConnectionError: botctl socket
closed`: OSP issues a FULL `map` (SV_Restart + FS_Startup, ~15-20 s,
`botctl: stopped` -> `started`) in ~1 of 7 intermission cycles instead of
the normal in-place `map_restart`. A full restart EOFs the established
sidecar connection (the socket file is recreated at the same path, but the
old connection is dead) and does NOT re-add rcon-added QVM bots. Fixes in
`env_duel.py`:

- **`_ensure_conn(budget=90)`** -- ping/reconnect loop; a full restart
  needs ~15-20 s, a cold engine start ~30-40 s, both within budget.
- **`_reconnect()`** -- re-establish the socket, re-adopt (or re-ADD) our
  client, restart the repeater thread if it died. The repeater itself now
  retries (sleep 1 s) instead of breaking, and reads `self.bc` per call, so
  it transparently picks up a replaced client.
- **`step()` guards the WHOLE frame walk** (`step()`/`_step()` split): a
  connection failure can happen anywhere in a step, not just at the first
  snap -- an engine shutdown disconnects the client one tick BEFORE
  `botctl: stopped`, so a step can be inside the absent-client wait or the
  post-done respawn wait when the socket dies (that exact escape was the
  first recovery-test failure). On loss: one `_reconnect()` per step, then
  a clean episode boundary (`done=True`, reward 0), so a restart costs one
  episode, not the run.
- **`_ensure_opponent()`** -- the QVM opponent (`dwcbot5`/NUDELpas) vanishes
  after a full restart; presence is judged from the SNAPSHOT, because
  managed botctl clients DO show up in rcon `status` but render exactly
  like QVM bots (`bot` in the address column) -- rcon cannot distinguish
  them. A missing opponent is re-added via rcon ON KNECHT (direct UDP from
  the laptop is blocked by the firewalld LAN block, so rcon goes through
  `ssh ... python3 tmp/rl/rcon.py`, loopback on knecht). Throttled to 10 s.
- `recovery_test.py` exercises the real path: env session live, engine
  killed at t=20 s, cold start at t=65 s, plus a 20 s self-heal phase at
  the end (the OSP "Server command overflow" kicks ALL clients at some map
  restarts -- right in that window -- so the env's `_ensure_present`
  re-ADD + the opponent guard have to recover first). VERIFIED 2026-11-20:
  0 errors across the ~56 s outage, one slow step -> `done=True` (r=0),
  steps keep flowing, both clients back at the final check
  (`opp_back=True`).

## 1on1 learning-signal changes (2026-11-20)

- **Death is a penalty (−0.5), not a terminal state** -- the GRU hidden
  state carries across death->respawn so the bot can learn "where I died /
  where the enemy is"; only the intermission (or a reconnect) ends an
  episode. Previously the bot "forgot" everything every ~170 steps.
- **Velocity only across two ALIVE frames** -- a death->respawn position
  jump no longer produces a one-frame velocity spike in the obs.
- **`dmg` / `kills` per rollout** in `train.py` stats (zeroed after read) --
  the direct learning-signal metric for an untrained-vs-trained comparison.

## TDM 2v2 env (2026-11-20)

Game layer on `DuelEnv` (the port plan's "TDM = a game layer on the
current duel_env.py"): agent (managed, red) + 1 QVM teammate (red) vs 2
QVM opponents (blue) on a second engine (port 28135, home
`~/.q3a_rl_tdm`) so the 28134 duel env can train in parallel.

- **Maps:** the 8-map clan TDM list (user constraint: TDM test matches
  ONLY on pro-q3dm6 q3dm7 ospdm5 ospdm6 cpm4 q3dm14tmp vq2dm1
  ospdm15rc3), all BSP+AAS verified on knecht. q3dm14tmp's AAS is NOT in
  any pak -- it lives in the engine's extracted home cache
  (`~/.q3a/baseq3/q3dm14tmp.2dcfd82c.pk3`), which the start script copies
  into the fresh home's baseq3. Same feeder design as the duel env
  (rotation only; no addbot).
- **Team assignment (proven 2026-11-21 against the running QVM):** the
  botctl ADD "team" field is userinfo METADATA only (botctl.c: the game
  module does the assignment; "the client should pick a team with a
  TEXT command") -> the env sends `team red` via `bc.text()`. **The join
  is only accepted BEFORE the match has started** (fewer than
  match_minplayers=2 players in; `g_doWarmup 0` in the cfg means no
  warmup -- the match starts as soon as 2 players are in). While the
  match is running the running QVM (zz-osp-server4, OSP 1.04 -- NOT the
  1.03a xq3osp source, whose SetTeam has no such gate) rejects it with
  a SINGLE-CLIENT print that the dedicated build never logs
  (SV_SendServerCommand only echoes "broadcast:" prints) -- completely
  silent in the server log. The working moment is the POST-SPAWN
  WINDOW: after InitGame the managed client auto-reconnects faster than
  the 3 QVM bots, so `_step` re-asserts `team red` every 2 s for 30 s
  after a restart (tracked on `_last_restart_id`, NOT `self.prev` -- the
  done branch resets prev to the first new-map frame, so a prev-based
  restart check is dead code), then falls back to the 10 s guard period
  (mid-match re-asserts are harmless silent no-ops). **The re-assert
  is DECOUPLED from the PERS_TEAM gate (2026-11-21, smoke #3):** the
  running QVM can claim `PERS_TEAM=RED` for a client that is NOT
  really in the team -- a fresh ADD with team metadata showed the
  agent "RED" in the snapshot on the first post-restart load while
  the server had it as SPECTATOR (no "entered the game" broadcast,
  no userinfo `t\1`, no sessionTeam; the next load reverted it to 3
  and that window was missed). Gating the re-assert on `me["team"]`
  alone would then skip the one window where the text is needed, and
  a stuck phantom would look like a healthy RED to the obs. During
  the restart window the text now goes out regardless of PERS_TEAM:
  for a REAL red join `SetTeam` is a clean no-op (1.03a g_cmds.c:
  `team == sessionTeam -> return`, no state change), for the phantom
  it performs the join. Success = snapshot team flips to TEAM_RED (1)
  + the "entered the game (RED)" broadcast.
  QVM bots get their team from the rcon addbot 3rd argument
  (`addbot <file> <skill> <team>` -- the 4v4 harness on 28131 runs
  exactly that way); SVF_BOT clients are exempt from the late-join
  gate, so the bots re-enter every map. Snapshot team values: TEAM_FREE
  0, TEAM_RED 1, TEAM_BLUE 2, TEAM_SPECTATOR 3 (game entity team).
  Enemy/teammate selection is judged against the team CONSTANTS, not my
  current team field (a TEAM_FREE/SPECTATOR window must not turn the
  teammate into an "enemy").
- **Opponent ownership:** one guard, three named bots (funnames from
  `scripts/bots.txt` in `zzz_dwc_bots.pk3`: dwcbot2 = DISCO,
  dwcbot7 = [BOT]TrapDou, dwcbot9 = [BOT]RamMan). Per-bot presence from
  `rcon status` (color codes stripped; the snapshot carries no names),
  15 s per-bot absence grace, re-add ONLY the missing bot WITH its team.
  Bots survive same-process map changes (auto-reconnect); unconditional
  re-adds stack duplicates (the duel-env lesson).
- **Sidecar is SINGLE-CONNECTION (engine design):** `Botctl_HandleConnection`
  in the listener thread blocks until the connection closes before the
  next accept. While the training env is connected, a SECOND sidecar
  connect blocks (observed: a one-shot status client hung behind the
  smoke test). State checks while a client is connected: `rcon status`
  over UDP, not a second sidecar client.
- **cfg** (`rl_tdm.cfg`): g_gametype 3, g_friendlyFire 1 (matches the
  production TDM server config), g_teamAutoJoin 0, g_teamForceBalance 0,
  timelimit 10 (60 s wall at TS 10 -> the rotation mechanism works
  unchanged), bot_memory 0, g_spskill 3.
- **Obs (54-dim):** duel's 21 self floats + 9-float relative blocks for
  enemy 1/enemy 2 (nearest blues) + teammate (nearest red) + both team
  scores/100 + dead/intermission + 2 reserved. New net needed (44->54
  input); reward = the duel placeholder, team-attributed (teammate
  damage/kills credit too -- documented). **Team scores sum ALIVE
  players only, SANITY-BOUNDED (2026-11-21):** two garbage families
  both measured in the smokes: (1) dead player blobs carry garbage
  u32 fields (`score` = 4294967297 = 2^32+1 on a dead blue) -- the
  same family as the dead-blob health garbage from the spike; (2)
  freshly RECONNECTED clients carry garbage PERS_SCORE for a few
  frames even while alive (measured 4294967295 = 0xFFFFFFFF on an
  ALIVE blue at the first frames of a load -- the alive-only filter
  does NOT catch that one). Summing them would drive obs[48]/[49] to
  ~4e7 for a few frames. Real TDM scores are < 100 per 60 s match, so
  scores above `SCORE_MAX_SANITY` (10000) are skipped for that frame
  (minor transient undercount << a 4e7 spike). (The in-game
  scoreboard counts dead players' persistent session scores, but the
  botctl snapshot only sees the ps blob; alive-only + bounded is the
  consistent bounded signal.)
- **Intermission stall -- root cause + fix (2026-11-21):** the first
  TDM smoke stalled 100 s in the intermission after its first rotation
  (q3dm7 -> ospdm5, timelimit hit), then cascaded: the env kept reading
  a frozen snapshot blob while the server moved on. Root cause (proven
  in the OSP source, xq3osp): `CheckExit` (g_main.c ~1640-1736) exits
  the intermission only when >= 1 HUMAN client is ready -- `SVF_BOT`
  is skipped in the ready count, and a botctl managed client is
  human-like, so with no ready press `ready=0, notReady=1` and
  `if (!ready && notReady) return;` waits forever. The ready flag is
  set by a RISING BUTTON EDGE in `ClientIntermissionThink`
  (g_active.c:790-808: ATTACK/USE_HOLDABLE, 300 ms debounce) -- the
  TEXT command "ready" does NOT work (`OSP_Cmd_Ready_f`,
  g_osp_cmds.c:653, bails "Match is already in progress!" outside
  warmup). `readyToExit` is reset to 0 by BeginIntermission
  (g_main.c:1335), so one edge per intermission always toggles the
  right way. The DUEL env never stalled only because the training loop
  samples a new action every step and the fire-dim rising edge
  ACCIDENTALLY self-ready'd the agent (89 natural exits in run3b's 2 h
  log); the smoke sent no actions at all -> guaranteed stall. Fix:
  while `self._intermission` is set (TDMEnv._step from the snapshot's
  pm_type), the base repeater (env_duel._repeat_loop) releases all
  buttons on tick 1 and holds ATTACK on every later tick -- exactly
  one rising edge, exit after the 5 s intermission minimum.
- **Done-branch respawn wait + the double-load pattern (2026-11-21,
  smoke #2 forensics):** smoke #2 (a mid-match boot) stayed SPECTATOR
  through rotation #1 and only joined on rotation #2 -- and that only
  because one 2 s re-assert happened to win a narrow window. Cause: the
  done branch's `DuelEnv._wait_alive` waits for the agent to come back
  as PM_NORMAL, which a SPECTATOR agent never reaches in the pre-match
  window (it only becomes PM_NORMAL AFTER `team red` lands) -> the full
  15 s timeout ran on EVERY rotation, freezing the step loop (and the 2
  s fast re-asserts) through the entire post-spawn window (the client
  reconnects +4-8 s after G_Init, the QVM bots +6-10 s) -> the first
  `team red` landed mid-match = silently rejected (measured: 15 s block
  01:23:07->:22 on load #A, text at rel 95, after the bots). Fix: a
  `TDMEnv._wait_alive(ref_snap=...)` override that polls while the
  snapshot's `restart_id` is still the old map's -- that also COLLAPSES
  the ~1-2 s intermission into the single episode boundary (the base
  wait's storm-prevention role; a plain visibility check would return
  immediately, because the client IS visible in the intermission, and
  re-fire done=True on every step until the reload) -- then returns at
  the first frame the client is visible on the NEW map, ANY pm_type
  (RED agents spawn PM_NORMAL as before; spectator agents return as
  TEAM_SPECTATOR with the step loop unblocked, so the fast re-assert
  wins the window deterministically). The duel env keeps the PM_NORMAL
  wait (an FFA agent always reaches it). **Double-load pattern (same
  forensics, systematic, every rotation):** the natural end produces
  TWO full QVM loads 2 s apart -- load #A starts when the 2nd player is
  in, the 3rd QVM bot then joins late, `match_latejoin 0` rejects it ->
  full same-map restart -> load #B is the real 60 s match. Load #B
  re-opens a FRESH join window (the wait fix + 2 s re-asserts cover
  both). Side effect: the nextmap cvar can run one rotation stale if a
  natural end lands between feeder shots (pro-q3dm6 ran twice in a row
  once) -- the feeder tolerates it because its rotation is keyed on the
  OBSERVED live map.
- **Env hardening (2026-11-21, both envs):** (1) STALE WATCHDOG --
  `_snap_until_new` tracks the last seen `snap.time` in instance state;
  no new frame for 35 s (`STALE_SNAP_S`) raises `StaleSnapshotError`
  -> the step() guard does a FORCED re-socket (a plain ping "succeeds"
  on a live-but-stale connection) + re-ADOPT/re-ADD + clean episode
  boundary; the freshness window resets on every reconnect. A full
  spawn freezes frames for <= ~20 s (FS_Startup + QVM recompile), well
  under the window. (2) NON-SILENT re-ADD -- `_ensure_present` no
  longer swallows a failed ADD (3x retry, then raise): the old
  except:pass left a zombie env on a dead slot that kept stepping on
  nothing (the TDM cascade). (3) TDM immediate team re-assert -- on a
  restart_id change the `team red` TEXT goes out on the first frame the
  client is visible again, so the TEAM_FREE window (in which teammate/
  enemy could be misclassified) is ~1-2 frames, not the 10 s guard
  period.
- **Engine-side lifecycle facts (2026-11-20/21, log-verified):** a full
  in-process `map` change does NOT stop botctl (no `botctl: stopped`
  in the 2 h duel log with 89 rotations; only 2 genuine
  conn-loss/re-ADD cycles) and the managed client SURVIVES it
  (auto-reconnect, same slot). `OP_LIST` returns MANAGED clients only
  (QVM bots never appear there -- the rcon-status funname check is
  why the TDM bot guard uses rcon, not the sidecar). `sv.time` is a
  SERVER-WIDE continuous counter, NOT per-level (computing an
  "intermission duration" across a map change from it is meaningless).
  The rcon `status` client lines have LEADING SPACES (`%4i`) --
  `grep "^[0-9] "` misses them all and reports "0 clients"; the
  sidecar snapshot is the ground truth for client presence.

## Never touch

`~/q3e` and `~/.q3a` on the laptop (user's internet-play install),
`~/.q3a_duel` on knecht (botmem duel-training data).
