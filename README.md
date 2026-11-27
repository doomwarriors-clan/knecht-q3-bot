# knecht-q3-bot — the DoomWarriors Quake 3 port of BobbyBones

A fork of [BobbyBones](https://github.com/dclark202/bobbybones-quake), the Quake Live duel bot that was trained from scratch by reinforcement learning. BobbyBones plays Quake Live through a server plugin; **this project ports the same approach to Quake 3**: the clan runs the [Quake3e](https://github.com/doomwarriors-clan/Quake3e) engine (`dwc` branch) with the [OSP](https://github.com/doomwarriors-clan/osp) game mod, and the goal is a bot that learns our duel and TDM maps by playing against our own clan botfiles — with human-like senses only (field of view, line of sight, no wall hacks).

## Status (2026-11-27)

| Milestone | State |
|---|---|
| **Phase 0 — spike:** prove an external process can drive a bot client slot in Quake3e + OSP **per frame** (usercmds incl. substeps, game-state read-back, lifecycle edges like kicks / map rotation / sidecar death) | **DONE** — scenario suite T1–T10, final matrix **12/12 green** on the merged engine (2026-11-20); see `spike/` + `docs/spike/SPIKE_ENGINE.md` |
| Engine-side `botctl` control channel (Unix socket, **off by default**, `botctl_enable 0`) | **merged** into [doomwarriors-clan/Quake3e](https://github.com/doomwarriors-clan/Quake3e) `dwc` (PR #13), shipped in release `dwc-2026.11.20` |
| **Training PoC:** PPO on live dedicated duel servers against clan botfiles | **IN PROGRESS** — telemetry fix landed and verified; honest baseline measured (see findings); **run8 full-budget run (1200 games/env) live since 2026-11-27 08:03** with a review gate at 400 games/env. Canonical training stack lives in the private Forgejo `homelab` repo (`rl/`, run8 + README recipe); this repo's `phase3-poc` branch is being synced from it |
| TDM 2v2 env (8 clan TDM maps) | built + smoke-verified, idle until the duel learning signal matures |
| Sim parity (port the upstream `sim/` approach to Quake 3) | not started — next after the training PoC |
| Live deploy on production servers | **open clan decision** — only after a trained net proves itself; the engine channel stays off by default until then |

## Training findings (2026-11-26/27)

- **run6 (duel vs skill-5 clan botfiles, ~4 M steps, multi-env PPO v2):** flat — K/D stuck at ~0.04 across six 200-rollout buckets, no skill trend; more wall-time alone did not help. Decision: change the learning setup, not the budget.
- **run7 v2.3 (launched 2026-11-26):** 3 parallel duel envs (Sarge / Doom / Daemia at skill 2 — opponent curriculum), frag-dominant reward v3 (kill +10 / death −3 / intermission outcome ±20/5), multi-env PPO learner (`ppo3`/`train3`), policy initialized from run6, fresh value head. Infrastructure ran green (watchdog, per-env fault isolation, ~150–190 steps/s).
- **Telemetry finding (early review after 622 games):** the per-game log recorded **0 frags / 0 wins**, while the server's own `games.log` showed real kills — and a naturally finished duel confirmed the engine scoreboard also saw almost none (17 RL-bot kills in ~209 games vs skill 2; ~390 damage dealt per game without closing kills). Root causes found: the duel servers' **map feeder force-cuts games** (`nextmap`/watchdog `force map` every ~150 s — those games write no `Exit:`/`score:` lines, so outcome detection and score read-back operate on garbage boundaries), and the env's kill detection relies on a snapshot alive→dead edge that does not fire reliably under those conditions. **Consequence: the +10 kill reward was almost never delivered — the run optimized damage/survival shaping but was never reinforced for finishing kills.**
- **Ground-truth reconciliation harness built and fix verified (2026-11-26):** `rl/harness_kill.py` + `rl/harness_server.sh` (dedicated duel server with **natural game endings**, no map feeder) reconcile env counters against `Kill:` lines in `osp/games.log` segment by segment. **Mechanical root cause found:** the botctl protocol client parsed the player block **unsigned** — the engine's score field legitimately carries `-1`, which arrived as `4294967295` and was discarded by every sanity filter, so the RL bot's own score/frags were structurally unreadable. Fix: signed parsing (`botctl.py`) + the bot's own `PERS_SCORE` channel is now the authoritative frag counter (`env_duel7.py`; opponent-death edges downgraded to diagnostics). Verified against ground truth: 12-game lethal-policy run 1 frag env == 1 log kill, 11/11 segments exact; 18-game run vs skill-0 opponent 4 == 4, 17/17 exact; live-server smoke PASS. Canonical: Forgejo `homelab` `main` (`bbda6ba`, `5444787`).
- **run8 validation window (2026-11-27, 300 games):** launched fresh from `run6_policy.pt` with a reset value head and the fixed telemetry — and it measured the first **honest** baseline: **0 kills in 300 games vs skill 2** (K/D 0.000; 1398 opponent frags, 1619 deaths, 86.8 k damage dealt; 297 loss / 3 anomalous wins). Cross-checked against the servers' own `games.log`: zero `Killed by rlbot8*` lines — the env now reports the truth. Conclusion: the telemetry was blind **and** the policy genuinely cannot finish kills; run6/run7's apparent progress was partly a measurement artifact. The `score_me` channel also reads slightly negative (engine's `-1` init per spawn) while all opponent-side channels are exact — cosmetic, tracked.
- **Training servers switched to natural game endings (2026-11-27):** `rl/start_rl_duel7.sh` reworked — `fraglimit 5` + `timelimit 10`, map rotation only after a natural `Exit:` (OSP `nextmap` path), idle guard `map` at most once per 900 s window (observed: fires only while no client is connected). Side finding documented: a forced `map` kicks the `addbot` opponent; the env's `_ensure_opponent` re-adds it within ~15 s. Smoke on all three servers green; watchdog `scripts/run7_healthcheck.py --run <runname>` now parameterized per run (`run8` -> bots `rlbot8a/b/c`). Commits: Forgejo `5161ba3`, `02d6f26`, `717aff6`, `e51bef4`.
- **Next steps:** run8 full budget (1200 games/env, live since 08:03) with a **review gate at 400 games/env**: if even one kill appears, the +10 signal has a foothold; if still 0, switch strategy — either an opponent-skill warmup (skill 0/1 where the bot can experience kills) or a behavioural-cloning bootstrap from the heuristic policy (which demonstrably kills: 4 kills/18 games vs skill 0 in the harness). Curriculum gate (skill 2 → 3) stays at K/D ≥ 0.5. Network architecture (GRU policy + PPO, 1.89 M params) is unchanged so far; checkpoints remain usable as `--load` starting points.

## How it fits together

```
training process (Python, consumer GPU)
   │  observations (game-state snapshots)        usercmds every frame (+ substeps)
   ▼                                             ▲
botctl Unix-socket channel  ────────  Quake3e dedicated server (dwc build, USE_BOTCTL=1)
   ▲                                  + OSP QVM + clan botfiles
   └─ rcon: swaps the opponent bot at episode boundaries
```

The bot occupies a **regular client slot** (it shows up in `status`, joins teams, scores like a player) but is driven frame-by-frame from outside the engine — no scripting embedded in the engine, no changes to the game QVM. The spike proved the whole channel incl. its failure modes; the training PoC reuses the exact same protocol.

## What lives where in this repo

- `spike/` — Phase 0 client side: `botctl.py` (stdlib protocol client), `sidecar.py` (scenario brains T1–T10), `run_spike.sh` (spike driver), `rcon_probe.py`
- `docs/spike/` — spike design (V1/V2), measurement methodology, the full 12/12 results matrix, and the archived engine issue/PR text (`QUAKE3E_ISSUE.md`)
- `phase3-poc` (branch, merge pending) — the RL training stack: duel / duel-v2 / duel7 environments on the botctl channel, PPO (`ppo2`/`ppo3`), policy nets (`net2`), training loops (`train2`/`train3`), smoke + recovery + rotation tests, server start/stop scripts. Being refreshed from the canonical Forgejo `homelab` `rl/` tree (run7 v2.3 + ground-truth harness) before its merge
- everything else (`sim/`, `tools/`, `plugins/`, `minqlx/`, `server/`, …) — the original BobbyBones Quake Live project, kept untouched as design reference

## Relationship to upstream

Upstream BobbyBones keeps moving fast, for **Quake Live**. We treat its code as a reference, keep all clan work **purely additive** (new directories only, no edits to upstream files), and pull upstream improvements in as periodic **upstream-sync merge PRs** into `main` — next one planned after the `phase3-poc` merge, and always refreshed before the sim-parity phase starts. QL-specific parts (minqlx plugin, QL maps and physics, QL sim tuning) are never ported blindly: each is evaluated against Quake 3 mechanics when its phase begins.

## Related clan repos

- [Quake3e](https://github.com/doomwarriors-clan/Quake3e) (`dwc`) — the engine incl. the `botctl` half of this project
- [osp](https://github.com/doomwarriors-clan/osp) — the game mod on our servers (needs no botctl changes)
- [dwc-bots](https://github.com/doomwarriors-clan/dwc-bots) — the clan botfiles used as training opponents
- [q3-servers](https://github.com/doomwarriors-clan/q3-servers) — production server configs (a later live integration would touch this)

## License

Upstream code in this repository is covered by the [GPL-3.0](LICENSE) license (`LICENSE`); `WEIGHTS_LICENSE.md` covers upstream's published Quake Live model weights. The engine-side `botctl` code lives in the Quake3e fork under its GPL terms.
