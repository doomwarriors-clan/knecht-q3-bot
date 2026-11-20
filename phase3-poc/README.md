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
- Server: FFA duel (g_gametype 1) on q3dm17, opponent = `dwcbot5` (skill 5,
  "NUDELpas") from `zzz_dwc_bots.pk3`. While `sv_timescale > 1` the engine
  rejects all real clients (bots-only). Start/stop: `~/tmp/rl/` on knecht
  (`start_rl_env.sh` / `stop_rl_env.sh`).
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

## Never touch

`~/q3e` and `~/.q3a` on the laptop (user's internet-play install),
`~/.q3a_duel` on knecht (botmem duel-training data).
