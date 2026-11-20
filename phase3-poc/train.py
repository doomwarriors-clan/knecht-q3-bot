"""
train.py -- PPO training loop for the duel PoC (GPU: laptop RTX 2080).

Usage:
  .venv/bin/python train.py --minutes 5 --name run1
  (optional: --sock <unix socket> --rollout 2048 --lr 3e-4 --seed 1
             --load run1_policy.pt)

The env (engine on knecht, botctl socket SSH-forwarded) paces the rollout
at the engine tick rate (40 steps/s at sv_fps 40). Each rollout: 2048 steps,
then 4 PPO epochs. Stats -> stats_<name>.jsonl, policy -> <name>_policy.pt,
pid -> <name>.pid (read by impact.py).

NOTE (PoC scope): placeholder reward, 5-dim action, weapon pinned to MG,
no BPTT through the GRU chain. This run measures the WORKLOAD of a real
PPO setup on the laptop, not learning quality.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_duel import DuelEnv, OBS_DIM, ACT_DIM  # noqa: E402
from net import DuelPolicy  # noqa: E402
from ppo import sample, ppo_update  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default=os.path.expanduser("~/tmp/rl-lab/botctl-28134.sock"))
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="run1")
    ap.add_argument("--load", default="")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    with open(f"{args.name}.pid", "w") as f:
        f.write(str(os.getpid()))

    env = DuelEnv(args.sock)
    policy = DuelPolicy(OBS_DIM, ACT_DIM).to(dev)
    if args.load:
        policy.load_state_dict(torch.load(args.load, map_location=dev))
        print(f"loaded policy from {args.load}", flush=True)
    n_params = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"policy: {n_params/1e6:.2f} M params on {dev} | socket {args.sock}", flush=True)

    R = args.rollout
    H = policy.hidden
    t_end = time.time() + args.minutes * 60.0
    it = 0
    tot_steps = 0
    stats_f = open(f"stats_{args.name}.jsonl", "a")
    policy.eval()

    obs_now = env.reset()
    h = None
    while time.time() < t_end:
        it += 1
        t0 = time.time()
        buf = []
        with torch.no_grad():
            for _ in range(R):
                if time.time() > t_end:
                    break
                # 1) policy step on the current observation
                obs_t = torch.from_numpy(obs_now).unsqueeze(0).to(dev)
                h_t = (torch.zeros(1, H, device=dev) if h is None
                       else torch.from_numpy(h).unsqueeze(0).to(dev))
                a, logp, v, h_new = sample(policy, obs_t, h_t)
                a_np = a.squeeze(0).cpu().numpy()
                h_in = np.zeros(H, dtype=np.float32) if h is None else h

                # 2) stage the action, let the engine take the frame
                env.send_action(a_np)
                obs_next, r, done = env.step()

                buf.append(dict(
                    obs=obs_now,
                    h=h_in,
                    act=a_np,
                    logp=float(logp.item()),
                    v=float(v.item()),
                    r=float(r),
                    done=bool(done),
                ))
                tot_steps += 1
                obs_now = obs_next
                h = None if done else h_new[0].cpu().numpy()
        collect_s = time.time() - t0

        # 3) PPO update on the GPU
        t1 = time.time()
        loss = ppo_update(policy, opt, buf, dev)
        update_s = time.time() - t1

        acts = np.array([b["act"] for b in buf])
        rewards = np.array([b["r"] for b in buf])
        dones = np.array([b["done"] for b in buf])
        row = dict(
            it=it, steps=len(buf), collect_s=round(collect_s, 1),
            update_s=round(update_s, 2),
            rate=round(len(buf) / max(1e-6, collect_s), 1),
            loss=round(loss, 4),
            rew=round(float(rewards.mean()), 4),
            done_frac=round(float(dones.mean()), 4),
            fire=round(float((acts[:, 4] > 0).mean()), 3),
            act_abs=round(float(np.abs(acts).mean()), 3),
            tot_steps=tot_steps,
            gpu_mem_mib=int(torch.cuda.max_memory_allocated() // (1024 * 1024)),
        )
        stats_f.write(json.dumps(row) + "\n")
        stats_f.flush()
        print(f"[{args.name} it {it}] {row}", flush=True)
        torch.save(policy.state_dict(), f"{args.name}_policy.pt")

    env.close()
    stats_f.close()
    print(f"done: {tot_steps} steps over {it} rollouts; policy -> {args.name}_policy.pt",
          flush=True)


if __name__ == "__main__":
    main()
