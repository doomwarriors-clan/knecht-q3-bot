"""
ppo.py -- minimal PPO (GAE, clipped surrogate) for the duel PoC.

Squashed-Gaussian actions: a = tanh(z), log p(a) = N(z|mu,sigma).logprob
- log(1 - a^2). The GRU hidden state is stored per step and used as a fixed
input on recomputation (no BPTT across the rollout in this PoC).
"""

import numpy as np
import torch
import torch.nn.functional as F


def gather(policy, obs, h):
    mu, logstd, v, _ = policy(obs, h)
    return mu, logstd, v


def sample(policy, obs, h, deterministic=False):
    """Single rollout step. Returns (a, logp, v, h_new)."""
    mu, logstd, v, h_new = policy(obs, h)
    std = logstd.exp()
    if deterministic:
        a = torch.tanh(mu)
        logp = torch.zeros(a.shape[0], device=mu.device)
    else:
        z = mu + std * torch.randn_like(mu)
        a = torch.tanh(z)
        logp = (
            torch.distributions.Normal(mu, std).log_prob(z)
            - torch.log(1.0 - a * a + 1e-6)
        ).sum(-1)
    return a, logp, v, h_new


def squashed_logprob(mu, std, a):
    """Log-prob of squashed action a = tanh(z) under N(mu, std)."""
    z = torch.atanh(torch.clamp(a, -0.999, 0.999))
    lp = torch.distributions.Normal(mu, std).log_prob(z)
    return (lp - torch.log(1.0 - a * a + 1e-6)).sum(-1)


def ppo_update(policy, opt, buf, dev,
               epochs=4, batch=256, clip=0.2, gamma=0.99, lam=0.95,
               vf_coef=0.5, ent_coef=0.01):
    """Update on a buffer of dicts: obs (D,), h (H,), act (A,), logp, v, r, done."""
    n = len(buf)
    obs = torch.from_numpy(np.ascontiguousarray([b["obs"] for b in buf])).to(dev)
    h = torch.from_numpy(np.ascontiguousarray([b["h"] for b in buf])).to(dev)
    acts = torch.from_numpy(np.ascontiguousarray([b["act"] for b in buf])).to(dev)
    logps_old = torch.tensor([b["logp"] for b in buf], dtype=torch.float32, device=dev)
    rs = torch.tensor([b["r"] for b in buf], dtype=torch.float32, device=dev)
    dones = torch.tensor([b["done"] for b in buf], dtype=torch.float32, device=dev)

    # ---- GAE
    with torch.no_grad():
        _, _, v = gather(policy, obs, h)
        vals = v.squeeze(-1)
        adv = torch.zeros(n, device=dev)
        last = 0.0
        for t in range(n - 1, -1, -1):
            next_v = vals[t + 1] if t + 1 < n else 0.0
            delta = rs[t] + gamma * next_v * (1.0 - dones[t]) - vals[t]
            last = delta + gamma * lam * (1.0 - dones[t]) * last
            adv[t] = last
        rets = adv + vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ---- clipped PPO
    policy.train()
    n_upd = 0
    tot_loss = 0.0
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, batch):
            bidx = torch.tensor(idx[s:s + batch], dtype=torch.long, device=dev)
            mu, logstd, v = gather(policy, obs[bidx], h[bidx])
            std = logstd.exp()
            logp = squashed_logprob(mu, std, acts[bidx])
            ratio = torch.exp(logp - logps_old[bidx])
            s1 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[bidx]
            s2 = ratio * adv[bidx]
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(v.squeeze(-1), rets[bidx])
            ent = torch.distributions.Normal(mu, std).entropy().sum(-1).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_upd += 1
            tot_loss += loss.item()
    policy.eval()
    return tot_loss / max(1, n_upd)
