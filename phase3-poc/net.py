"""
net.py -- policy/value network for the duel PoC.

Architecture (workload representative of the real BobbyBones-style net,
scaled to ~1.1 M params): MLP encoder (44 -> 512 -> 512, tanh) -> single-step
GRU (512) -> Gaussian action head (5) + value head (1). The GRU hidden state
is carried by the caller (reset per episode); PPO recomputes each step from
the stored hidden state (no BPTT through the chain in this PoC).
"""

import torch
import torch.nn as nn


class DuelPolicy(nn.Module):
    def __init__(self, obs_dim=44, act_dim=5, hidden=512):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head_mu = nn.Linear(hidden, act_dim)
        self.head_v = nn.Linear(hidden, 1)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs, h):
        """obs: (B, D), h: (B, H) -> (mu (B,A), logstd (A,), v (B,1), h_new (B,H))"""
        e = self.enc(obs)
        out, h_new = self.gru(e.unsqueeze(1), h.unsqueeze(0))
        out = out.squeeze(1)
        mu = self.head_mu(out)
        v = self.head_v(out)
        return mu, self.logstd, v, h_new.squeeze(0)
