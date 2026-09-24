"""方策・価値ネットワーク（PyTorch）。

1 陣営の戦闘機 nf 機をまとめて入力し、機体ごとの行動分布と陣営の価値を出す。
- 自機・味方・敵・誘導弾警報をそれぞれ MLP で埋め込み、自機を query にした注意機構で集約
- 陣営内の全機の埋め込みの平均を各機に加えて連携できるようにする（1 エージェントで全機を操縦する想定）
- 射撃ヘッドは「撃たない」と敵スロットごとのロジットを出すポインタ型
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..obs import ALLY_DIM, ENEMY_DIM, MWS_DIM, SELF_DIM, TeamObs, action_dims

OBS_KEYS = ("self", "ally", "enemy", "mws", "mask")


@dataclass
class ModelConfig:
    num_fighters: int = 4
    hidden: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


def obs_to_arrays(obs: TeamObs) -> dict[str, np.ndarray]:
    return {
        "self": obs.self_feats,
        "ally": obs.ally_feats,
        "enemy": obs.enemy_feats,
        "mws": obs.mws_feats,
        "mask": obs.action_mask,
    }


def stack_obs(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([it[k] for it in items]) for k in OBS_KEYS}


def to_tensors(arrs: dict[str, np.ndarray], device="cpu") -> dict[str, torch.Tensor]:
    out = {}
    for k, v in arrs.items():
        if k == "mask":
            out[k] = torch.as_tensor(v, dtype=torch.bool, device=device)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o), nn.SiLU())


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, query: torch.Tensor, ents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # query (N, H), ents (N, K, H), mask (N, K)
        scores = (self.q(query).unsqueeze(1) * self.k(ents)).sum(-1) * self.scale
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1) * mask.any(-1, keepdim=True)
        return (w.unsqueeze(-1) * self.v(ents)).sum(1)


class PolicyNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        self.dims = action_dims(cfg.num_fighters)
        self.self_enc = _mlp(SELF_DIM, H, H)
        self.ally_enc = _mlp(ALLY_DIM, H, H)
        self.enemy_enc = _mlp(ENEMY_DIM, H, H)
        self.mws_enc = _mlp(MWS_DIM, H // 2, H)
        self.ally_pool = MaskedAttentionPool(H)
        self.enemy_pool = MaskedAttentionPool(H)
        self.trunk = nn.Sequential(nn.Linear(4 * H, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.team_mix = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU())
        self.turn = nn.Linear(H, self.dims[0])
        self.pitch = nn.Linear(H, self.dims[1])
        self.throttle = nn.Linear(H, self.dims[2])
        self.fire_none = nn.Linear(H, 1)
        self.fire_slot = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, 1))
        self.value = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, 1))
        for head in (self.turn, self.pitch, self.throttle, self.fire_none):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """ロジット [旋回, 経路角, スロットル, 射撃]（各 (B, nf, n)、無効な行動は -1e9）と価値 (B,) を返す。"""
        s_in = obs["self"]
        B, nf = s_in.shape[:2]
        N = B * nf
        s = self.self_enc(s_in).reshape(N, -1)
        a_in = obs["ally"].reshape(N, -1, ALLY_DIM)
        e_in = obs["enemy"].reshape(N, -1, ENEMY_DIM)
        m_in = obs["mws"].reshape(N, -1, MWS_DIM)
        a = self.ally_enc(a_in)
        e = self.enemy_enc(e_in)
        m = self.mws_enc(m_in)
        a_mask = a_in[..., 0] > 0.5
        e_mask = e_in[..., 0] > 0.5
        m_mask = (m_in[..., 0] > 0.5).float().unsqueeze(-1)
        a_pool = self.ally_pool(s, a, a_mask)
        e_pool = self.enemy_pool(s, e, e_mask)
        m_pool = (m * m_mask).sum(1) / m_mask.sum(1).clamp(min=1.0)
        h = self.trunk(torch.cat([s, a_pool, e_pool, m_pool], dim=-1)).reshape(B, nf, -1)
        team = h.mean(dim=1, keepdim=True).expand(-1, nf, -1)
        hc = self.team_mix(torch.cat([h, team], dim=-1))

        ne = e.shape[1]
        slot_in = torch.cat([hc.reshape(N, 1, -1).expand(-1, ne, -1), e], dim=-1)
        fire = torch.cat([self.fire_none(hc), self.fire_slot(slot_in).reshape(B, nf, ne)], dim=-1)
        logits = [self.turn(hc), self.pitch(hc), self.throttle(hc), fire]

        mask = obs["mask"]
        out = []
        off = 0
        for lg, d in zip(logits, self.dims):
            out.append(lg.masked_fill(~mask[..., off : off + d], -1e9))
            off += d
        value = self.value(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)).squeeze(-1)
        return out, value

    @staticmethod
    def distribution_stats(logits: list[torch.Tensor], actions: torch.Tensor | None = None):
        """機体ごとの対数尤度（各ヘッドの和）、ヘッドごとのエントロピー (..., 4)、行動を返す。

        actions が None ならサンプルする。
        """
        logps, ents, acts = [], [], []
        for h, lg in enumerate(logits):
            logp_all = F.log_softmax(lg, dim=-1)
            if actions is None:
                a = torch.distributions.Categorical(logits=lg).sample()
            else:
                a = actions[..., h]
            acts.append(a)
            logps.append(logp_all.gather(-1, a.unsqueeze(-1)).squeeze(-1))
            p = logp_all.exp()
            ents.append(-(p * torch.where(p > 0, logp_all, torch.zeros_like(logp_all))).sum(-1))
        return torch.stack(logps, -1).sum(-1), torch.stack(ents, -1), torch.stack(acts, -1)

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        """行動 (B, nf, 4)、機体ごとの対数尤度 (B, nf)、価値 (B,) を返す。"""
        logits, value = self(obs)
        if deterministic:
            actions = torch.stack([lg.argmax(-1) for lg in logits], dim=-1)
            logp, _, _ = self.distribution_stats(logits, actions)
        else:
            logp, _, actions = self.distribution_stats(logits)
        return actions, logp, value


def anchor_kl(anchor_logits: list[torch.Tensor], logits: list[torch.Tensor]) -> torch.Tensor:
    """KL(アンカー方策 || 現在の方策) を機体ごと（各ヘッドの和）に返す。"""
    total = 0.0
    for a, lg in zip(anchor_logits, logits):
        pa = F.softmax(a, dim=-1)
        total = total + (pa * (F.log_softmax(a, dim=-1) - F.log_softmax(lg, dim=-1))).sum(-1)
    return total


def save_checkpoint(path, model: PolicyNet, env_cfg: dict, extra: dict | None = None) -> None:
    torch.save(
        {
            "model_cfg": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
            "env_cfg": env_cfg,
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path, map_location="cpu") -> tuple[PolicyNet, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = PolicyNet(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
