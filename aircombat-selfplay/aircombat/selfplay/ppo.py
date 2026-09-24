"""PPO の更新。

陣営単位の報酬と価値から GAE で求めた利得を陣営内の各機に共有し、
方策比は機体ごとに取る（MAPPO と同じ考え方）。撃墜された機体のステップは方策の損失から除く。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import OBS_KEYS, PolicyNet, to_tensors
from .rollout import Trajectory


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 512  # 1 ミニバッチの陣営ステップ数
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.05  # これを超えたらそのイテレーションの更新を打ち切る


class PPO:
    def __init__(self, model: PolicyNet, cfg: PPOConfig, device: str = "cpu"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    def update(self, trajs: list[Trajectory], rng: np.random.Generator) -> dict:
        cfg = self.cfg
        obs = {k: np.concatenate([t.obs[k] for t in trajs]) for k in OBS_KEYS}
        actions = np.concatenate([t.actions for t in trajs])
        logp_old = np.concatenate([t.logp for t in trajs])
        alive = np.concatenate([t.alive for t in trajs]).astype(np.float32)
        adv = np.concatenate([t.advantages for t in trajs])
        ret = np.concatenate([t.returns for t in trajs])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(adv)

        self.model.train()
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        stop = False
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                mb = perm[start : start + cfg.minibatch_size]
                t_obs = to_tensors({k: v[mb] for k, v in obs.items()}, self.device)
                t_act = torch.as_tensor(actions[mb], device=self.device)
                t_old = torch.as_tensor(logp_old[mb], device=self.device)
                t_alive = torch.as_tensor(alive[mb], device=self.device)
                t_adv = torch.as_tensor(adv[mb], device=self.device).unsqueeze(-1)
                t_ret = torch.as_tensor(ret[mb], device=self.device)

                logits, value = self.model(t_obs)
                logp, ent, _ = self.model.distribution_stats(logits, t_act)
                log_ratio = logp - t_old
                ratio = log_ratio.exp()
                surr = torch.min(ratio * t_adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * t_adv)
                denom = t_alive.sum().clamp(min=1.0)
                policy_loss = -(surr * t_alive).sum() / denom
                entropy = (ent * t_alive).sum() / denom
                value_loss = ((value - t_ret) ** 2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = (((ratio - 1) - log_ratio) * t_alive).sum() / denom
                    clip_frac = (((ratio - 1).abs() > cfg.clip).float() * t_alive).sum() / denom
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(kl.item())
                stats["clip_frac"].append(clip_frac.item())
                if cfg.target_kl and kl.item() > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.model.eval()
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        out["samples"] = int(n)
        out["early_stop"] = stop
        # 価値関数の当てはまり（1 に近いほど良い）
        values = np.concatenate([t.values for t in trajs])
        var = np.var(ret)
        out["explained_var"] = float(1 - np.var(ret - values) / var) if var > 1e-8 else 0.0
        return out
