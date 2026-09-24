"""模倣学習（行動クローニング）による初期化。

ゼロからの強化学習は、遠距離から誘導弾を撃ち尽くす段階を抜けるまでに時間がかかる。
そこでまずルールベースのエージェント（教師）の行動を真似るようにネットワークを学習し、
その重みから自己対戦の強化学習を始める（aircombat train --init）。

- 教師と対戦相手を組み合わせて対戦させ、教師側の観測と行動を記録する
- 方策は各行動ヘッドの交差エントロピー、価値は割引収益の二乗誤差で学習する
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..agents import make_agent
from ..config import EnvConfig
from ..env import AirCombatEnv
from .model import OBS_KEYS, ModelConfig, PolicyNet, obs_to_arrays, save_checkpoint, to_tensors


@dataclass
class ImitationConfig:
    teacher: str = "rule"
    opponents: list[str] = field(
        default_factory=lambda: ["rule", "rule_defensive", "rule_aggressive", "random", "straight"]
    )
    episodes: int = 120
    stride: int = 2  # 何ステップごとに記録するか（連続するステップはよく似ているので間引く。射撃した場面は必ず記録）
    fire_weight: float = 10.0  # 教師が撃った場面の射撃ヘッドの損失の重み（撃つ場面はまれなので強調する）
    epochs: int = 6
    batch_size: int = 512
    lr: float = 1e-3
    value_coef: float = 0.5
    gamma: float = 0.995  # 価値の学習に使う割引率（強化学習と揃える）
    hidden: int = 128
    workers: int = 3
    seed: int = 0


def _discounted(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


_ENV: AirCombatEnv | None = None


def _init(env_cfg: dict) -> None:
    global _ENV
    torch.set_num_threads(1)
    _ENV = AirCombatEnv(EnvConfig.from_dict(env_cfg))


def _collect(args) -> dict:
    """1 対戦を行い、教師側（両陣営とも教師なら両方）の観測・行動・割引収益を返す。"""
    teacher, opponent, seed, teacher_team, stride, gamma = args
    env = _ENV
    agents = {teacher_team: make_agent(teacher), 1 - teacher_team: make_agent(opponent)}
    record = [teacher_team] + ([1 - teacher_team] if opponent == teacher else [])
    obs = env.reset(seed=seed)
    for k, a in agents.items():
        a.reset(k, env.cfg, seed=seed * 2 + k)
    buf = {k: {"obs": [], "act": [], "rew": []} for k in record}
    done = False
    while not done:
        actions = {k: agents[k].act(obs[k]) for k in (0, 1)}
        for k in record:
            buf[k]["obs"].append(obs_to_arrays(obs[k]))
            buf[k]["act"].append(np.asarray(actions[k], dtype=np.int64))
        obs, rewards, done, info = env.step(actions)
        for k in record:
            buf[k]["rew"].append(float(rewards[k]))
    out = {k: [] for k in OBS_KEYS}
    out["act"], out["ret"], out["alive"] = [], [], []
    for k in record:
        ret = _discounted(np.asarray(buf[k]["rew"], dtype=np.float32), gamma)
        for t in range(len(ret)):
            if t % stride != 0 and not np.any(buf[k]["act"][t][:, 3] > 0):
                continue
            o = buf[k]["obs"][t]
            for key in OBS_KEYS:
                v = o[key]
                out[key].append(v if key == "mask" else v.astype(np.float16))
            out["act"].append(buf[k]["act"][t])
            out["ret"].append(ret[t])
            out["alive"].append(o["self"][:, 0] > 0.5)
    return {
        "data": {k: np.stack(v) for k, v in out.items()},
        "score": info["outcome"].scores[teacher_team],
        "steps": len(buf[teacher_team]["rew"]),
    }


def collect_demonstrations(env_cfg: EnvConfig, cfg: ImitationConfig, log=print) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    tasks = [
        (
            cfg.teacher,
            cfg.opponents[i % len(cfg.opponents)],
            int(rng.integers(2**31 - 1)),
            int(rng.integers(2)),
            cfg.stride,
            cfg.gamma,
        )
        for i in range(cfg.episodes)
    ]
    t0 = time.time()
    if cfg.workers > 0:
        with mp.get_context("spawn").Pool(cfg.workers, initializer=_init, initargs=(env_cfg.to_dict(),)) as pool:
            results = pool.map(_collect, tasks, chunksize=1)
    else:
        _init(env_cfg.to_dict())
        results = [_collect(t) for t in tasks]
    data = {k: np.concatenate([r["data"][k] for r in results]) for k in results[0]["data"]}
    log(
        f"教師 {cfg.teacher} の対戦 {len(results)} 回（平均スコア {np.mean([r['score'] for r in results]):.3f}）から "
        f"{len(data['act'])} サンプルを収集しました（{time.time() - t0:.0f} 秒）"
    )
    return data


def train_imitation(
    data: dict[str, np.ndarray], model: PolicyNet, cfg: ImitationConfig, log=print
) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    n = len(data["act"])
    perm = rng.permutation(n)
    n_val = max(1, n // 20)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs * (len(train_idx) // cfg.batch_size + 1)))

    def batch(idx):
        obs = to_tensors({k: data[k][idx] for k in OBS_KEYS})
        return obs, torch.as_tensor(data["act"][idx]), torch.as_tensor(data["ret"][idx]), torch.as_tensor(
            data["alive"][idx], dtype=torch.float32
        )

    def losses(idx):
        obs, act, ret, alive = batch(idx)
        logits, value = model(obs)
        ce = 0.0
        correct = []
        denom = alive.sum().clamp(min=1.0)
        for h, lg in enumerate(logits):
            logp = F.log_softmax(lg, dim=-1)
            nll = -logp.gather(-1, act[..., h : h + 1]).squeeze(-1)
            w = alive
            if h == 3:
                w = alive * torch.where(act[..., 3] > 0, cfg.fire_weight, 1.0)
            ce = ce + (nll * w).sum() / w.sum().clamp(min=1.0)
            correct.append(((lg.argmax(-1) == act[..., h]).float() * alive).sum() / denom)
        vloss = ((value - ret) ** 2).mean()
        # 射撃はほとんどのステップで「撃たない」なので、教師が撃った場面での一致率も見る
        fired = (act[..., 3] > 0).float() * alive
        recall = ((logits[3].argmax(-1) == act[..., 3]).float() * fired).sum() / fired.sum().clamp(min=1.0)
        correct.append(recall)
        return ce, vloss, torch.stack(correct)

    stats = {}
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(train_idx)
        tot = []
        for s in range(0, len(order), cfg.batch_size):
            ce, vloss, _ = losses(order[s : s + cfg.batch_size])
            loss = ce + cfg.value_coef * vloss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot.append(loss.item())
        model.eval()
        with torch.no_grad():
            accs, vls = [], []
            for s in range(0, len(val_idx), cfg.batch_size):
                _, vloss, acc = losses(val_idx[s : s + cfg.batch_size])
                accs.append(acc.numpy())
                vls.append(vloss.item())
        acc = np.mean(accs, axis=0)
        stats = {
            "train_loss": float(np.mean(tot)),
            "val_value_loss": float(np.mean(vls)),
            "val_acc_turn": float(acc[0]),
            "val_acc_pitch": float(acc[1]),
            "val_acc_throttle": float(acc[2]),
            "val_acc_fire": float(acc[3]),
            "val_fire_recall": float(acc[4]),
        }
        log(
            f"epoch {epoch + 1}/{cfg.epochs}  loss {stats['train_loss']:.3f}  "
            f"一致率 旋回 {acc[0]:.2f} 経路角 {acc[1]:.2f} スロットル {acc[2]:.2f} 射撃 {acc[3]:.3f}"
            f"（撃った場面 {acc[4]:.2f}）  "
            f"価値誤差 {stats['val_value_loss']:.4f}"
        )
    return stats


def pretrain(env_cfg: EnvConfig, cfg: ImitationConfig, out: str | Path, log=print) -> Path:
    torch.manual_seed(cfg.seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = collect_demonstrations(env_cfg, cfg, log)
    model = PolicyNet(ModelConfig(num_fighters=env_cfg.scenario.num_fighters, hidden=cfg.hidden))
    stats = train_imitation(data, model, cfg, log)
    model.eval()
    save_checkpoint(out, model, env_cfg.to_dict(), {"imitation": asdict(cfg), "imitation_stats": stats})
    log(f"保存しました: {out}")
    return out
