"""自己対戦による強化学習（PPO + リーグ）。

1 イテレーションの流れ:
1. リーグから episodes_per_iter 回分の対戦相手を抽選（最新の自分 / 固定の相手 / 過去のスナップショット）
2. ワーカープロセスで対戦を実行して軌跡を集める
3. PPO で方策を更新し、対戦結果でリーグの勝率推定と Glicko-2 レーティングを更新
4. 一定間隔でスナップショットをリーグに追加、固定の相手に対する評価、リプレイの保存

run_dir には以下が保存される:
- config.json: 使った設定
- checkpoint.pt: 再開用（モデル・最適化器・リーグ・乱数）
- latest.pt / best.pt: エージェントとしてそのまま使えるモデル（aircombat match などに渡せる）
- snapshots/iter_XXXXXX.pt: リーグに入れたスナップショット
- metrics.jsonl: イテレーションごとの統計
- replays/iter_XXXXXX.html: 評価対戦のリプレイ
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..agents import make_agent
from ..config import EnvConfig, merge_into_dataclass
from ..env import AirCombatEnv
from ..match import run_match, save_replay
from .league import LEARNER, League, LeagueConfig
from .model import ModelConfig, PolicyNet, load_checkpoint, save_checkpoint
from .policy_agent import PolicyAgent
from .ppo import PPO, PPOConfig
from .rollout import EpisodeResult, EpisodeSpec, RolloutWorker, _init_worker, _run_task


@dataclass
class TrainConfig:
    iterations: int = 300
    episodes_per_iter: int = 32
    num_workers: int = 3  # 0 ならメインプロセスで実行
    envs_per_worker: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    hidden: int = 128
    seed: int = 0
    snapshot_interval: int = 10
    eval_interval: int = 10
    eval_episodes: int = 16  # 評価相手 1 体あたりの対戦数（陣営を交互に入れ替える）
    eval_opponents: list[str] = field(default_factory=lambda: ["rule"])
    eval_deterministic: bool = False  # 評価で学習者に最も確率の高い行動を取らせる
    replay_interval: int = 50
    init_checkpoint: str | None = None  # 学習済みモデルから始める場合のパス
    ppo: PPOConfig = field(default_factory=PPOConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrainConfig":
        cfg = cls()
        if data:
            merge_into_dataclass(cfg, data)
        return cfg


def load_settings(path: str | Path | None) -> tuple[EnvConfig, TrainConfig]:
    """{"env": {...}, "train": {...}} 形式の JSON を読む。"""
    if path is None:
        return EnvConfig(), TrainConfig()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return EnvConfig.from_dict(data.get("env")), TrainConfig.from_dict(data.get("train"))


def _print_flush(msg: str) -> None:
    print(msg, flush=True)


class SelfPlayTrainer:
    def __init__(
        self, env_cfg: EnvConfig, cfg: TrainConfig, run_dir: str | Path, resume: bool = False, log=_print_flush
    ):
        self.env_cfg = env_cfg
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log = log
        for sub in ("", "snapshots", "replays"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.model_cfg = ModelConfig(num_fighters=env_cfg.scenario.num_fighters, hidden=cfg.hidden)
        self.model = PolicyNet(self.model_cfg)
        self.model.eval()
        self.ppo = PPO(self.model, cfg.ppo)
        self.league = League(cfg.league)
        self.iteration = 0
        self.best_eval = -np.inf
        self._pool = None
        self._local: RolloutWorker | None = None

        ckpt_path = self.run_dir / "checkpoint.pt"
        if resume and ckpt_path.exists():
            self._load(ckpt_path)
            self.log(f"resumed from {ckpt_path} (iteration {self.iteration})")
        elif cfg.init_checkpoint:
            ckpt = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["state_dict"])
            self.log(f"initialized weights from {cfg.init_checkpoint}")
        if cfg.init_checkpoint and (cfg.ppo.anchor_coef > 0 or cfg.ppo.anchor_coef_final > 0):
            anchor, _ = load_checkpoint(cfg.init_checkpoint)
            self.ppo.set_anchor(anchor)
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"env": env_cfg.to_dict(), "train": asdict(cfg)}, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------ 収集
    def _ensure_workers(self) -> None:
        c = self.cfg
        if c.num_workers > 0 and self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                c.num_workers,
                initializer=_init_worker,
                initargs=(self.env_cfg.to_dict(), self.model_cfg.to_dict(), c.gamma, c.gae_lambda, c.envs_per_worker),
            )
        if c.num_workers == 0 and self._local is None:
            self._local = RolloutWorker(
                self.env_cfg.to_dict(), self.model_cfg.to_dict(), c.gamma, c.gae_lambda, c.envs_per_worker
            )

    def close(self, terminate: bool = False) -> None:
        if self._pool is not None:
            if terminate:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None

    def _run(self, specs: list[EpisodeSpec]):
        self._ensure_workers()
        weights = self.run_dir / "current.pt"
        torch.save({"state_dict": self.model.state_dict()}, weights)
        version = (self.iteration, time.time())
        if self._pool is None:
            self._local.load_weights(str(weights), version)
            return self._local.run(specs)
        size = max(1, self.cfg.envs_per_worker)
        chunks = [specs[i : i + size] for i in range(0, len(specs), size)]
        trajs, results = [], []
        for t, r in self._pool.imap_unordered(_run_task, [(str(weights), version, ch) for ch in chunks]):
            trajs.extend(t)
            results.extend(r)
        return trajs, results

    def _make_specs(self) -> tuple[list[EpisodeSpec], dict[str, str]]:
        specs, names = [], {}
        for _ in range(self.cfg.episodes_per_iter):
            name = self.league.sample(self.rng)
            spec = self.league.spec_of(name)
            names[spec] = name
            specs.append(
                EpisodeSpec(
                    seed=int(self.rng.integers(2**31 - 1)),
                    opponent=spec,
                    learner_team=int(self.rng.integers(2)),
                )
            )
        return specs, names

    # ------------------------------------------------------------ 学習
    def train(self) -> None:
        c = self.cfg
        ok = False
        try:
            while self.iteration < c.iterations:
                t0 = time.time()
                specs, names = self._make_specs()
                trajs, results = self._run(specs)
                t_collect = time.time() - t0
                self.league.record([(names[r.opponent], r.learner_score) for r in results])
                stats = self.ppo.update(trajs, self.rng, self.iteration)
                self.iteration += 1
                it = self.iteration
                entry = {
                    "iteration": it,
                    "episodes": len(results),
                    "team_steps": int(sum(len(t) for t in trajs)),
                    "collect_sec": round(t_collect, 2),
                    "total_sec": None,
                    "opponents": self._summarize(results, names),
                    "ppo": stats,
                }
                if c.snapshot_interval and it % c.snapshot_interval == 0:
                    path = self.run_dir / "snapshots" / f"iter_{it:06d}.pt"
                    self._save_model(path)
                    self.league.add(path.stem, str(path), kind="snapshot", iteration=it)
                if c.eval_interval and it % c.eval_interval == 0:
                    entry["eval"] = self.evaluate()
                    score = float(np.mean([v["mean_score"] for v in entry["eval"].values()]))
                    if score > self.best_eval:
                        self.best_eval = score
                        self._save_model(self.run_dir / "best.pt")
                if c.replay_interval and it % c.replay_interval == 0:
                    self.save_replay(self.run_dir / "replays" / f"iter_{it:06d}.html")
                entry["league"] = self.league.table()
                entry["total_sec"] = round(time.time() - t0, 2)
                self._save_model(self.run_dir / "latest.pt")
                self._save(self.run_dir / "checkpoint.pt")
                with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.log(self._format(entry))
            ok = True
        finally:
            # 中断（Ctrl+C など）のときは実行中の対戦を待たずにワーカーを止める
            self.close(terminate=not ok)

    def evaluate(self, opponents: list[str] | None = None, episodes: int | None = None) -> dict:
        """固定の相手と対戦して平均スコア・勝率を返す（乱数シードは毎回同じ）。"""
        opponents = opponents or self.cfg.eval_opponents
        episodes = episodes or self.cfg.eval_episodes
        specs = [
            EpisodeSpec(
                seed=10_000 + i // 2,
                opponent=o,
                learner_team=i % 2,
                collect=False,
                deterministic=self.cfg.eval_deterministic,
            )
            for o in opponents
            for i in range(episodes)
        ]
        _, results = self._run(specs)
        out = {}
        for o in opponents:
            rs = [r for r in results if r.opponent == o]
            out[o] = _stats(rs)
        return out

    def save_replay(self, path: Path, opponent: str | None = None, seed: int = 10_000) -> Path:
        env = AirCombatEnv(self.env_cfg)
        me = PolicyAgent(self.model, deterministic=self.cfg.eval_deterministic, name=f"learner@{self.iteration}")
        opp_name = opponent or self.cfg.eval_opponents[0]
        opp = make_agent(opp_name)
        result = run_match(env, me, opp, seed=seed, record=True, blue_name=me.name, red_name=opp_name)
        return save_replay(result.replay, path)

    # ------------------------------------------------------------ 保存
    def _save_model(self, path: Path) -> None:
        save_checkpoint(path, self.model, self.env_cfg.to_dict(), {"iteration": self.iteration})

    def _save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model_cfg": self.model_cfg.to_dict(),
                "state_dict": self.model.state_dict(),
                "optimizer": self.ppo.opt.state_dict(),
                "league": self.league.state_dict(),
                "iteration": self.iteration,
                "best_eval": self.best_eval,
                "rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "env_cfg": self.env_cfg.to_dict(),
            },
            tmp,
        )
        tmp.replace(path)

    def _load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.ppo.opt.load_state_dict(ckpt["optimizer"])
        self.league = League.from_state_dict(ckpt["league"])
        self.iteration = ckpt["iteration"]
        self.best_eval = ckpt.get("best_eval", -np.inf)
        self.rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"])

    # ------------------------------------------------------------ 表示
    def _summarize(self, results: list[EpisodeResult], names: dict[str, str]) -> dict:
        groups = defaultdict(list)
        for r in results:
            groups[names.get(r.opponent, r.opponent)].append(r)
        return {name: _stats(rs) for name, rs in sorted(groups.items())}

    def _format(self, e: dict) -> str:
        opp = " ".join(
            f"{name}:{v['mean_score']:.2f}({v['games']})" for name, v in e["opponents"].items() if name != "self"
        )
        n_self = e["opponents"].get("self", {}).get("games", 0)
        p = e["ppo"]
        learner = self.league.ratings.get(LEARNER)
        line = (
            f"it {e['iteration']:4d} | {e['episodes']:3d} ep {e['team_steps'] / 1000:5.1f}k steps "
            f"{e['total_sec']:5.1f}s | self {n_self} | {opp} | "
            f"ent {p['entropy']:.2f} kl {p['approx_kl']:.3f} ev {p['explained_var']:.2f}"
            + (f" akl {p['anchor_kl']:.3f}" if p.get("anchor_coef") else "")
            + " | "
            f"rating {learner.rating:.0f}±{learner.rd:.0f}"
        )
        if "eval" in e:
            line += " | eval " + " ".join(
                f"{k}:{v['mean_score']:.3f} (W{v['wins']}/D{v['draws']}/L{v['losses']})" for k, v in e["eval"].items()
            )
        return line


def _stats(rs: list[EpisodeResult]) -> dict:
    if not rs:
        return {"games": 0, "mean_score": float("nan"), "wins": 0, "draws": 0, "losses": 0}
    scores = [r.learner_score for r in rs]
    wins = sum(1 for r in rs if r.winner == r.learner_team)
    draws = sum(1 for r in rs if r.winner == -1)
    hit = [r.outcome["hit_rates"][r.learner_team] for r in rs]
    return {
        "games": len(rs),
        "mean_score": round(float(np.mean(scores)), 4),
        "wins": wins,
        "draws": draws,
        "losses": len(rs) - wins - draws,
        "hit_rate": round(float(np.mean(hit)), 3),
        "mean_steps": round(float(np.mean([r.steps for r in rs])), 1),
    }
