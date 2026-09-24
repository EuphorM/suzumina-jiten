"""ロールアウト（学習データの収集）。

RolloutWorker は複数の環境を並行して進め、学習中の方策の推論を環境をまたいでまとめて行う。
対戦相手は episode ごとに指定する:
- "self": 現在の方策同士の対戦（両陣営のデータを学習に使う）
- 組み込みのエージェント名（"rule" など）
- スナップショット（*.pt）のパス
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents import BUILTIN, make_agent
from ..agents.base import Agent
from ..config import EnvConfig
from ..env import AirCombatEnv
from .model import OBS_KEYS, ModelConfig, PolicyNet, load_checkpoint, obs_to_arrays, stack_obs, to_tensors
from .policy_agent import PolicyAgent

SELF = "self"


@dataclass
class EpisodeSpec:
    seed: int
    opponent: str
    learner_team: int = 0  # opponent が "self" のときは無視（両陣営とも学習者）
    collect: bool = True  # False なら評価用（軌跡を返さない）
    deterministic: bool = False  # True なら学習者は最も確率の高い行動を取る（評価用）


@dataclass
class Trajectory:
    obs: dict[str, np.ndarray]  # 各 (T, ...)
    actions: np.ndarray  # (T, nf, 4)
    logp: np.ndarray  # (T, nf)
    alive: np.ndarray  # (T, nf)
    values: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)
    advantages: np.ndarray = field(default=None)  # (T,)
    returns: np.ndarray = field(default=None)  # (T,)

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class EpisodeResult:
    opponent: str
    learner_team: int
    winner: int
    scores: tuple[float, float]
    outcome: dict
    steps: int

    @property
    def learner_score(self) -> float:
        return self.scores[self.learner_team]


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float, last_value: float = 0.0):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


class _Buffer:
    def __init__(self):
        self.obs: list[dict[str, np.ndarray]] = []
        self.actions, self.logp, self.alive, self.values, self.rewards = [], [], [], [], []

    def to_trajectory(self, gamma: float, lam: float) -> Trajectory:
        values = np.asarray(self.values, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        adv, ret = compute_gae(rewards, values, gamma, lam)
        return Trajectory(
            obs={k: np.stack([o[k] for o in self.obs]) for k in OBS_KEYS},
            actions=np.asarray(self.actions, dtype=np.int64),
            logp=np.asarray(self.logp, dtype=np.float32),
            alive=np.asarray(self.alive, dtype=bool),
            values=values,
            rewards=rewards,
            advantages=adv,
            returns=ret.astype(np.float32),
        )


@dataclass
class _Slot:
    spec: EpisodeSpec
    obs: dict
    learner_teams: tuple[int, ...]
    opponent: Agent | None
    buffers: dict[int, _Buffer]
    steps: int = 0


class RolloutWorker:
    def __init__(self, env_cfg: dict, model_cfg: dict, gamma: float, lam: float, envs: int = 4, cache_size: int = 16):
        self.env_cfg = EnvConfig.from_dict(env_cfg)
        self.envs = [AirCombatEnv(self.env_cfg) for _ in range(max(1, envs))]
        self.model = PolicyNet(ModelConfig(**model_cfg))
        self.model.eval()
        self.version = None
        self.gamma = gamma
        self.lam = lam
        self._models: OrderedDict[str, PolicyNet] = OrderedDict()
        self._cache_size = cache_size

    def set_weights(self, state_dict, version) -> None:
        self.model.load_state_dict(state_dict)
        self.version = version

    def load_weights(self, path: str, version) -> None:
        if version != self.version:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.set_weights(ckpt["state_dict"], version)

    def _opponent(self, spec: str) -> Agent | None:
        if spec == SELF:
            return None
        if spec in BUILTIN or ":" in spec:
            return make_agent(spec)
        model = self._models.get(spec)
        if model is None:
            model, _ = load_checkpoint(spec)
            self._models[spec] = model
            while len(self._models) > self._cache_size:
                self._models.popitem(last=False)
        else:
            self._models.move_to_end(spec)
        return PolicyAgent(model, deterministic=False, name=spec)

    def _start(self, env: AirCombatEnv, spec: EpisodeSpec) -> _Slot:
        obs = env.reset(seed=spec.seed)
        opp = self._opponent(spec.opponent)
        if opp is None:
            learner_teams: tuple[int, ...] = (0, 1)
        else:
            learner_teams = (spec.learner_team,)
            opp.reset(1 - spec.learner_team, env.cfg, seed=spec.seed)
        return _Slot(spec, obs, learner_teams, opp, {k: _Buffer() for k in learner_teams})

    @torch.no_grad()
    def run(self, specs: list[EpisodeSpec]) -> tuple[list[Trajectory], list[EpisodeResult]]:
        queue = list(specs)
        slots: list[_Slot | None] = [None] * len(self.envs)
        for i, env in enumerate(self.envs):
            if queue:
                slots[i] = self._start(env, queue.pop(0))
        trajs: list[Trajectory] = []
        results: list[EpisodeResult] = []
        while any(s is not None for s in slots):
            index, items = [], []
            for i, st in enumerate(slots):
                if st is None:
                    continue
                for team in st.learner_teams:
                    index.append((i, team))
                    items.append(obs_to_arrays(st.obs[team]))
            logits, values = self.model(to_tensors(stack_obs(items)))
            logp, _, acts = PolicyNet.distribution_stats(logits)
            det = [slots[i].spec.deterministic for i, _ in index]
            if any(det):
                greedy = torch.stack([lg.argmax(-1) for lg in logits], dim=-1)
                acts = torch.where(torch.tensor(det)[:, None, None], greedy, acts)
            acts, logp, values = acts.numpy(), logp.numpy(), values.numpy()
            per_env: dict[int, dict[int, np.ndarray]] = {}
            for j, (i, team) in enumerate(index):
                per_env.setdefault(i, {})[team] = acts[j]
                st = slots[i]
                if st.spec.collect:
                    b = st.buffers[team]
                    b.obs.append(items[j])
                    b.actions.append(acts[j])
                    b.logp.append(logp[j])
                    b.alive.append(st.obs[team].alive.copy())
                    b.values.append(values[j])
            for i, st in enumerate(slots):
                if st is None:
                    continue
                actions = per_env[i]
                if st.opponent is not None:
                    opp_team = 1 - st.spec.learner_team
                    actions[opp_team] = st.opponent.act(st.obs[opp_team])
                obs, rewards, done, info = self.envs[i].step(actions)
                st.obs = obs
                st.steps += 1
                if st.spec.collect:
                    for team in st.learner_teams:
                        st.buffers[team].rewards.append(float(rewards[team]))
                if done:
                    outcome = info["outcome"]
                    # 自己対戦でも結果は 1 件だけ記録する（軌跡は両陣営分を学習に使う）
                    results.append(
                        EpisodeResult(
                            opponent=st.spec.opponent,
                            learner_team=st.learner_teams[0],
                            winner=outcome.winner,
                            scores=outcome.scores,
                            outcome=outcome.to_dict(),
                            steps=st.steps,
                        )
                    )
                    if st.spec.collect:
                        for team in st.learner_teams:
                            trajs.append(st.buffers[team].to_trajectory(self.gamma, self.lam))
                    slots[i] = self._start(self.envs[i], queue.pop(0)) if queue else None
        return trajs, results


# ---------------------------------------------------------------- 並列実行
_WORKER: RolloutWorker | None = None


def _init_worker(env_cfg: dict, model_cfg: dict, gamma: float, lam: float, envs: int) -> None:
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = RolloutWorker(env_cfg, model_cfg, gamma, lam, envs)


def _run_task(args):
    weights_path, version, specs = args
    _WORKER.load_weights(weights_path, version)
    return _WORKER.run(specs)
