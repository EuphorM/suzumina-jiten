"""他の強化学習ライブラリ向けのラッパー。

SingleTeamEnv は 1 陣営だけを外から操作し、相手陣営は固定のエージェントが動かす
Gymnasium 互換の環境（reset/step の戻り値が Gymnasium と同じ形）。
gymnasium がインストールされていれば observation_space / action_space も持つ。

    env = SingleTeamEnv(opponent="rule")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

行動は (num_fighters * 4,) の MultiDiscrete（機体ごとに [旋回, 経路角, スロットル, 射撃]）。
action_masks() は sb3-contrib の MaskablePPO がそのまま使える形の真偽配列を返す。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import make_agent
from .agents.base import Agent
from .config import EnvConfig
from .env import AirCombatEnv
from .obs import ALLY_DIM, ENEMY_DIM, MWS_DIM, MWS_SLOTS, SELF_DIM, TeamObs

try:
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
except ImportError:  # gymnasium なしでも同じメソッドで使える
    gym = None
    spaces = None
    _Base = object


def team_obs_to_dict(obs: TeamObs) -> dict[str, np.ndarray]:
    return {
        "self": obs.self_feats,
        "ally": obs.ally_feats,
        "enemy": obs.enemy_feats,
        "mws": obs.mws_feats,
        "action_mask": obs.action_mask.astype(np.int8),
    }


class SingleTeamEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | dict | None = None,
        opponent: str | Agent = "rule",
        team: int | str = "random",
    ):
        self.env = AirCombatEnv(config)
        self.nf = self.env.nf
        self.dims = np.array(self.env.action_dims)
        self._opponent_spec = opponent
        self._team_choice = team
        self._rng = np.random.default_rng()
        self.team = 0
        self.opponent: Agent | None = None
        self._obs: dict[int, TeamObs] = {}
        if spaces is not None:
            nf, ne = self.nf, self.nf + 1
            # 特徴量はおおむね ±3 に正規化してある
            box = lambda *shape: spaces.Box(-10.0, 10.0, shape=shape, dtype=np.float32)  # noqa: E731
            self.observation_space = spaces.Dict(
                {
                    "self": box(nf, SELF_DIM),
                    "ally": box(nf, nf, ALLY_DIM),
                    "enemy": box(nf, ne, ENEMY_DIM),
                    "mws": box(nf, MWS_SLOTS, MWS_DIM),
                    "action_mask": spaces.MultiBinary([nf, int(self.dims.sum())]),
                }
            )
            self.action_space = spaces.MultiDiscrete(np.tile(self.dims, nf))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)  # self.np_random を seed で初期化する（Gymnasium の約束）
            self._rng = self.np_random
        elif seed is not None:
            self._rng = np.random.default_rng(seed)
        self.team = int(self._rng.integers(2)) if self._team_choice == "random" else int(self._team_choice)
        spec = self._opponent_spec
        self.opponent = make_agent(spec) if isinstance(spec, str) else spec
        env_seed = int(self._rng.integers(2**31 - 1))
        self._obs = self.env.reset(seed=env_seed)
        self.opponent.reset(1 - self.team, self.env.cfg, seed=env_seed)
        return team_obs_to_dict(self._obs[self.team]), {"team": self.team}

    def step(self, action: Any):
        a = np.asarray(action, dtype=int).reshape(self.nf, 4)
        opp = 1 - self.team
        actions = {self.team: a, opp: self.opponent.act(self._obs[opp])}
        self._obs, rewards, done, info = self.env.step(actions)
        out_info: dict[str, Any] = {"team": self.team, "time": info["time"]}
        if done:
            out_info["outcome"] = info["outcome"].to_dict()
            out_info["score"] = info["outcome"].scores[self.team]
        return team_obs_to_dict(self._obs[self.team]), float(rewards[self.team]), bool(done), False, out_info

    def action_masks(self) -> np.ndarray:
        return self._obs[self.team].action_mask.reshape(-1).copy()

    @property
    def team_obs(self) -> TeamObs:
        """現在の TeamObs（view など辞書に入らない情報も含む）。"""
        return self._obs[self.team]
