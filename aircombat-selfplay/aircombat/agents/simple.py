"""単純なベースライン（動作確認・学習初期の対戦相手用）。"""

from __future__ import annotations

import numpy as np

from ..obs import NEUTRAL_ACTION, TeamObs, action_dims
from .base import Agent


class StraightAgent(Agent):
    """何もしない（直進・水平・速度維持・射撃なし）。"""

    name = "straight"

    def act(self, obs: TeamObs) -> np.ndarray:
        return np.tile(np.array(NEUTRAL_ACTION), (obs.self_feats.shape[0], 1))


class RandomAgent(Agent):
    """有効な行動から一様に選ぶ。射撃は fire_prob の確率でだけ試みる。"""

    name = "random"

    def __init__(self, fire_prob: float = 0.05, hold_steps: int = 5):
        self.fire_prob = fire_prob
        self.hold_steps = hold_steps  # 機動を何ステップ維持するか（毎ステップ変えると蛇行するだけになる）

    def reset(self, team, cfg, seed=None):
        super().reset(team, cfg, seed)
        self._held = None
        self._count = 0

    def act(self, obs: TeamObs) -> np.ndarray:
        nf = obs.self_feats.shape[0]
        dims = action_dims(nf)
        offs = np.cumsum((0,) + dims)
        if self._held is None or self._count % self.hold_steps == 0:
            held = np.zeros((nf, 3), dtype=int)
            for e in range(nf):
                for h in range(3):
                    valid = np.flatnonzero(obs.action_mask[e, offs[h] : offs[h + 1]])
                    held[e, h] = self.rng.choice(valid)
            self._held = held
        self._count += 1
        out = np.zeros((nf, 4), dtype=int)
        out[:, :3] = self._held
        for e in range(nf):
            fire_valid = np.flatnonzero(obs.action_mask[e, offs[3] + 1 : offs[4]])
            if fire_valid.size and self.rng.random() < self.fire_prob:
                out[e, 3] = int(self.rng.choice(fire_valid)) + 1
            # 撃墜された機体の行動はマスクに合わせて中立にする
            if not obs.alive[e]:
                out[e] = NEUTRAL_ACTION
        return out
