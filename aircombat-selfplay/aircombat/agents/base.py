"""エージェントの共通インターフェース。"""

from __future__ import annotations

import numpy as np

from ..config import EnvConfig
from ..obs import TeamObs


class Agent:
    """1 陣営（戦闘機 num_fighters 機）をまとめて操縦するエージェント。

    観測は陣営座標なので、青・赤どちらに割り当てられても同じ実装で動く。
    act() は (num_fighters, 4) の整数配列 [旋回, 経路角, スロットル, 射撃] を返す。
    """

    name = "agent"

    def reset(self, team: int, cfg: EnvConfig, seed: int | None = None) -> None:
        self.team = team
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def act(self, obs: TeamObs) -> np.ndarray:
        raise NotImplementedError
