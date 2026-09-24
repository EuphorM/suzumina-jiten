import numpy as np
import pytest

from aircombat.config import EnvConfig
from aircombat.sim.scenario import generate_initial_state


def make_cfg(**scenario) -> EnvConfig:
    """テスト用: 個体差なし・空域外での撃墜なし・安全装置なし。"""
    base = {"performance_jitter": 0.0, "out_of_bounds": "ignore", "altitude_guard": 0.0, "boundary_guard": 0.0}
    base.update(scenario)
    return EnvConfig.from_dict({"scenario": base})


def isolated_state(cfg: EnvConfig, alt: float = 10_000.0):
    """全機を互いに探知できない位置（空域の四隅付近、外向き）に置いた初期状態。"""
    init = generate_initial_state(cfg, np.random.default_rng(0))
    n = len(init.speed)
    half = n // 2
    for i in range(n):
        team = 0 if i < half else 1
        j = i % half
        x = -95_000.0 if team == 0 else 95_000.0
        y = (-70_000.0 + 4_000.0 * j) if team == 0 else (70_000.0 - 4_000.0 * j)
        init.pos[i] = [x, y, alt]
        init.heading[i] = np.pi if team == 0 else 0.0  # 外向き（敵に背を向ける）
        init.speed[i] = 250.0
    init.escort_center[:] = init.pos[[half - 1, n - 1], :2]
    return init


@pytest.fixture
def cfg():
    return make_cfg()
