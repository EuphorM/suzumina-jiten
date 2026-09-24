import numpy as np
import pytest

from aircombat.config import EnvConfig
from aircombat.wrappers import SingleTeamEnv

gym = pytest.importorskip("gymnasium")


def test_single_team_env_passes_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = SingleTeamEnv(EnvConfig.from_dict({"scenario": {"time_limit": 30}}), opponent="rule")
    check_env(env, skip_render_check=True)


def test_single_team_env_episode_and_masks():
    env = SingleTeamEnv(EnvConfig.from_dict({"scenario": {"time_limit": 60}}), opponent="straight", team=1)
    obs, info = env.reset(seed=0)
    assert info["team"] == 1
    assert env.observation_space.contains(obs)
    total = 0.0
    for _ in range(100):
        mask = env.action_masks()
        assert mask.shape == (int(env.action_space.nvec.sum()),)
        # マスクで有効な選択肢から選ぶ
        action = []
        off = 0
        for d in env.action_space.nvec:
            valid = np.flatnonzero(mask[off : off + d])
            action.append(int(valid[0]))
            off += d
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        total += reward
        if terminated:
            break
    assert terminated and not truncated
    assert info["outcome"]["reason"] == "time_limit" and info["score"] == 0.5
