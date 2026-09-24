import numpy as np
import pytest

from aircombat.agents import make_agent
from aircombat.agents.simple import RandomAgent
from aircombat.config import EnvConfig, ScoreConfig
from aircombat.env import AirCombatEnv
from aircombat.match import run_match
from aircombat.obs import ALLY_DIM, ENEMY_DIM, MWS_DIM, MWS_SLOTS, SELF_DIM, action_dims
from aircombat.rules import compute_scores, judge, speed_factor
from conftest import isolated_state, make_cfg


# ---------------------------------------------------------------- スコア
def test_score_formula():
    sc = ScoreConfig()
    # 全弾命中・900 秒以内の勝利は満点
    assert compute_scores(sc, 0, 500.0, 1200.0, [4, 0], [4, 0]) == pytest.approx((1.0, 0.0))
    # 命中率 50%・1050 秒: 0.6 + 0.3*0.5 + 0.1*(150/300)
    assert compute_scores(sc, 1, 1050.0, 1200.0, [3, 6], [0, 3]) == pytest.approx((0.2, 0.8))
    assert compute_scores(sc, -1, 1200.0, 1200.0, [3, 6], [0, 3]) == (0.5, 0.5)
    assert speed_factor(sc, 1200.0, 1200.0) == 0.0


def test_escort_kill_ends_game_with_winner():
    cfg = make_cfg()
    env = AirCombatEnv(cfg)
    env.reset(initial_state=isolated_state(cfg))
    sim = env.sim
    sim.launched[:] = [2, 0]
    sim.hits[:] = [1, 0]
    sim._destroy(int(sim.escort_idx[1]), "missile", 0)
    out = judge(sim, cfg)
    assert out.winner == 0 and out.reason == "escort_destroyed"
    assert out.scores[0] == pytest.approx(0.6 + 0.3 * 0.5 + 0.1)
    assert sum(out.scores) == pytest.approx(1.0)


def test_time_limit_is_a_draw():
    cfg = make_cfg(time_limit=5.0)
    env = AirCombatEnv(cfg)
    obs = env.reset(initial_state=isolated_state(cfg))
    straight = make_agent("straight")
    done = False
    while not done:
        obs, rewards, done, info = env.step({0: straight.act(obs[0]), 1: straight.act(obs[1])})
    assert info["outcome"].reason == "time_limit"
    assert info["outcome"].scores == (0.5, 0.5)
    assert rewards == pytest.approx([0.0, 0.0])


def test_no_means_ends_game_early():
    cfg = make_cfg()
    env = AirCombatEnv(cfg)
    obs = env.reset(initial_state=isolated_state(cfg))
    env.sim.missiles_left[:] = 0
    straight = make_agent("straight")
    obs, _, done, info = env.step({0: straight.act(obs[0]), 1: straight.act(obs[1])})
    assert done and info["outcome"].reason == "no_means"


# ---------------------------------------------------------------- 観測・行動
def test_observation_shapes_and_masks():
    env = AirCombatEnv()
    obs = env.reset(seed=1)
    nf = env.nf
    for k in (0, 1):
        o = obs[k]
        assert o.self_feats.shape == (nf, SELF_DIM)
        assert o.ally_feats.shape == (nf, nf, ALLY_DIM)
        assert o.enemy_feats.shape == (nf, nf + 1, ENEMY_DIM)
        assert o.mws_feats.shape == (nf, MWS_SLOTS, MWS_DIM)
        assert o.action_mask.shape == (nf, sum(action_dims(nf)))
        assert np.isfinite(o.self_feats).all() and np.isfinite(o.enemy_feats).all()
        # 射撃しない、は常に選べる
        off = sum(action_dims(nf)[:3])
        assert o.action_mask[:, off].all()


def test_symmetric_scenario_gives_identical_observations():
    cfg = EnvConfig.from_dict({"scenario": {"symmetric": True}})
    env = AirCombatEnv(cfg)
    obs = env.reset(seed=3)
    agent = [make_agent("rule"), make_agent("rule")]
    for k in (0, 1):
        agent[k].reset(k, cfg, seed=0)
    for _ in range(60):
        for name in ("self_feats", "ally_feats", "enemy_feats", "mws_feats", "action_mask"):
            np.testing.assert_allclose(getattr(obs[0], name), getattr(obs[1], name), atol=1e-4, err_msg=name)
        acts = {k: agent[k].act(obs[k]) for k in (0, 1)}
        np.testing.assert_array_equal(acts[0], acts[1])
        obs, rewards, done, _ = env.step(acts)
        assert rewards[0] == pytest.approx(rewards[1])
        if done:
            break


def test_masked_random_actions_never_fire_invalidly():
    # 近い位置から始めて、ランダムでも射撃機会が生じるようにする
    env = AirCombatEnv(EnvConfig.from_dict({"scenario": {"time_limit": 300, "fighter_x_range": [-25_000, -20_000]}}))
    for seed in range(2):
        obs = env.reset(seed=seed)
        agents = [RandomAgent(fire_prob=0.5), RandomAgent(fire_prob=0.5)]
        for k in (0, 1):
            agents[k].reset(k, env.cfg, seed=seed)
        done = False
        while not done:
            obs, _, done, _ = env.step({k: agents[k].act(obs[k]) for k in (0, 1)})
        assert env.invalid_fire.sum() == 0
        assert env.sim.launched.sum() > 0


def test_action_out_of_range_raises():
    env = AirCombatEnv()
    env.reset(seed=0)
    bad = np.zeros((env.nf, 4), dtype=int)
    bad[0, 0] = 99
    with pytest.raises(ValueError):
        env.step({0: bad, 1: np.zeros((env.nf, 4), dtype=int)})


def test_same_seed_is_deterministic():
    env = AirCombatEnv()
    a = run_match(env, make_agent("rule"), make_agent("rule_aggressive"), seed=5)
    b = run_match(env, make_agent("rule"), make_agent("rule_aggressive"), seed=5)
    assert a.outcome == b.outcome and a.steps == b.steps


def test_enemy_track_memory_and_bearing_only():
    cfg = make_cfg(track_memory=10.0)
    env = AirCombatEnv(cfg)
    init = isolated_state(cfg)
    nf = cfg.scenario.num_fighters
    init.pos[0] = [0.0, 0.0, 10_000.0]
    init.heading[0] = 0.0
    init.pos[nf + 1] = [60_000.0, 0.0, 10_000.0]  # レーダーで見える
    init.heading[nf + 1] = 0.0  # 遠ざかる
    init.pos[nf + 2] = [5_000.0, 20_000.0, 10_000.0]  # IRST のみ
    obs = env.reset(initial_state=init)
    v = obs[0].view
    assert v.enemy_tracked[0] and not v.enemy_tracked[1]
    ef = obs[0].enemy_feats[0]
    slots = obs[0].enemy_slots[0]
    s_radar = int(np.flatnonzero(slots == 0)[0])
    s_irst = int(np.flatnonzero(slots == 1)[0])
    assert ef[s_radar, 2] == 1 and ef[s_radar, 9] > 0  # 航跡あり・距離あり
    assert ef[s_irst, 4] == 1 and ef[s_irst, 9] == 0  # 方位のみ（距離なし）
    # 自機が反対を向くとレーダーから外れ、推測位置で track_memory 秒だけ残る
    env.sim.heading[0] = np.pi
    env.sim.vel[0] *= -1
    straight = make_agent("straight")
    for t in range(12):
        obs, _, _, _ = env.step({0: straight.act(obs[0]), 1: straight.act(obs[1])})
        v = obs[0].view
        if t < 9:
            assert v.enemy_known[0] and not v.enemy_tracked[0]
    assert not v.enemy_known[0]


def test_rewards_are_zero_sum_for_missile_kills():
    env = AirCombatEnv(EnvConfig.from_dict({"scenario": {"time_limit": 600}}))
    total = np.zeros(2)
    obs = env.reset(seed=4)
    agents = [make_agent("rule"), make_agent("rule_aggressive")]
    for k in (0, 1):
        agents[k].reset(k, env.cfg, seed=0)
    done = False
    while not done:
        obs, r, done, info = env.step({k: agents[k].act(obs[k]) for k in (0, 1)})
        crashes = [e for e in info["events"] if e.kind == "destroyed" and e.detail != "missile"]
        if not crashes:
            assert r.sum() == pytest.approx(0.0, abs=1e-9)
        total += r
    assert np.isfinite(total).all()


def test_2d_mode_masks_pitch_and_keeps_altitude():
    env = AirCombatEnv(EnvConfig.from_dict({"scenario": {"mode": "2d", "fixed_altitude": 9_000}}))
    obs = env.reset(seed=0)
    d_turn, d_pitch = action_dims(env.nf)[:2]
    pitch_mask = obs[0].action_mask[:, d_turn : d_turn + d_pitch]
    assert pitch_mask.sum(axis=1).tolist() == [1] * env.nf and pitch_mask[:, 2].all()
    agents = [make_agent("rule"), make_agent("rule")]
    for k in (0, 1):
        agents[k].reset(k, env.cfg, seed=0)
    for _ in range(200):
        obs, _, done, _ = env.step({k: agents[k].act(obs[k]) for k in (0, 1)})
        if done:
            break
    np.testing.assert_allclose(env.sim.pos[:, 2], 9_000.0)
    assert np.all(env.sim.gamma == 0.0)
