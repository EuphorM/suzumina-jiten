import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aircombat.agents import make_agent  # noqa: E402
from aircombat.config import EnvConfig  # noqa: E402
from aircombat.env import AirCombatEnv  # noqa: E402
from aircombat.match import run_match  # noqa: E402
from aircombat.obs import action_dims  # noqa: E402
from aircombat.selfplay.model import ModelConfig, PolicyNet, obs_to_arrays, stack_obs, to_tensors  # noqa: E402
from aircombat.selfplay.rollout import EpisodeSpec, RolloutWorker, compute_gae  # noqa: E402
from aircombat.selfplay.trainer import SelfPlayTrainer, TrainConfig  # noqa: E402


def test_gae_matches_manual_computation():
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    gamma, lam = 0.9, 0.8
    adv, ret = compute_gae(rewards, values, gamma, lam)
    d2 = 1.0 - 0.7
    d1 = 0.0 + gamma * 0.7 - 0.6
    d0 = 0.0 + gamma * 0.6 - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-5)
    np.testing.assert_allclose(ret, adv + values, rtol=1e-6)


def test_policy_samples_only_valid_actions():
    torch.manual_seed(0)
    env = AirCombatEnv()
    obs = env.reset(seed=0)
    model = PolicyNet(ModelConfig(num_fighters=env.nf, hidden=32))
    # 敵が射程内に入るまで進めて、射撃マスクに有効なスロットがある状態も確かめる
    straight = make_agent("straight")
    for _ in range(120):
        t = to_tensors(stack_obs([obs_to_arrays(obs[0]), obs_to_arrays(obs[1])]))
        acts, logp, value = model.act(t)
        assert acts.shape == (2, env.nf, 4) and logp.shape == (2, env.nf) and value.shape == (2,)
        dims = action_dims(env.nf)
        for b in range(2):
            mask = obs[b].action_mask
            off = 0
            for h, d in enumerate(dims):
                for e in range(env.nf):
                    assert mask[e, off + int(acts[b, e, h])]
                off += d
        obs, _, done, _ = env.step({0: acts[0].numpy(), 1: straight.act(obs[1])})
        if done:
            break
    assert env.invalid_fire.sum() == 0


def test_rollout_worker_collects_consistent_trajectories():
    torch.manual_seed(0)
    env_cfg = EnvConfig.from_dict({"scenario": {"time_limit": 60}})
    worker = RolloutWorker(env_cfg.to_dict(), ModelConfig(hidden=32).to_dict(), 0.99, 0.95, envs=2)
    specs = [
        EpisodeSpec(seed=1, opponent="self"),
        EpisodeSpec(seed=2, opponent="rule", learner_team=1),
        EpisodeSpec(seed=3, opponent="straight", learner_team=0, collect=False),
    ]
    trajs, results = worker.run(specs)
    assert len(results) == 3
    assert len(trajs) == 3  # 自己対戦は両陣営分、評価用 (collect=False) は無し
    for t in trajs:
        T = len(t)
        assert T == 60
        assert t.obs["self"].shape[0] == T and t.actions.shape == (T, 4, 4)
        assert t.alive.shape == (T, 4) and t.advantages.shape == (T,)


def test_trainer_runs_saves_and_resumes(tmp_path):
    env_cfg = EnvConfig.from_dict({"scenario": {"time_limit": 40}})
    cfg = TrainConfig.from_dict(
        {
            "iterations": 2,
            "episodes_per_iter": 3,
            "num_workers": 0,
            "envs_per_worker": 3,
            "hidden": 32,
            "snapshot_interval": 1,
            "eval_interval": 2,
            "eval_episodes": 2,
            "replay_interval": 2,
            "ppo": {"minibatch_size": 64, "epochs": 1},
            "league": {"initial_opponents": ["rule", "straight"]},
        }
    )
    logs = []
    trainer = SelfPlayTrainer(env_cfg, cfg, tmp_path, log=logs.append)
    trainer.train()
    for name in ("config.json", "checkpoint.pt", "latest.pt", "best.pt", "metrics.jsonl"):
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "replays" / "iter_000002.html").exists()
    assert len(list((tmp_path / "snapshots").glob("*.pt"))) == 2
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and "eval" in json.loads(lines[-1])
    assert "iter_000001" in trainer.league.members

    # 学習済みモデルはエージェントとしてそのまま対戦に使える
    env = AirCombatEnv(env_cfg)
    r = run_match(env, make_agent(str(tmp_path / "latest.pt")), make_agent("rule"), seed=0)
    assert r.steps > 0

    # 再開すると続きのイテレーションから始まる
    cfg.iterations = 3
    resumed = SelfPlayTrainer(env_cfg, cfg, tmp_path, resume=True, log=logs.append)
    assert resumed.iteration == 2
    assert set(resumed.league.members) == set(trainer.league.members)
    resumed.train()
    assert resumed.iteration == 3


def test_imitation_pretraining_produces_usable_model(tmp_path):
    from aircombat.selfplay.imitation import ImitationConfig, collect_demonstrations, pretrain

    env_cfg = EnvConfig.from_dict({"scenario": {"time_limit": 60, "fighter_x_range": [-25_000, -20_000]}})
    cfg = ImitationConfig(opponents=["straight"], episodes=2, epochs=1, workers=0, hidden=32, stride=3)
    data = collect_demonstrations(env_cfg, cfg, log=lambda *_: None)
    n = len(data["act"])
    # 60 ステップを 3 ステップおきに記録（2 戦で 40）し、教師が撃った場面はそれに加えて必ず記録する
    fired = np.any(data["act"][..., 3] > 0, axis=1)
    assert fired.any()
    assert 2 * 20 <= n <= 2 * 60
    assert data["self"].dtype == np.float16 and data["act"].shape == (n, 4, 4)
    out = pretrain(env_cfg, cfg, tmp_path / "bc.pt", log=lambda *_: None)
    env = AirCombatEnv(env_cfg)
    r = run_match(env, make_agent(str(out)), make_agent("straight"), seed=0)
    assert r.steps > 0
    # 模倣学習の重みから強化学習を始められる
    tcfg = TrainConfig.from_dict(
        {"iterations": 1, "episodes_per_iter": 1, "num_workers": 0, "hidden": 32, "init_checkpoint": str(out),
         "snapshot_interval": 0, "eval_interval": 0, "replay_interval": 0, "ppo": {"minibatch_size": 32, "epochs": 1}}
    )
    trainer = SelfPlayTrainer(env_cfg, tcfg, tmp_path / "run", log=lambda *_: None)
    loaded = torch.load(out, weights_only=False)["state_dict"]
    assert all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in loaded.items())
    trainer.train()


def test_anchor_kl_regularization():
    from aircombat.selfplay.model import anchor_kl
    from aircombat.selfplay.ppo import PPO, PPOConfig

    torch.manual_seed(0)
    env = AirCombatEnv()
    obs = env.reset(seed=0)
    t = to_tensors(stack_obs([obs_to_arrays(obs[0]), obs_to_arrays(obs[1])]))
    model = PolicyNet(ModelConfig(hidden=32))
    anchor = PolicyNet(ModelConfig(hidden=32))
    anchor.load_state_dict(model.state_dict())
    lg, _ = model(t)
    la, _ = anchor(t)
    assert torch.allclose(anchor_kl(la, lg), torch.zeros(2, 4), atol=1e-6)
    with torch.no_grad():
        model.turn.bias[0] += 1.0  # 1 つの選択肢だけ変えると分布が変わる
    lg2, _ = model(t)
    assert (anchor_kl(la, lg2) > 0).all()

    ppo = PPO(model, PPOConfig(anchor_coef=1.0, anchor_coef_final=0.1, anchor_decay_iters=10))
    assert ppo.anchor_coef(5) == 0.0  # アンカー未設定なら無効
    ppo.set_anchor(anchor)
    assert ppo.anchor_coef(0) == pytest.approx(1.0)
    assert ppo.anchor_coef(5) == pytest.approx(0.55)
    assert ppo.anchor_coef(20) == pytest.approx(0.1)

    worker = RolloutWorker(EnvConfig.from_dict({"scenario": {"time_limit": 20}}).to_dict(), ModelConfig(hidden=32).to_dict(), 0.99, 0.95, envs=1)
    worker.set_weights(model.state_dict(), 0)
    trajs, _ = worker.run([EpisodeSpec(seed=0, opponent="self")])
    stats = ppo.update(trajs, np.random.default_rng(0), iteration=0)
    assert stats["anchor_coef"] == pytest.approx(1.0) and stats["anchor_kl"] > 0


def test_deterministic_eval_specs_use_greedy_actions():
    torch.manual_seed(0)
    env_cfg = EnvConfig.from_dict({"scenario": {"time_limit": 30}})
    worker = RolloutWorker(env_cfg.to_dict(), ModelConfig(hidden=32).to_dict(), 0.99, 0.95, envs=2)
    specs = [EpisodeSpec(seed=7, opponent="straight", learner_team=0, collect=False, deterministic=True)] * 2
    _, results = worker.run(specs)
    assert results[0].outcome == results[1].outcome  # 決定的なので同じシードなら同じ結果


def test_init_checkpoint_network_config_takes_precedence(tmp_path):
    from aircombat.selfplay.model import save_checkpoint

    env_cfg = EnvConfig()
    src = PolicyNet(ModelConfig(hidden=16))
    save_checkpoint(tmp_path / "init.pt", src, env_cfg.to_dict())
    cfg = TrainConfig.from_dict({"hidden": 64, "num_workers": 0, "init_checkpoint": str(tmp_path / "init.pt")})
    trainer = SelfPlayTrainer(env_cfg, cfg, tmp_path / "run", log=lambda *_: None)
    assert trainer.model_cfg.hidden == 16
    for k, v in src.state_dict().items():
        assert torch.equal(v, trainer.model.state_dict()[k])
