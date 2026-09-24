import json

import numpy as np
import pytest

from aircombat.agents import make_agent
from aircombat.config import EnvConfig
from aircombat.env import AirCombatEnv
from aircombat.evaluate import round_robin
from aircombat.match import run_match, save_replay, summarize
from aircombat.selfplay.glicko2 import Rating, RatingTable, expected_score, update
from aircombat.selfplay.league import LEARNER, League, LeagueConfig
from aircombat.viewer import render_html


def test_rule_based_beats_passive_and_random():
    env = AirCombatEnv()
    for opp in ("straight", "random"):
        r = run_match(env, make_agent("rule"), make_agent(opp), seed=1)
        assert r.winner == 0, (opp, r.outcome)
        assert r.scores[0] > 0.6


def test_make_agent_rejects_unknown_spec():
    with pytest.raises(ValueError):
        make_agent("no_such_agent")


def test_custom_agent_by_import_path():
    agent = make_agent("aircombat.agents.simple:StraightAgent")
    assert agent.name == "straight"


def test_replay_json_and_html(tmp_path):
    env = AirCombatEnv(EnvConfig.from_dict({"scenario": {"time_limit": 120}}))
    r = run_match(env, make_agent("rule"), make_agent("rule"), seed=0, record=True)
    replay = r.replay
    assert len(replay["frames"]) == r.steps + 1
    assert replay["outcome"]["reason"] == r.outcome["reason"]
    html = render_html(replay)
    assert "/*__REPLAY_DATA__*/" not in html and "空戦リプレイ" in html
    save_replay(replay, tmp_path / "r.html")
    save_replay(replay, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["seed"] == 0


def test_summarize_counts_by_agent_name():
    env = AirCombatEnv(EnvConfig.from_dict({"scenario": {"time_limit": 60}}))
    rs = [run_match(env, make_agent("straight"), make_agent("straight"), seed=i, blue_name="a", red_name="b") for i in range(2)]
    s = summarize(rs, "a")
    assert s["games"] == 2 and s["draws"] == 2 and s["mean_score"] == 0.5


# ---------------------------------------------------------------- Glicko-2
def test_glicko2_matches_glickman_example():
    r = update(
        Rating(1500, 200, 0.06),
        [(Rating(1400, 30), 1.0), (Rating(1550, 100), 0.0), (Rating(1700, 300), 0.0)],
        tau=0.5,
    )
    assert r.rating == pytest.approx(1464.06, abs=0.02)
    assert r.rd == pytest.approx(151.52, abs=0.01)
    assert r.vol == pytest.approx(0.05999, abs=1e-5)


def test_rating_table_orders_stronger_player_first():
    t = RatingTable()
    games = [("strong", "weak", 0.9)] * 10 + [("weak", "mid", 0.3)] * 10 + [("strong", "mid", 0.7)] * 10
    for _ in range(3):
        t.update_period(games)
    names = [n for n, _ in t.leaderboard()]
    assert names == ["strong", "mid", "weak"]
    assert expected_score(t.get("strong"), t.get("weak")) > 0.5


def test_round_robin_leaderboard():
    env_cfg = EnvConfig.from_dict({"scenario": {"time_limit": 900}})
    res = round_robin(["rule", "straight"], env_cfg, games_per_pair=2, seed=0)
    board = res["leaderboard"]
    assert board[0]["agent"] == "rule"
    assert board[0]["wins"] == 2
    assert res["matrix"]["rule"]["straight"] > 0.5


# ---------------------------------------------------------------- リーグ
def test_pfsp_prefers_hard_opponents():
    league = League(LeagueConfig(initial_opponents=["easy", "hard"], self_play_prob=0.0))
    league.score["easy"] = 0.95
    league.score["hard"] = 0.2
    w = league.weights()
    assert w["hard"] > w["easy"]
    rng = np.random.default_rng(0)
    picks = [league.sample(rng) for _ in range(500)]
    assert picks.count("hard") > picks.count("easy") * 5


def test_league_snapshot_eviction_and_state_roundtrip():
    league = League(LeagueConfig(initial_opponents=["rule"], max_snapshots=2))
    for it in (10, 20, 30):
        league.add(f"iter_{it}", f"/tmp/iter_{it}.pt", iteration=it)
    assert "iter_10" not in league.members and "iter_30" in league.members
    assert "rule" in league.members
    league.record([("rule", 1.0), ("rule", 1.0), ("self", 0.0)])
    assert league.score["rule"] > 0.5 and league.games["rule"] == 2
    assert league.ratings.get(LEARNER).rating > 1500
    restored = League.from_state_dict(json.loads(json.dumps(league.state_dict())))
    assert restored.table() == league.table()
