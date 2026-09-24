"""対戦の実行と記録。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agents.base import Agent
from .env import AirCombatEnv


@dataclass
class MatchResult:
    blue: str
    red: str
    seed: int | None
    winner: int  # 0: 青, 1: 赤, -1: 引き分け
    scores: tuple[float, float]
    outcome: dict
    steps: int
    replay: dict | None = field(default=None, repr=False)

    def score_of(self, team: int) -> float:
        return self.scores[team]


def run_match(
    env: AirCombatEnv,
    blue: Agent,
    red: Agent,
    seed: int | None = None,
    record: bool = False,
    blue_name: str | None = None,
    red_name: str | None = None,
) -> MatchResult:
    obs = env.reset(seed=seed)
    agents = (blue, red)
    for k, agent in enumerate(agents):
        agent.reset(k, env.cfg, seed=None if seed is None else seed * 2 + k)
    recorder = ReplayRecorder(env, blue_name or blue.name, red_name or red.name, seed) if record else None
    if recorder:
        recorder.capture()
    steps = 0
    done = False
    info: dict = {}
    while not done:
        actions = {k: agents[k].act(obs[k]) for k in (0, 1)}
        obs, _, done, info = env.step(actions)
        steps += 1
        if recorder:
            recorder.capture(info["events"])
    outcome = info["outcome"]
    return MatchResult(
        blue=blue_name or blue.name,
        red=red_name or red.name,
        seed=seed,
        winner=outcome.winner,
        scores=outcome.scores,
        outcome=outcome.to_dict(),
        steps=steps,
        replay=recorder.finish(outcome) if recorder else None,
    )


class ReplayRecorder:
    """行動判断ごとの状態を記録し、ビューア用の JSON を作る。"""

    def __init__(self, env: AirCombatEnv, blue: str, red: str, seed):
        self.env = env
        sim = env.sim
        cfg = env.cfg
        self.data = {
            "version": 1,
            "blue": blue,
            "red": red,
            "seed": seed,
            "mode": cfg.scenario.mode,
            "dt": cfg.scenario.decision_interval,
            "arena": {
                "x_limit": cfg.arena.x_limit,
                "y_limit": cfg.arena.y_limit,
                "alt_max": cfg.arena.alt_max,
            },
            "time_limit": cfg.scenario.time_limit,
            "aircraft": [
                {"id": i, "team": int(sim.team[i]), "escort": bool(sim.is_escort[i])} for i in range(sim.n)
            ],
            "frames": [],
            "events": [],
        }

    def capture(self, events=()) -> None:
        sim = self.env.sim
        frame = {
            "t": round(sim.time, 2),
            # [x, y, alt, heading(deg), alive, missiles_left]
            "a": [
                [
                    round(float(sim.pos[i, 0])),
                    round(float(sim.pos[i, 1])),
                    round(float(sim.pos[i, 2])),
                    round(float(np.degrees(sim.heading[i])), 1),
                    int(sim.alive[i]),
                    int(sim.missiles_left[i]),
                ]
                for i in range(sim.n)
            ],
            # [id, team, x, y, alt, target, locked]
            "m": [
                [
                    int(m),
                    int(sim.m_team[m]),
                    round(float(sim.m_pos[m, 0])),
                    round(float(sim.m_pos[m, 1])),
                    round(float(sim.m_pos[m, 2])),
                    int(sim.m_target[m]),
                    int(sim.m_locked[m]),
                ]
                for m in sim.active_missiles()
            ],
            # 各陣営のレーダー航跡に載っている敵機
            "trk": [np.flatnonzero(sim.team_track[k]).tolist() for k in (0, 1)],
        }
        self.data["frames"].append(frame)
        for ev in events:
            self.data["events"].append(
                {
                    "t": round(ev.time, 2),
                    "kind": ev.kind,
                    "team": ev.team,
                    "actor": ev.actor,
                    "target": ev.target,
                    "missile": ev.missile,
                    "detail": ev.detail,
                }
            )

    def finish(self, outcome) -> dict:
        self.data["outcome"] = outcome.to_dict()
        return self.data


def save_replay(replay: dict, path: str | Path) -> Path:
    """拡張子が .html ならビューア付き HTML、それ以外は JSON で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".html", ".htm"):
        from .viewer import render_html

        path.write_text(render_html(replay), encoding="utf-8")
    else:
        path.write_text(json.dumps(replay, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def summarize(results: list[MatchResult], name: str) -> dict:
    """name の視点で勝敗・平均スコアを集計する。"""
    wins = draws = losses = 0
    scores = []
    for r in results:
        team = 0 if r.blue == name else 1 if r.red == name else None
        if team is None:
            continue
        scores.append(r.scores[team])
        if r.winner == -1:
            draws += 1
        elif r.winner == team:
            wins += 1
        else:
            losses += 1
    n = len(scores)
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "mean_score": float(np.mean(scores)) if n else float("nan"),
    }
