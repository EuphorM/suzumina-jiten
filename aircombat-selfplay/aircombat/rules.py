"""勝敗判定とスコア計算（第5回ルールの推定式）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig, ScoreConfig
from .sim.core import Simulation


@dataclass
class Outcome:
    winner: int  # 0: 青, 1: 赤, -1: 引き分け
    reason: str  # "escort_destroyed" / "both_escorts_destroyed" / "time_limit" / "no_means"
    time: float
    scores: tuple[float, float]
    launched: tuple[int, int] = (0, 0)
    hits: tuple[int, int] = (0, 0)
    detail: dict = field(default_factory=dict)

    @property
    def hit_rates(self) -> tuple[float, float]:
        return tuple(h / l if l > 0 else 0.0 for h, l in zip(self.hits, self.launched))  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "time": round(self.time, 3),
            "scores": [round(s, 6) for s in self.scores],
            "launched": list(self.launched),
            "hits": list(self.hits),
            "hit_rates": [round(h, 4) for h in self.hit_rates],
            **self.detail,
        }


def speed_factor(score_cfg: ScoreConfig, win_time: float, time_limit: float) -> float:
    if win_time <= score_cfg.full_speed_time:
        return 1.0
    span = max(time_limit - score_cfg.full_speed_time, 1e-9)
    return float(np.clip((time_limit - win_time) / span, 0.0, 1.0))


def winner_score(score_cfg: ScoreConfig, hits: int, launched: int, win_time: float, time_limit: float) -> float:
    hit_rate = hits / launched if launched > 0 else 0.0
    s = (
        score_cfg.win_base
        + score_cfg.hit_bonus * hit_rate
        + score_cfg.speed_bonus * speed_factor(score_cfg, win_time, time_limit)
    )
    return round(float(np.clip(s, 0.0, 1.0)), 12)


def compute_scores(
    score_cfg: ScoreConfig, winner: int, time: float, time_limit: float, launched, hits
) -> tuple[float, float]:
    if winner < 0:
        return (score_cfg.draw_score, score_cfg.draw_score)
    w = winner_score(score_cfg, int(hits[winner]), int(launched[winner]), time, time_limit)
    lose = round(1.0 - w, 12)
    return (w, lose) if winner == 0 else (lose, w)


def judge(sim: Simulation, cfg: EnvConfig) -> Outcome | None:
    """終局していれば Outcome を、続行なら None を返す。"""
    escort_dead = ~sim.alive[sim.escort_idx]
    time_limit = cfg.scenario.time_limit
    if escort_dead.all():
        winner, reason = -1, "both_escorts_destroyed"
    elif escort_dead[1]:
        winner, reason = 0, "escort_destroyed"
    elif escort_dead[0]:
        winner, reason = 1, "escort_destroyed"
    elif sim.time >= time_limit - 1e-6:
        winner, reason = -1, "time_limit"
    elif not sim.can_team_win(0) and not sim.can_team_win(1):
        winner, reason = -1, "no_means"
    else:
        return None
    scores = compute_scores(cfg.score, winner, sim.time, time_limit, sim.launched, sim.hits)
    lost = [int(np.sum(~sim.alive[f])) for f in sim.fighter_idx]
    return Outcome(
        winner=winner,
        reason=reason,
        time=float(sim.time),
        scores=scores,
        launched=(int(sim.launched[0]), int(sim.launched[1])),
        hits=(int(sim.hits[0]), int(sim.hits[1])),
        detail={"fighters_lost": lost},
    )
