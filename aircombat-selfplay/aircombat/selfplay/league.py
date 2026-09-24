"""対戦相手プール（リーグ）とマッチメイク。

- 最新の自分との対戦（self_play_prob）
- 固定の相手（ルールベースなど）と過去のスナップショットからの抽選
  抽選は PFSP（Prioritized Fictitious Self-Play）: 勝ちにくい相手ほど選ばれやすい
  重み = max(pfsp_floor, (1 - 学習者の対戦スコア)^pfsp_alpha)
- Glicko-2 で学習者とプールのレーティングを追跡する（公式の定量評価と同じ方式）

シミュレータには依存しないので、公式シミュレータでの自己対戦学習にも流用できる。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .glicko2 import Rating, RatingTable

LEARNER = "learner"


@dataclass
class LeagueConfig:
    self_play_prob: float = 0.35
    pfsp_alpha: float = 2.0
    pfsp_floor: float = 0.05
    max_snapshots: int = 20
    score_ema: float = 0.1  # 学習者の対戦スコアの指数移動平均の係数
    initial_opponents: list[str] = field(default_factory=lambda: ["rule", "rule_aggressive", "rule_defensive", "random"])


class League:
    def __init__(self, cfg: LeagueConfig):
        self.cfg = cfg
        self.members: dict[str, dict] = {}
        self.score: dict[str, float] = {}
        self.games: dict[str, int] = {}
        self.ratings = RatingTable()
        self.ratings.get(LEARNER)
        for name in cfg.initial_opponents:
            self.add(name, name, kind="fixed")

    def add(self, name: str, spec: str, kind: str = "snapshot", iteration: int = 0) -> None:
        self.members[name] = {"spec": spec, "kind": kind, "iteration": iteration}
        self.score.setdefault(name, 0.5)
        self.games.setdefault(name, 0)
        if kind == "snapshot":
            # スナップショットは作成時点の学習者のレーティングを引き継ぐ
            lr = self.ratings.get(LEARNER)
            self.ratings.ratings[name] = Rating(lr.rating, max(lr.rd, 60.0), lr.vol)
            self._evict()
        else:
            self.ratings.get(name)

    def _evict(self) -> list[str]:
        snaps = sorted(
            (n for n, m in self.members.items() if m["kind"] == "snapshot"),
            key=lambda n: self.members[n]["iteration"],
        )
        removed = []
        while len(snaps) > self.cfg.max_snapshots:
            name = snaps.pop(0)
            self.members.pop(name)
            removed.append(name)
        return removed

    def spec_of(self, name: str) -> str:
        return "self" if name == "self" else self.members[name]["spec"]

    def weights(self) -> dict[str, float]:
        c = self.cfg
        return {n: max(c.pfsp_floor, (1.0 - self.score[n]) ** c.pfsp_alpha) for n in self.members}

    def sample(self, rng: np.random.Generator) -> str:
        if not self.members or rng.random() < self.cfg.self_play_prob:
            return "self"
        w = self.weights()
        names = list(w)
        p = np.array([w[n] for n in names])
        return names[int(rng.choice(len(names), p=p / p.sum()))]

    def record(self, results: list[tuple[str, float]]) -> None:
        """results: [(相手の名前, 学習者のスコア)]。自己対戦（"self"）は無視する。"""
        games = []
        for name, score in results:
            if name == "self" or name not in self.members:
                continue
            a = self.cfg.score_ema
            self.score[name] = (1 - a) * self.score[name] + a * score
            self.games[name] += 1
            games.append((LEARNER, name, score))
        if games:
            self.ratings.update_period(games)

    def table(self) -> list[dict]:
        rows = []
        for name, r in self.ratings.leaderboard():
            if name != LEARNER and name not in self.members:
                continue
            rows.append(
                {
                    "name": name,
                    "rating": round(r.rating, 1),
                    "rd": round(r.rd, 1),
                    "learner_score": None if name == LEARNER else round(self.score[name], 3),
                    "games": None if name == LEARNER else self.games[name],
                }
            )
        return rows

    def state_dict(self) -> dict:
        return {
            "cfg": asdict(self.cfg),
            "members": self.members,
            "score": self.score,
            "games": self.games,
            "ratings": self.ratings.to_dict(),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "League":
        league = cls.__new__(cls)
        league.cfg = LeagueConfig(**d["cfg"])
        league.members = d["members"]
        league.score = d["score"]
        league.games = d["games"]
        league.ratings = RatingTable.from_dict(d["ratings"])
        return league
