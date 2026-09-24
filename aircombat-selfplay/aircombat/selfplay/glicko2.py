"""Glicko-2 レーティング（公式評価の定量評価と同じ方式）。

Mark E. Glickman, "Example of the Glicko-2 system" の手順をそのまま実装している。
スコアは勝ち 1 / 引き分け 0.5 / 負け 0 だけでなく、第5回ルールの 0〜1 の連続値スコアもそのまま使える。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SCALE = 173.7178


@dataclass
class Rating:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06

    def to_dict(self) -> dict:
        return {"rating": self.rating, "rd": self.rd, "vol": self.vol}

    @classmethod
    def from_dict(cls, d: dict) -> "Rating":
        return cls(d["rating"], d["rd"], d["vol"])


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def expected_score(a: Rating, b: Rating) -> float:
    """a が b に対して期待されるスコア。"""
    mu, mu_j = (a.rating - 1500.0) / SCALE, (b.rating - 1500.0) / SCALE
    phi_j = b.rd / SCALE
    return _E(mu, mu_j, phi_j)


def update(player: Rating, results: list[tuple[Rating, float]], tau: float = 0.5, eps: float = 1e-6) -> Rating:
    """1 レーティング期間の対戦結果 [(相手のレーティング, 自分のスコア)] で更新した新しい Rating を返す。"""
    mu = (player.rating - 1500.0) / SCALE
    phi = player.rd / SCALE
    sigma = player.vol
    if not results:
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        return Rating(player.rating, min(phi_star * SCALE, 350.0), sigma)

    v_inv = 0.0
    delta_sum = 0.0
    for opp, score in results:
        mu_j = (opp.rating - 1500.0) / SCALE
        phi_j = opp.rd / SCALE
        g = _g(phi_j)
        E = _E(mu, mu_j, phi_j)
        v_inv += g * g * E * (1.0 - E)
        delta_sum += g * (score - E)
    v = 1.0 / v_inv
    delta = v * delta_sum

    # 変動率 sigma の更新（Illinois 法）
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau
    fA, fB = f(A), f(B)
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    sigma_new = math.exp(A / 2.0)

    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + phi_new * phi_new * delta_sum
    return Rating(mu_new * SCALE + 1500.0, phi_new * SCALE, sigma_new)


class RatingTable:
    """複数プレイヤーのレーティングをまとめて管理する（1 回の update_period が 1 レーティング期間）。"""

    def __init__(self, tau: float = 0.5):
        self.tau = tau
        self.ratings: dict[str, Rating] = {}

    def get(self, name: str) -> Rating:
        if name not in self.ratings:
            self.ratings[name] = Rating()
        return self.ratings[name]

    def update_period(self, games: list[tuple[str, str, float]], frozen: set[str] | None = None) -> None:
        """games: [(A, B, A のスコア)]。frozen に含まれるプレイヤーは更新しない。"""
        frozen = frozen or set()
        snapshot = {name: Rating(r.rating, r.rd, r.vol) for name, r in self.ratings.items()}
        per_player: dict[str, list[tuple[Rating, float]]] = {}
        for a, b, score in games:
            ra = snapshot.get(a) or self.get(a)
            rb = snapshot.get(b) or self.get(b)
            per_player.setdefault(a, []).append((rb, score))
            per_player.setdefault(b, []).append((ra, 1.0 - score))
        for name, res in per_player.items():
            if name in frozen:
                continue
            self.ratings[name] = update(snapshot.get(name) or self.get(name), res, self.tau)

    def leaderboard(self) -> list[tuple[str, Rating]]:
        return sorted(self.ratings.items(), key=lambda kv: -kv[1].rating)

    def to_dict(self) -> dict:
        return {name: r.to_dict() for name, r in self.ratings.items()}

    @classmethod
    def from_dict(cls, d: dict, tau: float = 0.5) -> "RatingTable":
        t = cls(tau)
        t.ratings = {name: Rating.from_dict(r) for name, r in d.items()}
        return t
