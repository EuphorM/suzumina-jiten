"""ルールベースのエージェント。

学習初期の対戦相手や、リーグに常駐させる固定の相手として使う。
観測は TeamObs.view（陣営が知り得る情報だけ）しか使わない。

各機の判断:
1. 誘導弾警報があれば、誘導弾を背にして降下・増速で回避する
2. 役割（攻撃／護衛）ごとに目標を選ぶ。攻撃役は敵の護衛対象機を最優先
3. 射撃する距離: 敵戦闘機は回避されやすいので回避不能距離 range_ne の fighter_fire_ratio 倍、
   回避しない護衛対象機は最大射程 range_max の escort_fire_ratio 倍
4. 自分たちの誘導弾がシーカーで捕捉するまでは、目標をレーダー視野に収めたまま
   斜めに進む（クランク）ことで中間誘導を支援しつつ接近を遅らせる
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import G
from ..obs import NEUTRAL_ACTION, PITCH_OPTIONS_DEG, TURN_OPTIONS, TeamObs
from .base import Agent


class RuleBasedAgent(Agent):
    name = "rule"

    def __init__(
        self,
        guards: int = 1,
        fighter_fire_ratio: float = 1.2,
        fighter_fire_basis: str = "ne",
        escort_fire_ratio: float = 0.9,
        max_per_target: int = 1,
        cruise_alt: float = 11_000.0,
        crank_deg: float = 45.0,
        self_defense_range: float = 30_000.0,
        guard_radius: float = 45_000.0,
        evade: bool = True,
        name: str | None = None,
    ):
        self.guards = guards
        self.fighter_fire_ratio = fighter_fire_ratio
        self.fighter_fire_basis = fighter_fire_basis  # "ne"（回避不能距離）/ "max"（最大射程）
        self.escort_fire_ratio = escort_fire_ratio
        self.max_per_target = max_per_target
        self.cruise_alt = cruise_alt
        self.crank = math.radians(crank_deg)
        self.self_defense_range = self_defense_range
        self.guard_radius = guard_radius
        self.evade = evade
        if name:
            self.name = name

    def reset(self, team, cfg, seed=None):
        super().reset(team, cfg, seed)
        self._roles: dict[int, str] = {}
        self._alive_key: tuple = ()
        self._seeker_range = cfg.missile.seeker_range
        self._dt = cfg.scenario.decision_interval
        self._is_2d = cfg.scenario.mode == "2d"

    # ------------------------------------------------------------------
    def act(self, obs: TeamObs) -> np.ndarray:
        v = obs.view
        nf = len(v.fighter_ids)
        out = np.tile(np.array(NEUTRAL_ACTION), (nf, 1))
        self._assign_roles(v)
        for e in range(nf):
            if not v.alive[e]:
                continue
            heading, alt, throttle, target, fire = self._decide(e, v)
            out[e] = self._to_action(e, v, heading, alt, throttle)
            if fire and target is not None:
                out[e, 3] = obs.fire_action(e, target)
        return out

    def _assign_roles(self, v) -> None:
        key = tuple(bool(a) for a in v.alive)
        if key == self._alive_key:
            return
        self._alive_key = key
        alive = [e for e in range(len(v.alive)) if v.alive[e]]
        n_guard = min(self.guards, max(len(alive) - 1, 0))
        by_dist = sorted(alive, key=lambda e: float(np.linalg.norm(v.pos[e, :2] - v.escort_pos[:2])))
        self._roles = {e: ("guard" if i < n_guard else "attack") for i, e in enumerate(by_dist)}

    def _decide(self, e: int, v):
        pos = v.pos[e]
        known = v.enemy_known & v.enemy_alive
        esc = int(np.flatnonzero(v.enemy_is_escort)[0])

        # 1. 回避
        if self.evade and v.warning_mask[e].any():
            d = v.warnings[e][0]
            heading = math.atan2(-d[1], -d[0])
            alt = max(pos[2] - 4_000.0, 3_000.0)
            return heading, alt, 1, None, False

        dist = np.linalg.norm(v.enemy_pos - pos, axis=1)
        fighters = [j for j in range(len(known)) if known[j] and j != esc]
        nearest_fighter = min(fighters, key=lambda j: dist[j]) if fighters else None

        # 2. 目標選択
        target = None
        waypoint = None
        if self._roles.get(e) == "guard":
            threats = [j for j in fighters if np.linalg.norm(v.enemy_pos[j, :2] - v.escort_pos[:2]) < self.guard_radius]
            if threats:
                target = min(threats, key=lambda j: dist[j])
            else:
                waypoint = v.escort_pos[:2] + np.array([12_000.0, 0.0])
        else:
            if nearest_fighter is not None and dist[nearest_fighter] < self.self_defense_range:
                target = nearest_fighter
            elif known[esc]:
                target = esc
            elif nearest_fighter is not None:
                target = nearest_fighter
            else:
                waypoint = np.array([75_000.0, 0.0])

        # 3. 射撃判断
        fire = False
        if target is not None and v.launchable[e, target] and v.missiles_on[target] < self.max_per_target:
            if target == esc:
                fire = dist[target] <= self.escort_fire_ratio * v.range_max[e, target]
            else:
                basis = v.range_ne if self.fighter_fire_basis == "ne" else v.range_max
                fire = dist[target] <= self.fighter_fire_ratio * basis[e, target]

        # 4. 機動
        if target is not None:
            tp = v.enemy_pos[target] + v.enemy_vel[target] * min(dist[target] / 1_500.0, 20.0)
            bearing = math.atan2(tp[1] - pos[1], tp[0] - pos[0])
            supporting = (
                (v.missiles_on[target] > 0 or fire)
                and not v.locked_on[target]
                and dist[target] > 0.8 * self._seeker_range
            )
            if supporting:
                # 目標をレーダー視野の端に置いて中間誘導を続ける
                left = bearing + self.crank
                right = bearing - self.crank
                heading = left if abs(_wrap(left - v.heading[e])) < abs(_wrap(right - v.heading[e])) else right
                throttle = 0
            else:
                heading = bearing
                throttle = 1
            alt = self.cruise_alt
        else:
            wx, wy = waypoint
            if np.hypot(wx - pos[0], wy - pos[1]) < 5_000.0:
                heading = v.heading[e] + 0.5  # 目標点の近くでは旋回待機
                throttle = 0
            else:
                heading = math.atan2(wy - pos[1], wx - pos[0])
                throttle = 1
            alt = self.cruise_alt
        heading = self._keep_inside(pos, heading, v.arena)
        return heading, alt, throttle, target, fire

    @staticmethod
    def _keep_inside(pos, heading: float, arena) -> float:
        x_lim, y_lim = arena[0], arena[1]
        margin = 12_000.0
        if abs(pos[0]) > x_lim - margin or abs(pos[1]) > y_lim - margin:
            inward = math.atan2(-pos[1], -pos[0])
            if math.cos(heading - inward) < 0.2:
                return inward
        return heading

    def _to_action(self, e: int, v, heading: float, alt: float, throttle: int) -> np.ndarray:
        err = _wrap(heading - v.heading[e])
        omega = G * math.sqrt(max(v.perf["max_g"][e] ** 2 - 1.0, 0.0)) / max(v.speed[e], 1.0)
        frac = max(-1.0, min(1.0, err / max(omega * self._dt, 1e-6)))
        turn = int(np.argmin(np.abs(TURN_OPTIONS - frac)))
        if self._is_2d:
            pitch = NEUTRAL_ACTION[1]
        else:
            gamma_deg = max(-30.0, min(30.0, (alt - v.pos[e, 2]) / 2_500.0 * 30.0))
            pitch = int(np.argmin(np.abs(PITCH_OPTIONS_DEG - gamma_deg)))
        return np.array([turn, pitch, throttle + 1, 0])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def aggressive() -> RuleBasedAgent:
    """護衛なしで全機突撃し、最大射程付近から撃ち込む。"""
    return RuleBasedAgent(
        guards=0,
        fighter_fire_basis="max",
        fighter_fire_ratio=0.8,
        escort_fire_ratio=1.0,
        max_per_target=2,
        name="rule_aggressive",
    )


def defensive() -> RuleBasedAgent:
    """2 機で護衛し、引き付けてから撃つ。"""
    return RuleBasedAgent(
        guards=2,
        fighter_fire_ratio=1.0,
        escort_fire_ratio=0.8,
        max_per_target=1,
        name="rule_defensive",
    )
