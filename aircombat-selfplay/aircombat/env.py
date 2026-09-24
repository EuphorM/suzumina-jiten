"""自己対戦用の二陣営環境。

    env = AirCombatEnv()
    obs = env.reset(seed=0)                 # {0: TeamObs, 1: TeamObs}
    while True:
        actions = {0: blue.act(obs[0]), 1: red.act(obs[1])}
        obs, rewards, done, info = env.step(actions)
        if done:
            print(info["outcome"].scores)
            break

行動は陣営ごとに (num_fighters, 4) の整数配列で、各行は
(旋回, 経路角, スロットル, 射撃) の選択肢番号（obs.TURN_OPTIONS などを参照）。
射撃は 0 が「撃たない」、s>0 は TeamObs.enemy_slots の s-1 番目のスロットの敵を撃つ。
観測は陣営座標なので、同じエージェントをどちらの陣営にもそのまま使える。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import EnvConfig
from .obs import (
    NEUTRAL_ACTION,
    PITCH_OPTIONS_DEG,
    THROTTLE_OPTIONS,
    TURN_OPTIONS,
    ObservationBuilder,
    TeamObs,
    action_dims,
)
from .rules import Outcome, judge
from .sim.core import CAUSE_MISSILE, Event, Simulation
from .sim.missile_range import MissileRangeTable
from .sim.scenario import InitialState, generate_initial_state


class AirCombatEnv:
    def __init__(self, config: EnvConfig | dict | None = None):
        if config is None:
            config = EnvConfig()
        elif isinstance(config, dict):
            config = EnvConfig.from_dict(config)
        self.cfg = config.validate()
        self.nf = self.cfg.scenario.num_fighters
        self.action_dims = action_dims(self.nf)
        self.sim = Simulation(self.cfg)
        self.range_table = MissileRangeTable(self.cfg.missile)
        self.builder = ObservationBuilder(self.cfg, self.range_table)
        self.rng = np.random.default_rng()
        self.outcome: Outcome | None = None
        self.done = True
        self._obs: dict[int, TeamObs] = {}
        self.invalid_fire = np.zeros(2, dtype=int)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, initial_state: InitialState | None = None) -> dict[int, TeamObs]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        init = initial_state if initial_state is not None else generate_initial_state(self.cfg, self.rng)
        self.sim.reset(init)
        self.outcome = None
        self.done = False
        self.invalid_fire[:] = 0
        self._obs = {k: self.builder.build(self.sim, k) for k in (0, 1)}
        return self._obs

    def step(self, actions: dict[int, Any]) -> tuple[dict[int, TeamObs], np.ndarray, bool, dict]:
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        sim = self.sim
        start = len(sim.events)
        for k in (0, 1):
            self._apply(k, actions[k])
        outcome = None
        for _ in range(self.cfg.substeps):
            self._guard()
            sim.step()
            outcome = judge(sim, self.cfg)
            if outcome is not None:
                break
        sim.update_sensors()
        events = sim.events[start:]
        rewards = self._rewards(events, outcome)
        self._obs = {k: self.builder.build(sim, k) for k in (0, 1)}
        self.outcome = outcome
        self.done = outcome is not None
        info = {"time": sim.time, "events": events, "outcome": outcome}
        return self._obs, rewards, self.done, info

    @property
    def last_obs(self) -> dict[int, TeamObs]:
        return self._obs

    # ------------------------------------------------------------------
    def _apply(self, k: int, action: Any) -> None:
        sim = self.sim
        a = np.asarray(action, dtype=int).reshape(self.nf, 4)
        dims = self.action_dims
        for h in range(4):
            if np.any((a[:, h] < 0) | (a[:, h] >= dims[h])):
                raise ValueError(f"action head {h} out of range: {a[:, h]}")
        F = sim.fighter_idx[k]
        alive = sim.alive[F]
        a = np.where(alive[:, None], a, np.array(NEUTRAL_ACTION)[None, :])
        sim.cmd_turn[F] = TURN_OPTIONS[a[:, 0]]
        sim.cmd_gamma[F] = 0.0 if self.cfg.scenario.mode == "2d" else np.radians(PITCH_OPTIONS_DEG[a[:, 1]])
        sim.cmd_throttle[F] = THROTTLE_OPTIONS[a[:, 2]]
        slots = self._obs[k].enemy_slots
        E = sim.team_members[1 - k]
        for e in range(self.nf):
            f = int(a[e, 3])
            if f == 0 or not alive[e]:
                continue
            col = int(slots[e, f - 1])
            if col < 0 or sim.launch(int(F[e]), int(E[col])) < 0:
                self.invalid_fire[k] += 1

    def _guard(self) -> None:
        """高度下限・空域境界の安全装置でコマンドを上書きする。"""
        sim = self.sim
        sc = self.cfg.scenario
        ar = self.cfg.arena
        fighters = ~sim.is_escort & sim.alive
        if sc.altitude_guard > 0 and sc.mode == "3d":
            low = fighters & (sim.pos[:, 2] < sc.altitude_guard)
            sim.cmd_gamma[low] = np.maximum(sim.cmd_gamma[low], np.radians(10.0))
        if sc.boundary_guard > 0:
            m = sc.boundary_guard
            x, y = sim.pos[:, 0], sim.pos[:, 1]
            vx, vy = sim.vel[:, 0], sim.vel[:, 1]
            out = (
                ((x > ar.x_limit - m) & (vx > 0))
                | ((x < -ar.x_limit + m) & (vx < 0))
                | ((y > ar.y_limit - m) & (vy > 0))
                | ((y < -ar.y_limit + m) & (vy < 0))
            )
            out &= fighters
            if out.any():
                to_center = np.arctan2(-y[out], -x[out])
                err = (to_center - sim.heading[out] + np.pi) % (2 * np.pi) - np.pi
                sim.cmd_turn[out] = np.where(err >= 0, 1.0, -1.0)

    def _rewards(self, events: list[Event], outcome: Outcome | None) -> np.ndarray:
        rc = self.cfg.reward
        sim = self.sim
        r = np.zeros(2)
        for ev in events:
            if ev.kind == "destroyed" and not sim.is_escort[ev.target]:
                r[ev.team] += rc.lose_fighter
                if ev.detail == CAUSE_MISSILE:
                    r[1 - ev.team] += rc.kill_fighter
            elif ev.kind == "hit" and ev.detail == "escort":
                r[ev.team] += rc.hit_escort
                r[1 - ev.team] -= rc.hit_escort
            elif ev.kind == "launch":
                r[ev.team] += rc.launch
        if outcome is not None:
            r += rc.terminal_scale * (np.asarray(outcome.scores) - 0.5)
        return r
