"""空戦シミュレーションの中核。

機体は質点モデル（速さ・方位角・経路角）で、行動判断ごとに与えられた
旋回率・目標経路角・スロットルのコマンドに従って sim_dt 刻みで積分する。
誘導弾はブースト→抗力による減速の速度モデルと比例航法で飛翔し、
中間誘導は味方レーダーの航跡（データリンク）、終末はアクティブシーカーで行う。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import EnvConfig
from ..geometry import G, norm, safe_normalize, velocity_vector, wrap_angle
from .scenario import InitialState

CAUSE_MISSILE = "missile"
CAUSE_CRASH = "crash"
CAUSE_OUT = "out_of_bounds"

M_UNUSED, M_FLYING, M_HIT, M_MISS = 0, 1, 2, 3

PITCH_GAIN = 1.0  # 経路角コマンドへの追従ゲイン [1/s]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(m, 3) 同士の外積（np.cross より呼び出しが軽い）。"""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


@dataclass
class Event:
    time: float
    kind: str  # "launch" | "hit" | "miss" | "destroyed"
    team: int  # launch/hit/miss は射手の陣営、destroyed は被撃破機の陣営
    actor: int  # 射手の機体番号（なければ -1）
    target: int  # 目標または被撃破機の機体番号
    missile: int = -1
    detail: str = ""


class Simulation:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        sc = cfg.scenario
        self.nf = nf = sc.num_fighters
        self.n = n = 2 * (nf + 1)
        self.dt = sc.sim_dt
        self.is_2d = sc.mode == "2d"
        self.team = np.repeat([0, 1], nf + 1)
        self.escort_idx = np.array([nf, 2 * nf + 1])
        self.is_escort = np.zeros(n, dtype=bool)
        self.is_escort[self.escort_idx] = True
        self.fighter_idx = [np.arange(0, nf), np.arange(nf + 1, 2 * nf + 1)]
        self.team_members = [np.arange(0, nf + 1), np.arange(nf + 1, n)]
        ms = cfg.missile
        self._seeker_cos = np.cos(np.radians(ms.seeker_fov_deg))
        self._gimbal_cos = np.cos(np.radians(ms.gimbal_limit_deg))
        self.time = 0.0

    # ------------------------------------------------------------------ reset
    def reset(self, init: InitialState) -> None:
        n = self.n
        self.time = 0.0
        self.pos = init.pos.astype(float).copy()
        self.speed = init.speed.astype(float).copy()
        self.heading = wrap_angle(init.heading.astype(float))
        self.gamma = init.gamma.astype(float).copy()
        if self.is_2d:
            self.gamma[:] = 0.0
        self.vel = velocity_vector(self.speed, self.heading, self.gamma)
        self.perf = {k: v.astype(float).copy() for k, v in init.perf.items()}
        self._turn_rate_k = G * np.sqrt(np.maximum(self.perf["max_g"] ** 2 - 1.0, 0.0))  # 最大旋回率 = k / V
        self._pitch_rate_k = G * self.perf["max_g"] * 0.5  # 最大ピッチ率 = k / V
        self.escort_center = init.escort_center.astype(float).copy()
        self.escort_goal = init.escort_goal.astype(float).copy()
        self.escort_alt = init.escort_alt.astype(float).copy()
        self.escort_orbiting = np.array([self.cfg.escort.behavior == "orbit"] * 2)

        self.alive = np.ones(n, dtype=bool)
        self.death_time = np.full(n, np.nan)
        self.death_cause = [""] * n
        self.killer = np.full(n, -1, dtype=int)
        self.hits_taken = np.zeros(n, dtype=int)
        self.missiles_left = init.missiles.astype(int).copy()
        self.initial_missiles = self.missiles_left.copy()
        self.last_launch = np.full(n, -np.inf)

        self.cmd_turn = np.zeros(n)
        self.cmd_gamma = np.zeros(n)
        self.cmd_throttle = np.zeros(n, dtype=int)  # -1: 減速, 0: 速度維持, 1: 増速

        cap = max(int(self.missiles_left.sum()), 1)
        self.m_count = 0
        self.m_status = np.zeros(cap, dtype=int)
        self.m_active = np.zeros(cap, dtype=bool)
        self.m_pos = np.zeros((cap, 3))
        self.m_vel = np.zeros((cap, 3))
        self.m_shooter = np.full(cap, -1, dtype=int)
        self.m_target = np.full(cap, -1, dtype=int)
        self.m_team = np.zeros(cap, dtype=int)
        self.m_launch_time = np.zeros(cap)
        self.m_end_time = np.full(cap, np.nan)
        self.m_locked = np.zeros(cap, dtype=bool)
        self.m_aim_pos = np.zeros((cap, 3))
        self.m_aim_vel = np.zeros((cap, 3))
        self.m_aim_time = np.zeros(cap)

        self.launched = np.zeros(2, dtype=int)
        self.hits = np.zeros(2, dtype=int)
        self.events: list[Event] = []

        self.radar_det = np.zeros((n, n), dtype=bool)
        self.irst_det = np.zeros((n, n), dtype=bool)
        self.team_track = np.zeros((2, n), dtype=bool)
        self.last_seen_time = np.full((2, n), -np.inf)
        self.last_seen_pos = np.zeros((2, n, 3))
        self.last_seen_vel = np.zeros((2, n, 3))
        self.update_sensors()

    # --------------------------------------------------------------- commands
    def launchable(self, shooters: np.ndarray) -> np.ndarray:
        """shooters (k,) の各機が各機体 (n,) に今射撃できるかの (k, n) 真偽行列。"""
        shooters = np.asarray(shooters, dtype=int)
        p = self.perf
        rel = self.pos[None, :, :] - self.pos[shooters, None, :]
        dist = norm(rel)
        fwd = safe_normalize(self.vel[shooters])
        cosang = np.sum(rel * fwd[:, None, :], axis=-1) / np.maximum(dist, 1e-9)
        ready = (
            self.alive[shooters]
            & ~self.is_escort[shooters]
            & (self.missiles_left[shooters] > 0)
            & (self.time - self.last_launch[shooters] >= p["launch_interval"][shooters] - 1e-6)
        )
        shooter_team = self.team[shooters]
        ok = (
            ready[:, None]
            & self.alive[None, :]
            & (self.team[None, :] != shooter_team[:, None])
            & self.team_track[shooter_team]
            & (dist <= p["max_launch_range"][shooters, None])
            & (cosang >= np.cos(p["launch_fov"][shooters])[:, None])
        )
        return ok

    def launch(self, shooter: int, target: int) -> int:
        """誘導弾を発射する。発射できなければ -1 を返す。"""
        if not self.launchable(np.array([shooter]))[0, target]:
            return -1
        mid = self.m_count
        self.m_count += 1
        team = int(self.team[shooter])
        self.m_status[mid] = M_FLYING
        self.m_active[mid] = True
        self.m_pos[mid] = self.pos[shooter]
        self.m_vel[mid] = self.vel[shooter]
        self.m_shooter[mid] = shooter
        self.m_target[mid] = target
        self.m_team[mid] = team
        self.m_launch_time[mid] = self.time
        self.m_locked[mid] = False
        self.m_aim_pos[mid] = self.pos[target]
        self.m_aim_vel[mid] = self.vel[target]
        self.m_aim_time[mid] = self.time
        self.missiles_left[shooter] -= 1
        self.last_launch[shooter] = self.time
        self.launched[team] += 1
        self.events.append(Event(self.time, "launch", team, int(shooter), int(target), mid))
        return mid

    # ------------------------------------------------------------------ step
    def step(self) -> list[Event]:
        """sim_dt だけ進める。この刻みで起きたイベントを返す。"""
        dt = self.dt
        start = len(self.events)
        self._escort_autopilot()
        old_pos = self.pos.copy()
        self._update_aircraft(dt)
        self.time += dt
        self._update_missiles(dt, old_pos)
        self._check_limits()
        return self.events[start:]

    def _update_aircraft(self, dt: float) -> None:
        a = self.alive
        if not a.any():
            return
        p = self.perf
        V = self.speed
        gam = self.gamma
        if self.is_2d:
            gdot = 0.0
            gam_new = gam
        else:
            gdot_max = self._pitch_rate_k / V
            gdot = np.minimum(np.maximum(PITCH_GAIN * (self.cmd_gamma - gam), -gdot_max), gdot_max)
            gam_new = np.minimum(np.maximum(gam + gdot * dt, -p["max_pitch"]), p["max_pitch"])
        psidot = np.minimum(np.maximum(self.cmd_turn, -1.0), 1.0) * (self._turn_rate_k / V)
        psi_new = (self.heading + psidot * dt + np.pi) % (2.0 * np.pi) - np.pi

        # 重力と旋回による減速を、スロットルで補う（速度維持は損失を打ち消す）
        a_lat = V * (np.abs(psidot) + np.abs(gdot))
        loss = G * np.sin(gam_new) + p["turn_drag"] * a_lat
        hold = np.minimum(np.maximum(loss, -p["max_decel"]), p["max_accel"])
        thr = np.where(self.cmd_throttle > 0, p["max_accel"], np.where(self.cmd_throttle < 0, -p["max_decel"], hold))
        V_new = np.minimum(np.maximum(V + (thr - loss) * dt, p["min_speed"]), p["max_speed"])
        vel_new = velocity_vector(V_new, psi_new, gam_new)
        pos_new = self.pos + 0.5 * (self.vel + vel_new) * dt

        ceiling = self.cfg.arena.alt_max
        over = pos_new[:, 2] > ceiling
        if over.any():
            pos_new[over, 2] = ceiling
            gam_new = np.where(over, np.minimum(gam_new, 0.0), gam_new)
            vel_new = velocity_vector(V_new, psi_new, gam_new)

        if a.all():
            self.pos, self.vel, self.speed, self.heading, self.gamma = pos_new, vel_new, V_new, psi_new, gam_new
            return
        a3 = a[:, None]
        self.pos = np.where(a3, pos_new, self.pos)
        self.vel = np.where(a3, vel_new, self.vel)
        self.speed = np.where(a, V_new, V)
        self.heading = np.where(a, psi_new, self.heading)
        self.gamma = np.where(a, gam_new, gam)

    def _update_missiles(self, dt: float, old_pos: np.ndarray) -> None:
        """誘導弾を dt 進める。誘導計算は guidance_dt 刻みで、目標位置は機体の刻みの間を線形補間する。"""
        idx = np.flatnonzero(self.m_active)
        if idx.size == 0:
            return
        # 終末段階（目標まで terminal_range 以内）の誘導弾があるときだけ細かく刻む
        ms = self.cfg.missile
        rel = self.pos[self.m_target[idx]] - self.m_pos[idx]
        near = np.einsum("ij,ij->i", rel, rel).min() <= ms.terminal_range**2
        inner = max(1, int(np.ceil(dt / ms.guidance_dt - 1e-9))) if near else 1
        h = dt / inner
        t_start = self.time - dt
        new_pos = self.pos
        for s in range(inner):
            idx = np.flatnonzero(self.m_active)
            if idx.size == 0:
                return
            tgt = self.m_target[idx]
            a0 = s / inner
            a1 = (s + 1) / inner
            tpos0 = old_pos[tgt] * (1.0 - a0) + new_pos[tgt] * a0
            tpos1 = old_pos[tgt] * (1.0 - a1) + new_pos[tgt] * a1
            self._missile_substep(idx, tgt, h, t_start + h * (s + 1), tpos0, tpos1)

    def _missile_substep(self, idx, tgt, h: float, t_end: float, tpos0, tpos1) -> None:
        ms = self.cfg.missile
        P = self.m_pos[idx]
        Vv = self.m_vel[idx]
        spd = np.maximum(np.sqrt(np.einsum("ij,ij->i", Vv, Vv)), 1e-6)
        vhat = Vv / spd[:, None]
        t_alive = self.alive[tgt]
        tvel = self.vel[tgt]
        team = self.m_team[idx]

        # シーカー: 捕捉・ロック外れの判定
        locked = self.m_locked[idx]
        rel = tpos1 - P
        dist = np.maximum(np.sqrt(np.einsum("ij,ij->i", rel, rel)), 1e-6)
        cosang = np.einsum("ij,ij->i", rel, vhat) / dist
        acquire = ~locked & t_alive & (dist <= ms.seeker_range) & (cosang >= self._seeker_cos)
        locked = (locked | acquire) & t_alive & (cosang >= self._gimbal_cos)

        # 目標情報: ロック中はシーカー、未ロックなら味方レーダーの航跡（データリンク）
        aim_pos = self.m_aim_pos[idx]
        aim_vel = self.m_aim_vel[idx]
        aim_time = self.m_aim_time[idx]
        fresh = locked | (self.team_track[team, tgt] & t_alive)
        if fresh.any():
            aim_pos[fresh] = tpos1[fresh]
            aim_vel[fresh] = tvel[fresh]
            aim_time[fresh] = t_end
        pred = aim_pos + aim_vel * (t_end - aim_time)[:, None]

        # 比例航法 (PN)。目標が大きく横・後ろにあるときは純追尾で向きを変える
        r = pred - P
        rr = np.maximum(np.einsum("ij,ij->i", r, r), 1.0)
        rn = np.sqrt(rr)
        rhat = r / rn[:, None]
        vr = aim_vel - Vv
        omega = _cross(r, vr) / rr[:, None]
        closing = -np.einsum("ij,ij->i", r, vr) / rn
        acc = (ms.nav_gain * np.abs(closing))[:, None] * _cross(omega, rhat)
        acc -= np.einsum("ij,ij->i", acc, vhat)[:, None] * vhat
        amax = ms.max_g * G * np.minimum(1.0, (spd / ms.agility_ref_speed) ** 2)
        cos_aim = np.einsum("ij,ij->i", rhat, vhat)
        pursuit = cos_aim < 0.5
        if pursuit.any():
            perp = rhat[pursuit] - cos_aim[pursuit, None] * vhat[pursuit]
            acc[pursuit] = safe_normalize(perp) * amax[pursuit, None]
        if self.is_2d:
            acc[:, 2] = 0.0
        amag = np.sqrt(np.einsum("ij,ij->i", acc, acc))
        scale = np.minimum(1.0, amax / np.maximum(amag, 1e-9))
        acc *= scale[:, None]
        amag *= scale

        tof0 = t_end - h - self.m_launch_time[idx]
        thrust = np.where(tof0 < ms.boost_time, ms.boost_accel, 0.0)
        k = ms.drag_k0 * np.exp(-np.maximum(P[:, 2], 0.0) / ms.scale_height)
        spd_new = np.maximum(spd + (thrust - k * spd * spd - ms.induced_drag * amag) * h, 1.0)
        dvec = Vv + acc * h
        dvec /= np.maximum(np.sqrt(np.einsum("ij,ij->i", dvec, dvec)), 1e-9)[:, None]
        Vv_new = dvec * spd_new[:, None]
        P_new = P + 0.5 * (Vv + Vv_new) * h

        # 刻み内の最接近距離で命中判定（両者が等速直線運動すると近似）
        rel0 = P - tpos0
        d = (P_new - tpos1) - rel0
        sfrac = np.clip(-np.einsum("ij,ij->i", rel0, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-9), 0.0, 1.0)
        c = rel0 + sfrac[:, None] * d
        hit = t_alive & (np.einsum("ij,ij->i", c, c) <= ms.lethal_radius**2)

        self.m_pos[idx] = P_new
        self.m_vel[idx] = Vv_new
        self.m_locked[idx] = locked
        self.m_aim_pos[idx] = aim_pos
        self.m_aim_vel[idx] = aim_vel
        self.m_aim_time[idx] = aim_time

        tof1 = t_end - self.m_launch_time[idx]
        spent = ((tof1 > ms.boost_time) & (spd_new < ms.min_speed)) | (tof1 >= ms.max_flight_time)
        spent |= P_new[:, 2] < self.cfg.arena.alt_min
        done = hit | spent | ~t_alive
        if not done.any():
            return
        for j in np.flatnonzero(hit):
            mid = int(idx[j])
            victim = int(tgt[j])
            shooter = int(self.m_shooter[mid])
            mteam = int(self.m_team[mid])
            self._finish_missile(mid, M_HIT, t_end)
            self.hits[mteam] += 1
            kind = "escort" if self.is_escort[victim] else "fighter"
            self.events.append(Event(t_end, "hit", mteam, shooter, victim, mid, kind))
            if self.alive[victim]:
                self.hits_taken[victim] += 1
                need = self.cfg.escort.hits_to_kill if self.is_escort[victim] else 1
                if self.hits_taken[victim] >= need:
                    self._destroy(victim, CAUSE_MISSILE, shooter, t_end)
        for j in np.flatnonzero(done & ~hit):
            mid = int(idx[j])
            self._finish_missile(mid, M_MISS, t_end)
            reason = "target_destroyed" if not t_alive[j] else "spent"
            self.events.append(
                Event(t_end, "miss", int(self.m_team[mid]), int(self.m_shooter[mid]), int(tgt[j]), mid, reason)
            )

    def _finish_missile(self, mid: int, status: int, time: float) -> None:
        self.m_active[mid] = False
        self.m_status[mid] = status
        self.m_end_time[mid] = time

    def _destroy(self, i: int, cause: str, killer: int = -1, time: float | None = None) -> None:
        time = self.time if time is None else time
        self.alive[i] = False
        self.death_time[i] = time
        self.death_cause[i] = cause
        self.killer[i] = killer
        self.events.append(Event(time, "destroyed", int(self.team[i]), int(killer), int(i), -1, cause))

    def _check_limits(self) -> None:
        ar = self.cfg.arena
        crash = self.alive & (self.pos[:, 2] < ar.alt_min)
        for i in np.flatnonzero(crash):
            self._destroy(int(i), CAUSE_CRASH)
        if self.cfg.scenario.out_of_bounds == "destroy":
            out = (
                self.alive
                & ~self.is_escort
                & ((np.abs(self.pos[:, 0]) > ar.x_limit) | (np.abs(self.pos[:, 1]) > ar.y_limit))
            )
            for i in np.flatnonzero(out):
                self._destroy(int(i), CAUSE_OUT)

    # ---------------------------------------------------------------- escort
    def _escort_autopilot(self) -> None:
        ec = self.cfg.escort
        ar = self.cfg.arena
        for k in (0, 1):
            i = int(self.escort_idx[k])
            if not self.alive[i]:
                continue
            px, py, pz = (float(v) for v in self.pos[i])
            desired = None
            if ec.evade_range > 0:
                best = ec.evade_range
                for j in self.fighter_idx[1 - k]:
                    if self.alive[j] and self.team_track[k, j]:
                        d = math.hypot(self.pos[j, 0] - px, self.pos[j, 1] - py)
                        if d < best:
                            best = d
                            desired = math.atan2(py - self.pos[j, 1], px - self.pos[j, 0])
            if desired is None and not self.escort_orbiting[k]:
                gx, gy = self.escort_goal[k]
                if math.hypot(gx - px, gy - py) <= ec.orbit_radius:
                    self.escort_orbiting[k] = True
                    self.escort_center[k] = self.escort_goal[k]
                else:
                    desired = math.atan2(gy - py, gx - px)
            if desired is None:
                cx, cy = self.escort_center[k]
                r = max(math.hypot(px - cx, py - cy), 1.0)
                theta = math.atan2(py - cy, px - cx)
                desired = theta + math.pi / 2.0 + math.atan(2.0 * (r - ec.orbit_radius) / ec.orbit_radius)
            # 空域の端に近づいたら中央へ戻す
            margin = 5_000.0
            if abs(px) > ar.x_limit - margin or abs(py) > ar.y_limit - margin:
                desired = math.atan2(-py, -px)
            err = (desired - float(self.heading[i]) + math.pi) % (2.0 * math.pi) - math.pi
            self.cmd_turn[i] = min(max(err / 0.35, -1.0), 1.0)
            self.cmd_gamma[i] = min(max((float(self.escort_alt[k]) - pz) / 3_000.0, -0.15), 0.15)
            self.cmd_throttle[i] = 0

    # --------------------------------------------------------------- sensors
    def update_sensors(self) -> None:
        """レーダー・IRST の探知と、陣営で共有する航跡を更新する（行動判断の直前に呼ぶ）。"""
        p = self.perf
        rel = self.pos[None, :, :] - self.pos[:, None, :]  # rel[i, j] = pos_j - pos_i
        dist = norm(rel)
        fwd = safe_normalize(self.vel)
        cosang = np.sum(rel * fwd[:, None, :], axis=-1) / np.maximum(dist, 1e-9)
        pair = self.alive[:, None] & self.alive[None, :] & (self.team[:, None] != self.team[None, :])
        radar_range = p["radar_range"][:, None] * p["rcs"][None, :] ** 0.25
        self.radar_det = (
            pair
            & (p["has_radar"][:, None] > 0)
            & (dist <= radar_range)
            & (cosang >= np.cos(p["radar_fov"])[:, None])
        )
        self.irst_det = (
            pair
            & (p["has_irst"][:, None] > 0)
            & (dist <= p["irst_range"][:, None])
            & (cosang >= np.cos(p["irst_fov"])[:, None])
        )
        self.dist = dist
        for k in (0, 1):
            tracked = self.radar_det[self.team_members[k]].any(axis=0)
            self.team_track[k] = tracked
            self.last_seen_time[k, tracked] = self.time
            self.last_seen_pos[k, tracked] = self.pos[tracked]
            self.last_seen_vel[k, tracked] = self.vel[tracked]

    # --------------------------------------------------------------- queries
    def active_missiles(self) -> np.ndarray:
        return np.flatnonzero(self.m_active)

    def can_team_win(self, team: int) -> bool:
        """まだ相手の護衛対象機を撃墜しうるか（武装の残った戦闘機か飛翔中の誘導弾がある）。"""
        f = self.fighter_idx[team]
        if np.any(self.alive[f] & (self.missiles_left[f] > 0)):
            return True
        return bool(np.any(self.m_active & (self.m_team == team)))
