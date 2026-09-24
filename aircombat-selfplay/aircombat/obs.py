"""観測の生成。

各陣営には、その陣営が知り得る情報だけを陣営座標（敵が +x 方向）で渡す。
- TeamObs.*_feats: ニューラルネット向けに正規化した機体ごとの特徴量
- TeamObs.view: ルールベースのエージェント向けの生データ（陣営座標、SI 単位）

敵機の情報は陣営のレーダー航跡（データリンクで共有）から作る。
レーダーで見えていない敵は、見失ってから track_memory 秒までは推測位置で残り、
自機の IRST が捉えている敵は方位のみが分かる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvConfig
from .geometry import norm, rotate_yaw, wrap_angle
from .sim.core import Simulation
from .sim.missile_range import MissileRangeTable

DIST_SCALE = 100_000.0
SPEED_SCALE = 300.0
ALT_SCALE = 10_000.0
MWS_SLOTS = 4

SELF_DIM = 24
ALLY_DIM = 12
ENEMY_DIM = 26
MWS_DIM = 4

# 行動の選択肢
TURN_OPTIONS = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0])  # 最大旋回率に対する割合（正が左旋回）
PITCH_OPTIONS_DEG = np.array([-30.0, -10.0, 0.0, 10.0, 30.0])  # 目標経路角
THROTTLE_OPTIONS = np.array([-1, 0, 1])  # 減速 / 速度維持 / 増速
NEUTRAL_ACTION = (3, 2, 1, 0)  # 直進・水平・速度維持・射撃なし


def action_dims(num_fighters: int) -> tuple[int, int, int, int]:
    """(旋回, 経路角, スロットル, 射撃) の選択肢数。射撃は「撃たない」+ 敵スロット数。"""
    return (len(TURN_OPTIONS), len(PITCH_OPTIONS_DEG), len(THROTTLE_OPTIONS), num_fighters + 2)


@dataclass
class TeamView:
    """ルールベース向けの生データ（陣営座標）。敵の配列はスロット順ではなく [戦闘機..., 護衛対象機] の固定順。"""

    team: int
    time: float
    time_limit: float
    fighter_ids: np.ndarray  # (nf,) 機体番号
    pos: np.ndarray  # (nf, 3)
    vel: np.ndarray  # (nf, 3)
    speed: np.ndarray  # (nf,)
    heading: np.ndarray  # (nf,)
    gamma: np.ndarray  # (nf,)
    alive: np.ndarray  # (nf,)
    missiles_left: np.ndarray  # (nf,)
    can_fire: np.ndarray  # (nf,)
    perf: dict[str, np.ndarray]  # 各 (nf,)
    escort_pos: np.ndarray  # (3,)
    escort_vel: np.ndarray  # (3,)
    escort_alive: bool
    enemy_ids: np.ndarray  # (ne,) 機体番号
    enemy_is_escort: np.ndarray  # (ne,)
    enemy_alive: np.ndarray  # (ne,) 撃墜を確認した敵は False
    enemy_tracked: np.ndarray  # (ne,) 現在レーダー航跡あり
    enemy_known: np.ndarray  # (ne,) 航跡あり、または見失ってから track_memory 秒以内
    enemy_pos: np.ndarray  # (ne, 3) 推定位置
    enemy_vel: np.ndarray  # (ne, 3) 推定速度
    enemy_age: np.ndarray  # (ne,) 最後に探知してからの時間
    irst: np.ndarray  # (nf, ne) 自機 IRST で方位を捉えているか
    irst_dir: np.ndarray  # (nf, ne, 3) IRST 方位の単位ベクトル
    launchable: np.ndarray  # (nf, ne) 今射撃できるか
    range_max: np.ndarray  # (nf, ne) 射程推定（目標が現在の速度で飛び続けた場合）
    range_ne: np.ndarray  # (nf, ne) 射程推定（目標が真後ろへ逃げた場合 = 回避不能距離）
    missiles_on: np.ndarray  # (ne,) その敵を狙って飛翔中の味方誘導弾数
    locked_on: np.ndarray  # (ne,) その敵をシーカーが捕捉している味方誘導弾があるか
    warnings: np.ndarray  # (nf, MWS_SLOTS, 3) 自機を狙う誘導弾の方位
    warning_mask: np.ndarray  # (nf, MWS_SLOTS)
    arena: tuple[float, float, float, float]  # (x_limit, y_limit, alt_min, alt_max)


@dataclass
class TeamObs:
    team: int
    time: float
    self_feats: np.ndarray  # (nf, SELF_DIM)
    ally_feats: np.ndarray  # (nf, nf, ALLY_DIM) 他の戦闘機 (nf-1) + 自陣営の護衛対象機
    enemy_feats: np.ndarray  # (nf, nf+1, ENEMY_DIM) 敵戦闘機（推定距離順）+ 敵護衛対象機
    mws_feats: np.ndarray  # (nf, MWS_SLOTS, MWS_DIM)
    action_mask: np.ndarray  # (nf, sum(action_dims)) 各行動が有効か
    enemy_slots: np.ndarray  # (nf, nf+1) 各敵スロットに対応する view.enemy_* の添字（-1 は空き）
    alive: np.ndarray  # (nf,)
    view: TeamView

    def fire_action(self, fighter: int, enemy_index: int) -> int:
        """view.enemy_* の添字で指定した敵を撃つときの射撃行動値。スロットに無ければ 0（撃たない）。"""
        hits = np.flatnonzero(self.enemy_slots[fighter] == enemy_index)
        return int(hits[0]) + 1 if hits.size else 0


class ObservationBuilder:
    def __init__(self, cfg: EnvConfig, range_table: MissileRangeTable):
        self.cfg = cfg
        self.table = range_table
        self.nf = cfg.scenario.num_fighters
        self.dims = action_dims(self.nf)
        nf = self.nf
        # 各自機から見た味方スロット（他の戦闘機 + 護衛対象機）の、陣営内での機体番号
        self._ally_cols = np.array([[j for j in range(nf) if j != e] + [nf] for e in range(nf)], dtype=int)
        f = cfg.fighter
        self._base = {
            "max_speed": f.max_speed,
            "max_g": f.max_g,
            "radar_range": f.radar_range,
            "max_launch_range": f.max_launch_range,
        }

    def build(self, sim: Simulation, k: int) -> TeamObs:
        cfg = self.cfg
        nf = self.nf
        ne = nf + 1
        ar = cfg.arena
        sc = cfg.scenario
        t = sim.time
        sgn = 1.0 if k == 0 else -1.0
        F = sim.fighter_idx[k]
        E = sim.team_members[1 - k]
        own_escort = int(sim.escort_idx[k])
        n_missiles = max(cfg.fighter.num_missiles, 1)

        flip = np.array([sgn, sgn, 1.0])

        def tf(v: np.ndarray) -> np.ndarray:
            return v * flip

        pos = tf(sim.pos)
        vel = tf(sim.vel)
        heading = sim.heading if k == 0 else wrap_angle(sim.heading + np.pi)
        alive = sim.alive
        alive_f = alive[F]
        ego_pos = pos[F]
        ego_vel = vel[F]
        ego_h = heading[F]
        ego_speed = sim.speed[F]

        # ---- 陣営として把握している敵情報
        e_alive = alive[E]
        tracked = sim.team_track[k, E] & e_alive
        age = t - sim.last_seen_time[k, E]
        remembered = ~tracked & e_alive & (age <= sc.track_memory)
        ranged = tracked | remembered
        age = np.where(remembered, age, 0.0)
        dr_pos = tf(sim.last_seen_pos[k, E] + sim.last_seen_vel[k, E] * age[:, None])
        est_pos = np.where(tracked[:, None], pos[E], dr_pos)
        est_vel = np.where(tracked[:, None], vel[E], tf(sim.last_seen_vel[k, E]))

        irst = sim.irst_det[np.ix_(F, E)] & alive_f[:, None]
        bearing_only = irst & ~ranged[None, :]
        valid = (ranged[None, :] | bearing_only) & alive_f[:, None]

        # ---- 自機と各敵の幾何（(nf, ne) の組）
        rel_est = est_pos[None, :, :] - ego_pos[:, None, :]
        rel_true = pos[E][None, :, :] - ego_pos[:, None, :]
        rel = np.where(ranged[None, :, None], rel_est, rel_true)
        dist = np.maximum(norm(rel), 1.0)
        rel_b = rotate_yaw(rel, ego_h[:, None])
        u_b = rel_b / dist[..., None]
        horiz = np.maximum(np.hypot(u_b[..., 0], u_b[..., 1]), 1e-9)
        cos_az = u_b[..., 0] / horiz
        sin_az = u_b[..., 1] / horiz
        sin_el = u_b[..., 2]
        vel_b = rotate_yaw(np.broadcast_to(est_vel[None], (nf, ne, 3)), ego_h[:, None])
        los = rel / dist[..., None]
        closure = -np.sum((est_vel[None, :, :] - ego_vel[:, None, :]) * los, axis=-1)
        e_speed = np.maximum(norm(est_vel), 1e-6)
        ev_hat = est_vel / e_speed[:, None]
        to_ego = -los
        cos_aa = np.sum(ev_hat[None] * to_ego, axis=-1)
        sin_aa = ev_hat[None, :, 0] * to_ego[..., 1] - ev_hat[None, :, 1] * to_ego[..., 0]
        fwd = ego_vel / np.maximum(norm(ego_vel), 1e-6)[:, None]
        cos_off = np.sum(fwd[:, None, :] * los, axis=-1)
        in_fov = cos_off >= np.cos(sim.perf["radar_fov"][F])[:, None]
        # 射撃可否（sim.launchable と同じ条件。レーダー航跡のある敵は rel が真の相対位置と一致する）
        p = sim.perf
        ready = (
            alive_f
            & (sim.missiles_left[F] > 0)
            & (t - sim.last_launch[F] >= p["launch_interval"][F] - 1e-6)
        )
        launchable = (
            ready[:, None]
            & tracked[None, :]
            & valid
            & (dist <= p["max_launch_range"][F][:, None])
            & (cos_off >= np.cos(p["launch_fov"][F])[:, None])
        )

        mine = sim.m_active & (sim.m_team == k)
        missiles_on = np.bincount(sim.m_target[mine], minlength=sim.n)[E]
        locked_on = np.bincount(sim.m_target[mine & sim.m_locked], minlength=sim.n)[E] > 0
        own_in_flight = np.bincount(sim.m_shooter[mine], minlength=sim.n)[F]

        alt_pair = 0.5 * (ego_pos[:, None, 2] + est_pos[None, :, 2])
        v_away = np.sum(est_vel[None, :, :] * los, axis=-1)
        both = self.table.max_range(
            np.stack([alt_pair, alt_pair]),
            np.broadcast_to(ego_speed[None, :, None], (2, nf, ne)),
            np.stack([v_away, np.broadcast_to(e_speed[None, :], (nf, ne))]),
        )
        r_max, r_ne = both[0], both[1]
        ranged2 = np.broadcast_to(ranged[None, :], (nf, ne)) & valid
        r_max = np.where(ranged2, r_max, 0.0)
        r_ne = np.where(ranged2, r_ne, 0.0)

        # ---- 誘導弾警報 (MWS): 自機を狙って飛翔中の敵誘導弾（方位のみ）
        warnings = np.zeros((nf, MWS_SLOTS, 3))
        warning_mask = np.zeros((nf, MWS_SLOTS), dtype=bool)
        n_warn = np.zeros(nf, dtype=int)
        enemy_missiles = np.flatnonzero(sim.m_active & (sim.m_team != k))
        if enemy_missiles.size:
            d = tf(sim.m_pos[enemy_missiles])[None, :, :] - ego_pos[:, None, :]  # (nf, m, 3)
            dd = np.maximum(norm(d), 1e-6)
            on_me = (sim.m_target[enemy_missiles][None, :] == F[:, None]) & alive_f[:, None]
            on_me &= dd <= sim.perf["mws_range"][F][:, None]
            n_warn = on_me.sum(axis=1)
            if n_warn.any():
                order = np.argsort(np.where(on_me, dd, np.inf), axis=1)[:, :MWS_SLOTS]
                rows_w = np.arange(nf)[:, None]
                cnt = min(MWS_SLOTS, enemy_missiles.size)
                warning_mask[:, :cnt] = on_me[rows_w, order]
                warnings[:, :cnt] = (d / dd[..., None])[rows_w, order] * warning_mask[:, :cnt, None]

        # ---- 敵の特徴量（まず敵の固定順で作り、最後にスロット順へ並べ替える）
        vf = valid.astype(np.float32)
        rf = ranged2.astype(np.float32)
        ef_n = np.zeros((nf, ne, ENEMY_DIM), dtype=np.float32)
        ef_n[..., 0] = vf
        ef_n[..., 1] = sim.is_escort[E][None, :] * vf
        ef_n[..., 2] = tracked[None, :] * vf
        ef_n[..., 3] = remembered[None, :] * vf
        ef_n[..., 4] = bearing_only
        ef_n[..., 5] = age[None, :] / max(sc.track_memory, 1e-6) * rf
        ef_n[..., 6:9] = rel_b * (rf / DIST_SCALE)[..., None]
        ef_n[..., 9] = dist / DIST_SCALE * rf
        ef_n[..., 10] = cos_az * vf
        ef_n[..., 11] = sin_az * vf
        ef_n[..., 12] = sin_el * vf
        ef_n[..., 13:16] = vel_b * (rf / SPEED_SCALE)[..., None]
        ef_n[..., 16] = closure / (2 * SPEED_SCALE) * rf
        ef_n[..., 17] = cos_aa * rf
        ef_n[..., 18] = sin_aa * rf
        ef_n[..., 19] = est_pos[None, :, 2] / ALT_SCALE * rf
        ef_n[..., 20] = in_fov * vf
        ef_n[..., 21] = launchable
        ef_n[..., 22] = missiles_on[None, :] / n_missiles * vf
        ef_n[..., 23] = locked_on[None, :] * vf
        ratio_max = np.where(r_max > 0, dist / np.maximum(r_max, 1e-6), 3.0)
        ratio_ne = np.where(r_ne > 0, dist / np.maximum(r_ne, 1e-6), 3.0)
        ef_n[..., 24] = np.minimum(ratio_max, 3.0) / 3.0 * rf
        ef_n[..., 25] = np.minimum(ratio_ne, 3.0) / 3.0 * rf

        # スロットの並び: 戦闘機を（距離判明 → 方位のみ → 不明）かつ推定距離順、最後に護衛対象機
        cat = np.where(ranged2, 0, np.where(valid, 1, 2))[:, :nf]
        key = cat * 1e9 + np.where(cat == 0, dist[:, :nf], 0.0) + np.arange(nf)[None, :] * 1e-3
        cols = np.empty((nf, ne), dtype=int)
        cols[:, :nf] = np.argsort(key, axis=1)
        cols[:, nf] = nf
        rows = np.arange(nf)[:, None]
        ef = ef_n[rows, cols]
        enemy_slots = np.where(valid[rows, cols], cols, -1)
        launch_s = launchable[rows, cols]

        # ---- 自機の特徴量
        sf = np.zeros((nf, SELF_DIM), dtype=np.float32)
        sf[:, 0] = alive_f
        sf[:, 1] = ego_pos[:, 0] / ar.x_limit
        sf[:, 2] = ego_pos[:, 1] / ar.y_limit
        sf[:, 3] = ego_pos[:, 2] / ALT_SCALE
        sf[:, 4] = ego_speed / SPEED_SCALE
        sf[:, 5] = np.cos(ego_h)
        sf[:, 6] = np.sin(ego_h)
        sf[:, 7] = np.sin(sim.gamma[F])
        sf[:, 8] = sim.missiles_left[F] / n_missiles
        sf[:, 9] = ready
        sf[:, 10] = own_in_flight / n_missiles
        sf[:, 11] = n_warn / MWS_SLOTS
        sf[:, 12] = (ar.x_limit - ego_pos[:, 0]) / (2 * ar.x_limit)
        sf[:, 13] = (ego_pos[:, 0] + ar.x_limit) / (2 * ar.x_limit)
        sf[:, 14] = (ar.y_limit - ego_pos[:, 1]) / (2 * ar.y_limit)
        sf[:, 15] = (ego_pos[:, 1] + ar.y_limit) / (2 * ar.y_limit)
        sf[:, 16] = ego_pos[:, 2] / ar.alt_max
        sf[:, 17] = t / sc.time_limit
        sf[:, 18] = p["max_speed"][F] / self._base["max_speed"]
        sf[:, 19] = p["max_g"][F] / self._base["max_g"]
        sf[:, 20] = p["radar_range"][F] / self._base["radar_range"]
        sf[:, 21] = p["max_launch_range"][F] / self._base["max_launch_range"]
        sf[:, 22] = alive_f.sum() / nf
        sf[:, 23] = np.sum(~alive[E[:nf]]) / nf
        sf[~alive_f, 1:] *= 0.0  # 撃墜された機体は特徴量を消す（ただし陣営情報は残す）
        sf[~alive_f, 22] = alive_f.sum() / nf
        sf[~alive_f, 23] = np.sum(~alive[E[:nf]]) / nf
        sf[~alive_f, 17] = t / sc.time_limit

        # ---- 味方の特徴量（他の戦闘機 + 自陣営の護衛対象機）
        ally_ids = self._ally_cols + (0 if k == 0 else nf + 1)  # (nf, nf) 機体番号
        a_rel = pos[ally_ids] - ego_pos[:, None, :]
        a_dist = norm(a_rel)
        a_alive = alive[ally_ids] & alive_f[:, None]
        af = np.zeros((nf, nf, ALLY_DIM), dtype=np.float32)
        af[..., 0] = a_alive
        af[..., 1] = sim.is_escort[ally_ids]
        af[..., 2:5] = rotate_yaw(a_rel, ego_h[:, None]) / DIST_SCALE
        af[..., 5] = a_dist / DIST_SCALE
        af[..., 6:9] = rotate_yaw(vel[ally_ids], ego_h[:, None]) / SPEED_SCALE
        af[..., 9] = sim.missiles_left[ally_ids] / n_missiles
        warn_all = np.zeros(sim.n)
        warn_all[F] = n_warn
        af[..., 10] = warn_all[ally_ids] / MWS_SLOTS
        af[..., 11] = pos[ally_ids][..., 2] / ALT_SCALE
        af *= a_alive[..., None]
        af[..., 1] = sim.is_escort[ally_ids] & a_alive

        mf = np.zeros((nf, MWS_SLOTS, MWS_DIM), dtype=np.float32)
        mf[..., 0] = warning_mask
        mf[..., 1:4] = rotate_yaw(warnings, ego_h[:, None]) * warning_mask[..., None]

        # ---- 行動マスク
        d_turn, d_pitch, d_thr, d_fire = self.dims
        mask = np.zeros((nf, sum(self.dims)), dtype=bool)
        o1, o2, o3 = d_turn, d_turn + d_pitch, d_turn + d_pitch + d_thr
        mask[:, :d_turn] = True
        if sc.mode == "2d":
            mask[:, o1 + NEUTRAL_ACTION[1]] = True
        else:
            mask[:, o1:o2] = True
        mask[:, o2:o3] = True
        mask[:, o3] = True
        mask[:, o3 + 1 :] = launch_s
        dead = ~alive_f
        if dead.any():
            mask[dead] = False
            for off, a in zip((0, o1, o2, o3), NEUTRAL_ACTION):
                mask[dead, off + a] = True

        view = TeamView(
            team=k,
            time=t,
            time_limit=sc.time_limit,
            fighter_ids=F.copy(),
            pos=ego_pos,
            vel=ego_vel,
            speed=ego_speed.copy(),
            heading=ego_h,
            gamma=sim.gamma[F].copy(),
            alive=alive_f.copy(),
            missiles_left=sim.missiles_left[F].copy(),
            can_fire=ready,
            perf={key: val[F].copy() for key, val in p.items()},
            escort_pos=pos[own_escort],
            escort_vel=vel[own_escort],
            escort_alive=bool(alive[own_escort]),
            enemy_ids=E.copy(),
            enemy_is_escort=sim.is_escort[E].copy(),
            enemy_alive=e_alive.copy(),
            enemy_tracked=tracked,
            enemy_known=ranged,
            enemy_pos=est_pos,
            enemy_vel=est_vel,
            enemy_age=age,
            irst=irst,
            irst_dir=rel_true / np.maximum(norm(rel_true), 1e-6)[..., None] * irst[..., None],
            launchable=launchable,
            range_max=r_max,
            range_ne=r_ne,
            missiles_on=missiles_on,
            locked_on=locked_on,
            warnings=warnings,
            warning_mask=warning_mask,
            arena=(ar.x_limit, ar.y_limit, ar.alt_min, ar.alt_max),
        )
        return TeamObs(
            team=k,
            time=t,
            self_feats=sf,
            ally_feats=af,
            enemy_feats=ef,
            mws_feats=mf,
            action_mask=mask,
            enemy_slots=enemy_slots,
            alive=alive_f.copy(),
            view=view,
        )
