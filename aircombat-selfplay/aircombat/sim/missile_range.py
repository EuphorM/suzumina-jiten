"""誘導弾の射程推定。

誘導弾を目標方向へ直進させたときの飛翔距離 s(t) を高度・発射速度ごとに求め、
目標が視線方向に速度 v_away で遠ざかる（負なら近づく）場合に届く最大距離
R = max_t [ s(t) - v_away * t ] を（高度, 発射速度, v_away）の格子で前計算しておく。
問い合わせは三線形補間。横機動による減速は無視するので楽観的な値になる。
"""

from __future__ import annotations

import numpy as np

from ..config import MissileSpec


class MissileRangeTable:
    def __init__(
        self,
        spec: MissileSpec,
        alt_grid: np.ndarray | None = None,
        speed_grid: np.ndarray | None = None,
        v_away_grid: np.ndarray | None = None,
        dt: float = 0.5,
    ):
        self.alt_grid = np.linspace(0.0, 20_000.0, 41) if alt_grid is None else np.asarray(alt_grid, float)
        self.speed_grid = np.linspace(100.0, 500.0, 17) if speed_grid is None else np.asarray(speed_grid, float)
        self.v_away_grid = np.linspace(-700.0, 700.0, 57) if v_away_grid is None else np.asarray(v_away_grid, float)

        # s(t) を全格子点まとめて積分する
        t = np.arange(0.0, spec.max_flight_time + 1e-9, dt)
        k = spec.drag_k0 * np.exp(-self.alt_grid / spec.scale_height)[:, None]  # (A, 1)
        v = np.broadcast_to(self.speed_grid[None, :], (len(self.alt_grid), len(self.speed_grid))).copy()
        s = np.zeros_like(v)
        alive = np.ones_like(v, dtype=bool)
        dist = np.zeros((len(t),) + v.shape)
        sub = 10
        h = dt / sub
        for ti in range(1, len(t)):
            thrust = spec.boost_accel if t[ti - 1] < spec.boost_time else 0.0
            for _ in range(sub):
                v_new = np.maximum(v + (thrust - k * v * v) * h, 1.0)
                s = s + np.where(alive, 0.5 * (v + v_new) * h, 0.0)
                v = np.where(alive, v_new, v)
            if t[ti] > spec.boost_time:
                alive &= v >= spec.min_speed
            dist[ti] = np.where(alive, s, -np.inf)
        # R(alt, v0, v_away) = max_t [s(t) - v_away t]
        self.table = np.zeros(v.shape + (len(self.v_away_grid),))  # (A, V, W)
        for wi, va in enumerate(self.v_away_grid):
            reach = dist - va * t[:, None, None]
            self.table[:, :, wi] = np.maximum(reach.max(axis=0), 0.0)

    def max_range(self, alt, launch_speed, v_away) -> np.ndarray:
        """各要素ごとの最大射程 [m]。引数はブロードキャスト可能な配列。"""
        alt, launch_speed, v_away = np.broadcast_arrays(
            np.asarray(alt, float), np.asarray(launch_speed, float), np.asarray(v_away, float)
        )
        shape = alt.shape
        ai, aw = _interp_index(self.alt_grid, alt.ravel())
        vi, vw = _interp_index(self.speed_grid, launch_speed.ravel())
        wi, ww = _interp_index(self.v_away_grid, v_away.ravel())
        tab = self.table
        out = np.zeros(ai.shape)
        for da, fa in ((0, 1 - aw), (1, aw)):
            for dv, fv in ((0, 1 - vw), (1, vw)):
                out += fa * fv * ((1 - ww) * tab[ai + da, vi + dv, wi] + ww * tab[ai + da, vi + dv, wi + 1])
        return out.reshape(shape)


def _interp_index(grid: np.ndarray, x: np.ndarray):
    """等間隔格子 grid 上の補間位置（左端の添字と重み）。"""
    step = grid[1] - grid[0]
    f = np.minimum(np.maximum((x - grid[0]) / step, 0.0), len(grid) - 1.000001)
    i = f.astype(np.intp)
    return i, f - i
