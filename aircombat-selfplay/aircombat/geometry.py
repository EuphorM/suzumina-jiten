"""座標系と幾何計算の補助関数。

ワールド座標: 青陣営が x<0 側、赤陣営が x>0 側。z は高度（上向き）。
方位角 heading は +x 軸から反時計回り（左旋回が正）。

陣営座標 (team frame): 各陣営から見て「敵が +x 方向」になる座標系。
青陣営はワールド座標そのもの、赤陣営は z 軸まわりに 180 度回転したもの。
回転なので左右が入れ替わらず、同じ方策をどちらの陣営にも使える（自己対戦の前提）。
"""

from __future__ import annotations

import numpy as np

G = 9.80665
TWO_PI = 2.0 * np.pi


def wrap_angle(a):
    """角度を [-pi, pi) に正規化する。"""
    return (np.asarray(a) + np.pi) % TWO_PI - np.pi


def velocity_vector(speed, heading, gamma):
    """速さ・方位角・経路角から速度ベクトル (..., 3) を作る。"""
    speed, heading, gamma = np.broadcast_arrays(
        np.asarray(speed, dtype=float), np.asarray(heading, dtype=float), np.asarray(gamma, dtype=float)
    )
    out = np.empty(speed.shape + (3,))
    hs = speed * np.cos(gamma)
    out[..., 0] = hs * np.cos(heading)
    out[..., 1] = hs * np.sin(heading)
    out[..., 2] = speed * np.sin(gamma)
    return out


def team_sign(team: int) -> float:
    """陣営座標への変換で x, y に掛ける符号（青 +1 / 赤 -1）。"""
    return 1.0 if team == 0 else -1.0


def to_team_frame(team: int, vec):
    """ワールド座標のベクトル (..., 3) を陣営座標に変換する（逆変換も同じ式）。"""
    v = np.array(vec, dtype=float, copy=True)
    if team == 1:
        v[..., 0] *= -1.0
        v[..., 1] *= -1.0
    return v


def heading_to_team_frame(team: int, heading):
    return wrap_angle(np.asarray(heading) + (np.pi if team == 1 else 0.0))


def rotate_yaw(vec, heading):
    """ベクトル (..., 3) を方位角 heading の機体水平座標系（前方 x, 左 y, 上 z）に回す。"""
    vec = np.asarray(vec, dtype=float)
    c = np.cos(heading)
    s = np.sin(heading)
    x = vec[..., 0]
    y = vec[..., 1]
    rx = c * x + s * y
    out = np.empty(rx.shape + (3,))
    out[..., 0] = rx
    out[..., 1] = c * y - s * x
    out[..., 2] = vec[..., 2]
    return out


def norm(vec, axis=-1, keepdims=False):
    if axis == -1 and not keepdims:
        vec = np.asarray(vec, dtype=float)
        return np.sqrt(np.einsum("...i,...i->...", vec, vec))
    return np.sqrt(np.sum(np.square(vec), axis=axis, keepdims=keepdims))


def safe_normalize(vec, axis=-1, eps=1e-9):
    n = norm(vec, axis=axis, keepdims=True)
    return vec / np.maximum(n, eps)
