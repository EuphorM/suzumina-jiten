"""初期配置と機体性能の生成。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EnvConfig

# 機体ごとに持つ性能パラメータ（角度はラジアン）
PERF_KEYS = (
    "min_speed",
    "max_speed",
    "max_accel",
    "max_decel",
    "max_g",
    "turn_drag",
    "max_pitch",
    "rcs",
    "radar_range",
    "radar_fov",
    "irst_range",
    "irst_fov",
    "mws_range",
    "max_launch_range",
    "launch_fov",
    "launch_interval",
    "has_radar",
    "has_irst",
)

# performance_jitter で個体差を付けるパラメータ
JITTER_KEYS = ("max_speed", "max_accel", "max_g", "radar_range", "irst_range", "max_launch_range")


@dataclass
class InitialState:
    """ワールド座標での初期状態。機体番号は [青の戦闘機..., 青の護衛対象機, 赤の戦闘機..., 赤の護衛対象機]。"""

    pos: np.ndarray  # (n, 3)
    speed: np.ndarray  # (n,)
    heading: np.ndarray  # (n,)
    gamma: np.ndarray  # (n,)
    perf: dict[str, np.ndarray]  # 各 (n,)
    missiles: np.ndarray  # (n,) int
    escort_center: np.ndarray  # (2, 2) 旋回待機の中心（ワールド座標）
    escort_goal: np.ndarray  # (2, 2) advance 時の目標点（ワールド座標）
    escort_alt: np.ndarray  # (2,) 護衛対象機の維持高度


def fighter_perf(cfg: EnvConfig) -> dict[str, float]:
    f = cfg.fighter
    return {
        "min_speed": f.min_speed,
        "max_speed": f.max_speed,
        "max_accel": f.max_accel,
        "max_decel": f.max_decel,
        "max_g": f.max_g,
        "turn_drag": f.turn_drag,
        "max_pitch": np.radians(f.max_pitch_deg),
        "rcs": f.rcs_scale,
        "radar_range": f.radar_range,
        "radar_fov": np.radians(f.radar_fov_deg),
        "irst_range": f.irst_range,
        "irst_fov": np.radians(f.irst_fov_deg),
        "mws_range": f.mws_range,
        "max_launch_range": f.max_launch_range,
        "launch_fov": np.radians(f.launch_fov_deg),
        "launch_interval": f.launch_interval,
        "has_radar": 1.0,
        "has_irst": 1.0,
    }


def escort_perf(cfg: EnvConfig) -> dict[str, float]:
    e = cfg.escort
    f = cfg.fighter
    return {
        "min_speed": e.speed,
        "max_speed": e.speed,
        "max_accel": 0.0,
        "max_decel": 0.0,
        "max_g": e.max_g,
        "turn_drag": 0.0,
        "max_pitch": np.radians(10.0),
        "rcs": e.rcs_scale,
        "radar_range": f.radar_range if e.has_radar else 0.0,
        "radar_fov": np.radians(f.radar_fov_deg),
        "irst_range": 0.0,
        "irst_fov": 0.0,
        "mws_range": 0.0,
        "max_launch_range": 0.0,
        "launch_fov": 0.0,
        "launch_interval": 0.0,
        "has_radar": 1.0 if e.has_radar else 0.0,
        "has_irst": 0.0,
    }


def _sample_team(cfg: EnvConfig, rng: np.random.Generator) -> dict:
    """陣営座標（敵が +x 方向）で 1 陣営分の初期状態を作る。"""
    sc = cfg.scenario
    nf = sc.num_fighters
    is_2d = sc.mode == "2d"
    n = nf + 1

    pos = np.zeros((n, 3))
    pos[:nf, 0] = rng.uniform(*sc.fighter_x_range, size=nf)
    pos[:nf, 1] = rng.uniform(*sc.fighter_y_range, size=nf)
    pos[:nf, 2] = sc.fixed_altitude if is_2d else rng.uniform(*sc.fighter_alt_range, size=nf)
    speed = np.zeros(n)
    speed[:nf] = rng.uniform(*sc.fighter_speed_range, size=nf)
    heading = np.zeros(n)
    noise = np.radians(sc.heading_noise_deg)
    heading[:nf] = rng.uniform(-noise, noise, size=nf)

    ec = cfg.escort
    center = np.array([rng.uniform(*sc.escort_x_range), rng.uniform(*sc.escort_y_range)])
    escort_alt = sc.fixed_altitude if is_2d else float(rng.uniform(*sc.escort_alt_range))
    if ec.behavior == "orbit":
        ang = rng.uniform(-np.pi, np.pi)
        pos[nf, :2] = center + ec.orbit_radius * np.array([np.cos(ang), np.sin(ang)])
        heading[nf] = ang + np.pi / 2.0  # 反時計回りの接線方向
    else:
        pos[nf, :2] = center
        heading[nf] = 0.0
    pos[nf, 2] = escort_alt
    speed[nf] = ec.speed
    goal = np.array([ec.advance_x, center[1]])

    base_f = fighter_perf(cfg)
    base_e = escort_perf(cfg)
    perf = {k: np.array([base_f[k]] * nf + [base_e[k]], dtype=float) for k in PERF_KEYS}
    jit = sc.performance_jitter
    if jit > 0:
        for k in JITTER_KEYS:
            perf[k][:nf] *= rng.uniform(1.0 - jit, 1.0 + jit, size=nf)
    perf["max_speed"] = np.maximum(perf["max_speed"], perf["min_speed"])
    speed[:nf] = np.clip(speed[:nf], perf["min_speed"][:nf], perf["max_speed"][:nf])

    missiles = np.array([cfg.fighter.num_missiles] * nf + [0], dtype=int)
    return {
        "pos": pos,
        "speed": speed,
        "heading": heading,
        "perf": perf,
        "missiles": missiles,
        "center": center,
        "goal": goal,
        "escort_alt": escort_alt,
    }


def generate_initial_state(cfg: EnvConfig, rng: np.random.Generator) -> InitialState:
    blue = _sample_team(cfg, rng)
    red = blue if cfg.scenario.symmetric else _sample_team(cfg, rng)

    # 赤陣営は陣営座標から z 軸まわり 180 度回転してワールド座標へ
    red_pos = red["pos"].copy()
    red_pos[:, :2] *= -1.0
    pos = np.concatenate([blue["pos"], red_pos])
    speed = np.concatenate([blue["speed"], red["speed"]])
    heading = np.concatenate([blue["heading"], red["heading"] + np.pi])
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    perf = {k: np.concatenate([blue["perf"][k], red["perf"][k]]) for k in PERF_KEYS}
    missiles = np.concatenate([blue["missiles"], red["missiles"]])
    return InitialState(
        pos=pos,
        speed=speed,
        heading=heading,
        gamma=np.zeros(len(speed)),
        perf=perf,
        missiles=missiles,
        escort_center=np.stack([blue["center"], -red["center"]]),
        escort_goal=np.stack([blue["goal"], -red["goal"]]),
        escort_alt=np.array([blue["escort_alt"], red["escort_alt"]]),
    )
