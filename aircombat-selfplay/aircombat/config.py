"""環境設定。

すべてのパラメータは dataclass で定義し、JSON から部分的に上書きできる。
角度は人が読みやすいように度 (``*_deg``) で持ち、シミュレータ内部でラジアンに変換する。

公式シミュレータの正確な数値は公開されていないため、ここでの既定値は
第5回 空戦AIチャレンジの公開情報（戦闘機4機＋護衛対象機1機の5対5、
レーダー／IRST、誘導弾、0.0〜1.0 の連続値スコア）に沿って置いた「それらしい」値である。
公式ルールに合わせて調整する場合はこのファイルか JSON 設定を書き換える。
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ArenaConfig:
    """戦闘空域。座標は青陣営基準で x が敵方向、y が左、z が高度（上向き）。"""

    x_limit: float = 100_000.0  # |x| がこれを超えると空域外
    y_limit: float = 75_000.0  # |y| がこれを超えると空域外
    alt_min: float = 0.0  # これを下回ると墜落
    alt_max: float = 20_000.0  # 上昇限度（これ以上は上がれない）


@dataclass
class FighterSpec:
    """戦闘機の基準性能。シナリオ生成時に performance_jitter で個体差が付く。"""

    min_speed: float = 150.0  # [m/s]
    max_speed: float = 400.0  # [m/s]
    max_accel: float = 12.0  # 増速の最大加速度 [m/s^2]
    max_decel: float = 18.0  # 減速の最大加速度 [m/s^2]
    max_g: float = 7.0  # 旋回時の最大荷重倍数
    turn_drag: float = 0.1  # 横加速度 1 m/s^2 当たりの減速 [m/s^2]（旋回でエネルギーを失う）
    max_pitch_deg: float = 40.0  # 経路角の上限
    rcs_scale: float = 1.0  # 被探知性（レーダー探知距離は rcs^(1/4) 倍）
    radar_range: float = 100_000.0  # 基準目標に対するレーダー探知距離 [m]
    radar_fov_deg: float = 60.0  # レーダー視野の半頂角
    irst_range: float = 40_000.0  # IRST（赤外線捜索追尾）探知距離 [m]（方位のみ）
    irst_fov_deg: float = 90.0  # IRST 視野の半頂角
    mws_range: float = 30_000.0  # 誘導弾警報装置 (MWS) の探知距離 [m]（方位のみ）
    num_missiles: int = 4  # 誘導弾の搭載数
    max_launch_range: float = 80_000.0  # 射撃可能な最大距離 [m]
    launch_fov_deg: float = 60.0  # 射撃可能なオフボアサイト角
    launch_interval: float = 2.0  # 同一機の連続発射の最小間隔 [s]


@dataclass
class EscortSpec:
    """護衛対象機。自律飛行し、武装は持たない。"""

    speed: float = 220.0  # [m/s]
    max_g: float = 2.5
    rcs_scale: float = 4.0  # 戦闘機より大きく探知されやすい
    has_radar: bool = False  # True なら戦闘機と同じレーダーで味方の状況認識に寄与
    hits_to_kill: int = 1  # 撃墜に必要な命中数
    behavior: str = "orbit"  # "orbit"（旋回待機） / "advance"（前進後に旋回待機）
    orbit_radius: float = 8_000.0  # [m]
    advance_x: float = -30_000.0  # advance 時に向かう x 座標（自陣営基準）
    evade_range: float = 0.0  # >0 なら、この距離内に敵戦闘機を捕捉すると離脱方向へ逃げる


@dataclass
class MissileSpec:
    """誘導弾。ブースト→慣性飛翔の速度モデルと比例航法 (PN) 誘導。"""

    boost_time: float = 6.0  # [s]
    boost_accel: float = 120.0  # [m/s^2]
    drag_k0: float = 9.0e-5  # 海面高度での抗力係数: dv/dt = -k0 * exp(-h/H) * v^2
    scale_height: float = 9_000.0  # 大気密度のスケールハイト [m]
    induced_drag: float = 0.05  # 横加速度 1 m/s^2 当たりの減速 [m/s^2]
    max_g: float = 30.0  # 最大横加速度 [G]
    agility_ref_speed: float = 700.0  # これより遅いと機動力が速度の二乗で落ちる [m/s]
    nav_gain: float = 4.0  # 比例航法定数
    guidance_dt: float = 0.125  # 終末誘導の計算刻み [s]（粗いと近距離の命中精度が落ちる）
    terminal_range: float = 6_000.0  # 目標までこの距離以内なら guidance_dt で刻む [m]
    seeker_range: float = 20_000.0  # アクティブシーカーの捕捉距離 [m]
    seeker_fov_deg: float = 45.0  # シーカー捕捉視野の半頂角
    gimbal_limit_deg: float = 60.0  # これを超えるとロックが外れる
    lethal_radius: float = 50.0  # 近接信管の作動半径 [m]
    min_speed: float = 350.0  # これを下回ると失速（喪失）[m/s]
    max_flight_time: float = 120.0  # 最大飛翔時間 [s]


@dataclass
class ScenarioConfig:
    num_fighters: int = 4  # 陣営あたりの戦闘機数
    mode: str = "3d"  # "3d"（オープン部門相当）/ "2d"（ユース部門相当、高度固定）
    fixed_altitude: float = 10_000.0  # 2d モードの高度 [m]
    time_limit: float = 1200.0  # 対戦時間の上限 [s]
    sim_dt: float = 0.25  # 物理計算の刻み [s]
    decision_interval: float = 1.0  # 行動判断の間隔 [s]（sim_dt の整数倍）
    track_memory: float = 30.0  # 見失った航跡を保持する時間 [s]
    out_of_bounds: str = "destroy"  # 空域外に出た戦闘機の扱い: "destroy" / "ignore"
    # 行動の安全装置（公式サンプルの高度維持・空域外防止に相当）。0 で無効
    altitude_guard: float = 2_000.0  # この高度を下回ったら降下コマンドを水平以上に上書き [m]
    boundary_guard: float = 5_000.0  # 空域の端からこの距離以内で外向きなら中央へ旋回させる [m]
    # 初期配置（自陣営基準の座標。赤陣営は原点対称に写像される）
    fighter_x_range: tuple[float, float] = (-70_000.0, -55_000.0)
    fighter_y_range: tuple[float, float] = (-25_000.0, 25_000.0)
    fighter_alt_range: tuple[float, float] = (7_000.0, 12_000.0)
    fighter_speed_range: tuple[float, float] = (250.0, 300.0)
    heading_noise_deg: float = 15.0
    escort_x_range: tuple[float, float] = (-90_000.0, -80_000.0)
    escort_y_range: tuple[float, float] = (-20_000.0, 20_000.0)
    escort_alt_range: tuple[float, float] = (8_000.0, 10_000.0)
    performance_jitter: float = 0.1  # 機体性能の個体差（±割合）
    symmetric: bool = False  # True なら赤陣営を青陣営の点対称配置・同性能にする（公平な比較用）


@dataclass
class ScoreConfig:
    """第5回ルールの連続値スコアの推定式。

    勝者: win_base + hit_bonus * 命中率 + speed_bonus * 速度係数
    敗者: 1 - 勝者の得点、引き分け: draw_score
    速度係数は full_speed_time 秒以内の勝利で 1、そこから time_limit で 0 まで線形に下がる。
    """

    win_base: float = 0.6
    hit_bonus: float = 0.3
    speed_bonus: float = 0.1
    full_speed_time: float = 900.0
    draw_score: float = 0.5


@dataclass
class RewardConfig:
    """強化学習用の報酬（スコアとは別物）。ゼロサムになるよう対称に設計している。"""

    terminal_scale: float = 2.0  # 終局時に terminal_scale * (score - 0.5) を与える
    kill_fighter: float = 0.05  # 敵戦闘機の撃墜
    lose_fighter: float = -0.05  # 味方戦闘機の喪失（被撃墜・墜落・空域外）
    hit_escort: float = 0.0  # 護衛対象機への命中（hits_to_kill > 1 のときに有効）
    launch: float = 0.0  # 誘導弾の発射（負にすると撃ち過ぎを抑制）


@dataclass
class EnvConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    fighter: FighterSpec = field(default_factory=FighterSpec)
    escort: EscortSpec = field(default_factory=EscortSpec)
    missile: MissileSpec = field(default_factory=MissileSpec)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def validate(self) -> "EnvConfig":
        sc = self.scenario
        if sc.mode not in ("2d", "3d"):
            raise ValueError(f"scenario.mode must be '2d' or '3d', got {sc.mode!r}")
        ratio = sc.decision_interval / sc.sim_dt
        if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
            raise ValueError("scenario.decision_interval must be a positive multiple of scenario.sim_dt")
        if sc.num_fighters < 1:
            raise ValueError("scenario.num_fighters must be >= 1")
        if self.escort.behavior not in ("orbit", "advance"):
            raise ValueError(f"escort.behavior must be 'orbit' or 'advance', got {self.escort.behavior!r}")
        if sc.out_of_bounds not in ("destroy", "ignore"):
            raise ValueError(f"scenario.out_of_bounds must be 'destroy' or 'ignore', got {sc.out_of_bounds!r}")
        return self

    @property
    def substeps(self) -> int:
        return int(round(self.scenario.decision_interval / self.scenario.sim_dt))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EnvConfig":
        cfg = cls()
        if data:
            merge_into_dataclass(cfg, data)
        return cfg.validate()

    @classmethod
    def load(cls, path: str | Path) -> "EnvConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 学習設定ファイル（{"env": {...}, "train": {...}}）も受け付ける
        if "env" in data and not any(k in data for k in ("arena", "fighter", "scenario")):
            data = data["env"]
        return cls.from_dict(data)

    def copy(self) -> "EnvConfig":
        return copy.deepcopy(self)


def merge_into_dataclass(obj: Any, data: dict[str, Any], path: str = "") -> None:
    """dict の値で dataclass を再帰的に上書きする。未知のキーはエラーにする。"""
    names = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"unknown config key: {path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"config key {path}{key} expects an object")
            merge_into_dataclass(current, value, f"{path}{key}.")
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
