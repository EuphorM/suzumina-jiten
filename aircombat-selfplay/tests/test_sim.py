import numpy as np
import pytest

from aircombat.geometry import G
from aircombat.sim.core import CAUSE_CRASH, CAUSE_OUT, Simulation
from conftest import isolated_state, make_cfg


def run(sim: Simulation, seconds: float, sensors_every: int = 4):
    steps = int(round(seconds / sim.dt))
    for k in range(steps):
        sim.step()
        if (k + 1) % sensors_every == 0:
            sim.update_sensors()


def test_straight_flight_keeps_speed_and_heading(cfg):
    sim = Simulation(cfg)
    sim.reset(isolated_state(cfg))
    p0 = sim.pos[0].copy()
    run(sim, 10.0)
    assert sim.speed[0] == pytest.approx(250.0)
    assert sim.heading[0] == pytest.approx(-np.pi) or sim.heading[0] == pytest.approx(np.pi)
    assert np.linalg.norm(sim.pos[0] - p0) == pytest.approx(2_500.0, rel=1e-6)


def test_turn_rate_is_limited_by_g(cfg):
    sim = Simulation(cfg)
    sim.reset(isolated_state(cfg))
    h0 = sim.heading[0]
    sim.cmd_turn[0] = 1.0
    sim.cmd_throttle[0] = 1  # 増速しながら旋回
    sim.step()
    v = 250.0
    omega = G * np.sqrt(cfg.fighter.max_g**2 - 1) / v
    turned = (sim.heading[0] - h0 + np.pi) % (2 * np.pi) - np.pi
    assert turned == pytest.approx(omega * sim.dt, rel=1e-6)


def test_hold_throttle_compensates_turn_drag(cfg):
    sim = Simulation(cfg)
    sim.reset(isolated_state(cfg))
    sim.cmd_turn[0] = 1.0
    sim.cmd_throttle[0] = 0
    sim.cmd_turn[1] = 1.0
    sim.cmd_throttle[1] = -1
    run(sim, 5.0)
    assert sim.speed[0] == pytest.approx(250.0, abs=1e-6)
    assert sim.speed[1] < 200.0


def test_ground_crash_and_ceiling():
    cfg = make_cfg()
    sim = Simulation(cfg)
    init = isolated_state(cfg, alt=600.0)
    sim.reset(init)
    sim.cmd_gamma[0] = -np.radians(30)
    sim.cmd_gamma[1] = np.radians(30)
    sim.pos[1, 2] = cfg.arena.alt_max - 50.0
    run(sim, 20.0)
    assert not sim.alive[0]
    assert sim.death_cause[0] == CAUSE_CRASH
    assert sim.alive[1] and sim.pos[1, 2] <= cfg.arena.alt_max


def test_out_of_bounds_destroys_fighter_only():
    cfg = make_cfg(out_of_bounds="destroy")
    sim = Simulation(cfg)
    sim.reset(isolated_state(cfg))
    run(sim, 30.0)  # 外向きに 250 m/s で 30 秒 → x = -102.5 km
    nf = cfg.scenario.num_fighters
    assert not sim.alive[0] and sim.death_cause[0] == CAUSE_OUT
    assert sim.alive[nf]  # 護衛対象機は空域外判定の対象外（自律飛行で戻る）


def _duel(dist: float, aspect: str, evade: bool = False, mode: str = "3d", alt: float = 10_000.0):
    cfg = make_cfg(mode=mode)
    cfg.fighter.max_launch_range = 200_000.0
    sim = Simulation(cfg)
    init = isolated_state(cfg, alt=alt)
    nf = cfg.scenario.num_fighters
    shooter, target = 0, nf + 1
    init.pos[shooter] = [0.0, 0.0, alt]
    init.heading[shooter] = 0.0
    init.speed[shooter] = 280.0
    init.pos[target] = [dist, 0.0, alt]
    init.heading[target] = np.pi if aspect == "head" else 0.0
    init.speed[target] = 280.0
    sim.reset(init)
    mid = sim.launch(shooter, target)
    assert mid >= 0
    locked_seen = False
    k = 0
    while sim.m_active[mid]:
        if evade:
            sim.cmd_turn[target] = 1.0
            sim.cmd_throttle[target] = 1
        sim.step()
        k += 1
        locked_seen |= bool(sim.m_locked[mid])
        if k % 4 == 0:
            sim.update_sensors()
    kinds = [e.kind for e in sim.events if e.missile == mid]
    return sim, mid, kinds, locked_seen, target


def test_missile_hits_head_on_target_and_kills():
    sim, mid, kinds, locked, target = _duel(30_000.0, "head")
    assert "hit" in kinds and locked
    assert not sim.alive[target]
    assert sim.hits[0] == 1 and sim.launched[0] == 1


def test_missile_runs_out_of_energy_in_long_tail_chase():
    sim, mid, kinds, _, target = _duel(40_000.0, "tail")
    assert kinds[-1] == "miss"
    assert sim.alive[target]


def test_evasion_defeats_long_shot_but_not_short_shot():
    _, _, far, _, _ = _duel(40_000.0, "head", evade=True)
    _, _, near, _, _ = _duel(15_000.0, "head", evade=True)
    assert far[-1] == "miss"
    assert near[-1] == "hit"


def test_missile_stays_in_plane_in_2d():
    sim, mid, kinds, _, _ = _duel(30_000.0, "head", mode="2d")
    assert "hit" in kinds
    assert sim.m_pos[mid, 2] == pytest.approx(10_000.0)


def test_launch_requires_track_range_and_missiles(cfg):
    sim = Simulation(cfg)
    init = isolated_state(cfg)
    nf = cfg.scenario.num_fighters
    init.pos[0] = [0.0, 0.0, 10_000.0]
    init.heading[0] = 0.0
    init.pos[nf + 1] = [50_000.0, 0.0, 10_000.0]
    init.pos[nf + 2] = [0.0, 50_000.0, 10_000.0]  # 真横（視野外）
    sim.reset(init)
    ok = sim.launchable(np.array([0]))[0]
    assert ok[nf + 1] and not ok[nf + 2]
    assert not ok[1]  # 味方は撃てない
    assert sim.launch(0, nf + 1) >= 0
    assert sim.launch(0, nf + 1) == -1  # 連続発射の間隔
    sim.time += cfg.fighter.launch_interval
    sim.missiles_left[0] = 0
    assert sim.launch(0, nf + 1) == -1


def test_missile_to_destroyed_target_is_a_miss(cfg):
    sim = Simulation(cfg)
    init = isolated_state(cfg)
    nf = cfg.scenario.num_fighters
    init.pos[0] = [0.0, 0.0, 10_000.0]
    init.heading[0] = 0.0
    init.pos[nf + 1] = [50_000.0, 0.0, 10_000.0]
    init.heading[nf + 1] = np.pi
    sim.reset(init)
    mid = sim.launch(0, nf + 1)
    sim._destroy(nf + 1, CAUSE_CRASH)
    sim.step()
    ev = [e for e in sim.events if e.missile == mid and e.kind == "miss"]
    assert ev and ev[0].detail == "target_destroyed"


def test_radar_and_irst_detection(cfg):
    sim = Simulation(cfg)
    init = isolated_state(cfg)
    nf = cfg.scenario.num_fighters
    esc_red = 2 * nf + 1
    init.pos[0] = [0.0, 0.0, 10_000.0]
    init.heading[0] = 0.0
    init.pos[nf + 1] = [90_000.0, 0.0, 10_000.0]  # レーダー探知距離内
    init.pos[nf + 2] = [30_000.0, 30_000.0, 10_000.0]  # 45°: レーダー視野内
    init.pos[nf + 3] = [5_000.0, 20_000.0, 10_000.0]  # 76°: レーダー視野外だが IRST 視野内
    init.pos[esc_red] = [0.0, -115_000.0, 10_000.0]  # 真横
    sim.reset(init)
    assert sim.radar_det[0, nf + 1] and sim.radar_det[0, nf + 2]
    assert not sim.radar_det[0, nf + 3] and sim.irst_det[0, nf + 3]
    assert not sim.irst_det[0, nf + 1]  # IRST の探知距離外
    assert not sim.radar_det[0, esc_red]
    # 護衛対象機は RCS が大きいので戦闘機より遠くから見える
    sim.pos[esc_red] = [115_000.0, 0.0, 10_000.0]
    sim.pos[nf + 1] = [115_000.0, 5_000.0, 10_000.0]
    sim.update_sensors()
    assert sim.radar_det[0, esc_red] and not sim.radar_det[0, nf + 1]
    assert sim.team_track[0, esc_red]
