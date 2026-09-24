"""総当たり戦による評価（Glicko-2 レーティング）。

公式の定量評価と同じく、各組み合わせの対戦結果から Glicko-2 レーティングを計算する。
各組み合わせは同じ乱数シードで陣営を入れ替えて 2 戦ずつ行うので、初期配置の有利不利が相殺される。
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from collections import defaultdict

import numpy as np

from .agents import make_agent
from .config import EnvConfig
from .env import AirCombatEnv
from .match import run_match
from .selfplay.glicko2 import RatingTable

_ENV: AirCombatEnv | None = None


def _init(env_cfg: dict) -> None:
    global _ENV
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    _ENV = AirCombatEnv(EnvConfig.from_dict(env_cfg))


def _play(args) -> dict:
    a, b, seed, deterministic = args
    blue = make_agent(a, deterministic=deterministic)
    red = make_agent(b, deterministic=deterministic)
    r = run_match(_ENV, blue, red, seed=seed, blue_name=a, red_name=b)
    return {"blue": a, "red": b, "seed": seed, "scores": list(r.scores), "winner": r.winner, "outcome": r.outcome}


def round_robin(
    agents: list[str],
    env_cfg: EnvConfig,
    games_per_pair: int = 10,
    seed: int = 0,
    workers: int = 0,
    deterministic: bool = False,
    rounds: int = 5,
) -> dict:
    """games_per_pair は陣営入れ替えを含む総数（偶数に切り上げ）。"""
    if len(set(agents)) != len(agents):
        raise ValueError("agent specs must be unique")
    pairs = list(itertools.combinations(agents, 2))
    n_seeds = (games_per_pair + 1) // 2
    tasks = []
    for a, b in pairs:
        for i in range(n_seeds):
            s = seed * 100_003 + i
            tasks.append((a, b, s, deterministic))
            tasks.append((b, a, s, deterministic))
    if workers > 0:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(env_cfg.to_dict(),)) as pool:
            games = pool.map(_play, tasks, chunksize=1)
    else:
        _init(env_cfg.to_dict())
        games = [_play(t) for t in tasks]

    # 対戦をいくつかのレーティング期間に分けて Glicko-2 を更新する
    table = RatingTable()
    for name in agents:
        table.get(name)
    order = np.random.default_rng(seed).permutation(len(games))
    for chunk in np.array_split(order, max(1, rounds)):
        table.update_period([(games[i]["blue"], games[i]["red"], games[i]["scores"][0]) for i in chunk])

    per = defaultdict(lambda: {"games": 0, "score": 0.0, "wins": 0, "draws": 0, "losses": 0, "hits": 0, "launched": 0})
    matrix = defaultdict(lambda: defaultdict(list))
    for g in games:
        for team, name in ((0, g["blue"]), (1, g["red"])):
            p = per[name]
            p["games"] += 1
            p["score"] += g["scores"][team]
            p["hits"] += g["outcome"]["hits"][team]
            p["launched"] += g["outcome"]["launched"][team]
            if g["winner"] == -1:
                p["draws"] += 1
            elif g["winner"] == team:
                p["wins"] += 1
            else:
                p["losses"] += 1
        matrix[g["blue"]][g["red"]].append(g["scores"][0])
        matrix[g["red"]][g["blue"]].append(g["scores"][1])

    rows = []
    for name, r in table.leaderboard():
        p = per[name]
        rows.append(
            {
                "agent": name,
                "rating": round(r.rating, 1),
                "rd": round(r.rd, 1),
                "mean_score": round(p["score"] / max(p["games"], 1), 4),
                "wins": p["wins"],
                "draws": p["draws"],
                "losses": p["losses"],
                "hit_rate": round(p["hits"] / p["launched"], 3) if p["launched"] else 0.0,
            }
        )
    mat = {a: {b: round(float(np.mean(v)), 3) for b, v in row.items()} for a, row in matrix.items()}
    return {"leaderboard": rows, "matrix": mat, "games": games}


def format_leaderboard(result: dict) -> str:
    rows = result["leaderboard"]
    width = max(len(r["agent"]) for r in rows)
    lines = [f"{'agent':<{width}}  rating    rd   score   W/D/L     hit"]
    for r in rows:
        wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
        lines.append(
            f"{r['agent']:<{width}}  {r['rating']:6.1f} {r['rd']:5.1f}  {r['mean_score']:.3f}  {wdl:>8}  {r['hit_rate']:.2f}"
        )
    names = [r["agent"] for r in rows]
    lines.append("")
    lines.append("対戦スコア（行の視点）")
    short = [n if len(n) <= 14 else n[:13] + "…" for n in names]
    lines.append(" " * (width + 2) + " ".join(f"{s:>14}" for s in short))
    for a in names:
        cells = []
        for b in names:
            v = result["matrix"].get(a, {}).get(b)
            cells.append(f"{'-' if v is None else f'{v:.3f}':>14}")
        lines.append(f"{a:<{width}}  " + " ".join(cells))
    return "\n".join(lines)
