"""コマンドライン: python -m aircombat <command> ...

  match   BLUE RED       2 つのエージェントを対戦させる（リプレイ保存可）
  pretrain               ルールベースの模倣学習で初期モデルを作る
  train                  自己対戦で学習する
  eval    AGENT...       総当たり戦で Glicko-2 レーティングを出す
  replay  JSON -o HTML   保存したリプレイ JSON を HTML ビューアにする
  bench                  環境の速度を測る

エージェントの指定: random / straight / rule / rule_aggressive / rule_defensive /
学習済みモデル（*.pt）/ "パッケージ.モジュール:クラス名"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _env_config(args):
    from .config import EnvConfig

    cfg = EnvConfig.load(args.config) if getattr(args, "config", None) else EnvConfig()
    if getattr(args, "mode", None):
        cfg.scenario.mode = args.mode
    return cfg.validate()


def cmd_match(args) -> int:
    from .agents import make_agent
    from .env import AirCombatEnv
    from .match import run_match, save_replay, summarize

    env = AirCombatEnv(_env_config(args))
    results = []
    for i in range(args.n):
        seed = args.seed + i // 2 if args.swap else args.seed + i
        swap = args.swap and i % 2 == 1
        a, b = (args.red, args.blue) if swap else (args.blue, args.red)
        blue = make_agent(a, deterministic=args.deterministic)
        red = make_agent(b, deterministic=args.deterministic)
        record = bool(args.replay) and i == 0
        r = run_match(env, blue, red, seed=seed, record=record, blue_name=a, red_name=b)
        results.append(r)
        o = r.outcome
        win = {0: f"青({a})の勝ち", 1: f"赤({b})の勝ち", -1: "引き分け"}[r.winner]
        print(
            f"[{i + 1}/{args.n}] seed {seed}  青 {a} {r.scores[0]:.3f} - {r.scores[1]:.3f} 赤 {b}  {win}"
            f"  理由 {o['reason']}  {o['time']:.0f}s  発射 {o['launched']}  命中 {o['hits']}"
        )
        if record:
            path = save_replay(r.replay, args.replay)
            print(f"リプレイを保存しました: {path}")
    if args.n > 1:
        for name in dict.fromkeys([args.blue, args.red]):
            s = summarize(results, name)
            print(f"{name}: {s['games']} 戦 {s['wins']} 勝 {s['draws']} 分 {s['losses']} 敗  平均スコア {s['mean_score']:.3f}")
    return 0


def cmd_train(args) -> int:
    from .selfplay.trainer import SelfPlayTrainer, load_settings

    env_cfg, cfg = load_settings(args.config)
    if args.mode:
        env_cfg.scenario.mode = args.mode
    for key in ("iterations", "num_workers", "episodes_per_iter", "seed"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    if args.init:
        cfg.init_checkpoint = args.init
    run_dir = Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    trainer = SelfPlayTrainer(env_cfg.validate(), cfg, run_dir, resume=args.resume)
    print(f"学習を開始します: {run_dir}（イテレーション {trainer.iteration} → {cfg.iterations}）")
    trainer.train()
    print(f"完了しました。モデル: {run_dir / 'latest.pt'}")
    return 0


def cmd_pretrain(args) -> int:
    from .config import EnvConfig
    from .selfplay.imitation import ImitationConfig, pretrain
    from .selfplay.trainer import load_settings

    env_cfg = load_settings(args.config)[0] if args.config else EnvConfig()
    if args.mode:
        env_cfg.scenario.mode = args.mode
    cfg = ImitationConfig(
        teacher=args.teacher,
        episodes=args.episodes,
        epochs=args.epochs,
        workers=args.workers,
        hidden=args.hidden,
        seed=args.seed,
    )
    if args.opponents:
        cfg.opponents = args.opponents
    pretrain(env_cfg.validate(), cfg, args.out)
    return 0


def cmd_eval(args) -> int:
    from .evaluate import format_leaderboard, round_robin

    res = round_robin(
        args.agents,
        _env_config(args),
        games_per_pair=args.n,
        seed=args.seed,
        workers=args.workers,
        deterministic=args.deterministic,
    )
    print(format_leaderboard(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"結果を保存しました: {args.out}")
    return 0


def cmd_replay(args) -> int:
    from .match import save_replay

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = args.output or str(Path(args.input).with_suffix(".html"))
    print(save_replay(data, out))
    return 0


def cmd_bench(args) -> int:
    from .agents import make_agent
    from .env import AirCombatEnv

    env = AirCombatEnv(_env_config(args))
    agents = [make_agent("rule"), make_agent("rule")]
    steps = 0
    t0 = time.perf_counter()
    ep = 0
    while steps < args.steps:
        obs = env.reset(seed=ep)
        for k in (0, 1):
            agents[k].reset(k, env.cfg, seed=ep)
        done = False
        while not done and steps < args.steps:
            obs, _, done, _ = env.step({k: agents[k].act(obs[k]) for k in (0, 1)})
            steps += 1
        ep += 1
    dt = time.perf_counter() - t0
    print(f"{steps} ステップ / {dt:.2f} 秒 = {steps / dt:.0f} ステップ/秒（rule 同士、エージェントの計算込み）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aircombat", description="空戦AIチャレンジ風の自己対戦環境")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", help="環境設定の JSON（{'env': ...} 形式の学習設定も可）")
        sp.add_argument("--mode", choices=["2d", "3d"], help="2d: ユース部門相当 / 3d: オープン部門相当")

    m = sub.add_parser("match", help="対戦させる")
    m.add_argument("blue")
    m.add_argument("red")
    m.add_argument("-n", type=int, default=1, help="対戦数")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--swap", action="store_true", help="同じシードで陣営を入れ替えて交互に戦う")
    m.add_argument("--replay", help="最初の対戦のリプレイを保存（.html ならビューア付き、.json なら JSON）")
    m.add_argument("--deterministic", action="store_true", help="学習済みモデルを決定的（argmax）に動かす")
    common(m)
    m.set_defaults(func=cmd_match)

    t = sub.add_parser("train", help="自己対戦で学習する")
    t.add_argument("--config", help="学習設定の JSON（configs/*.json）")
    t.add_argument("--mode", choices=["2d", "3d"])
    t.add_argument("--out", help="出力ディレクトリ（既定: runs/日時）")
    t.add_argument("--resume", action="store_true", help="out の checkpoint.pt から再開する")
    t.add_argument("--init", help="この学習済みモデルの重みから始める")
    t.add_argument("--iterations", type=int)
    t.add_argument("--num-workers", dest="num_workers", type=int)
    t.add_argument("--episodes-per-iter", dest="episodes_per_iter", type=int)
    t.add_argument("--seed", type=int)
    t.set_defaults(func=cmd_train)

    pt = sub.add_parser("pretrain", help="ルールベースの模倣学習で初期モデルを作る")
    pt.add_argument("--out", required=True, help="保存先（*.pt）。train --init に渡す")
    pt.add_argument("--teacher", default="rule", help="真似る教師エージェント")
    pt.add_argument("--opponents", nargs="+", help="教師の対戦相手（既定: rule 系と random / straight）")
    pt.add_argument("--episodes", type=int, default=120)
    pt.add_argument("--epochs", type=int, default=6)
    pt.add_argument("--hidden", type=int, default=128, help="ネットワークの幅（train の hidden と揃える）")
    pt.add_argument("--workers", type=int, default=3)
    pt.add_argument("--seed", type=int, default=0)
    common(pt)
    pt.set_defaults(func=cmd_pretrain)

    e = sub.add_parser("eval", help="総当たり戦で評価する")
    e.add_argument("agents", nargs="+")
    e.add_argument("-n", type=int, default=10, help="1 組あたりの対戦数（陣営入れ替えを含む）")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--workers", type=int, default=0)
    e.add_argument("--deterministic", action="store_true")
    e.add_argument("--out", help="結果を JSON で保存")
    common(e)
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("replay", help="リプレイ JSON を HTML にする")
    r.add_argument("input")
    r.add_argument("-o", "--output")
    r.set_defaults(func=cmd_replay)

    b = sub.add_parser("bench", help="速度を測る")
    b.add_argument("--steps", type=int, default=3000)
    common(b)
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.set_printoptions(precision=3, suppress=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
