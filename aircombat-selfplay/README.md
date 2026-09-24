# 空戦AIチャレンジ 自己対戦環境（非公式）

防衛装備庁「第5回 空戦AIチャレンジ」の公開情報をもとにした、**軽量な空戦シミュレータと自己対戦（セルフプレイ）学習の一式**です。
戦闘機4機＋護衛対象機1機どうしの 5 対 5 の対戦を Python（numpy）だけで動かし、PPO とリーグ方式の自己対戦で方策を学習し、公式の定量評価と同じ Glicko-2 レーティングで強さを比べられます。

> [!IMPORTANT]
> これは公式シミュレータではありません。公式シミュレータは参加者にだけ配布されるため、機体・誘導弾の性能、センサーの仕様、スコア式の細部などは公開情報から推定して独自に作っています。
> 数値はすべて設定ファイルで変えられるので、公式ルールが分かっている部分はそれに合わせて調整してください。
> 対戦相手プール・PFSP・Glicko-2 の部分（`aircombat/selfplay/league.py`, `glicko2.py`）はシミュレータに依存しないので、公式シミュレータでの学習にも流用できます。

## できること

| コマンド | 内容 |
|---|---|
| `aircombat match` | 2 つのエージェントを対戦させ、結果とリプレイ（HTML）を出す |
| `aircombat pretrain` | ルールベースの行動を真似る模倣学習で、自己対戦の初期モデルを作る |
| `aircombat train` | 自己対戦で学習する（マルチプロセスで対戦を収集、途中から再開可） |
| `aircombat eval` | 総当たり戦で Glicko-2 レーティング・平均スコア・対戦表を出す |
| `aircombat replay` | 保存したリプレイ JSON を HTML ビューアにする |
| `aircombat bench` | 環境の速度を測る |

リプレイは 1 ファイルで開ける HTML です（上面図・高度図・イベントログ・再生コントロール、ダークモード・スマホ幅対応）。

## セットアップ

Python 3.10 以上。

```bash
cd aircombat-selfplay
python -m venv .venv && source .venv/bin/activate

# 対戦・評価だけなら numpy のみ
pip install -e .

# 学習もする場合（PyTorch。GPU が無ければ CPU 版で十分動きます）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,gym,dev]"
```

## クイックスタート

```bash
# ルールベース同士の対戦を 4 戦（同じシードで陣営を入れ替えて交互に）、最初の対戦をリプレイに保存
aircombat match rule rule_aggressive -n 4 --swap --replay replays/demo.html

# 組み込みエージェントの総当たり戦（1 組 10 戦、4 プロセス並列）
aircombat eval rule rule_aggressive rule_defensive random straight -n 10 --workers 4

# おすすめ: ルールベースの模倣学習で初期化してから自己対戦で強化する
aircombat pretrain --out runs/bc.pt --episodes 120 --workers 4
aircombat train --config configs/finetune.json --init runs/bc.pt --out runs/exp1

# ゼロから自己対戦で学習（時間はかかる）。Ctrl+C で止めても --resume で続きから再開できる
aircombat train --config configs/default.json --out runs/exp0
aircombat train --config configs/default.json --out runs/exp0 --resume

# 学習したモデルをルールベースと対戦させる・レーティングを測る
aircombat match runs/exp1/latest.pt rule -n 10 --swap --replay replays/learned.html
aircombat eval runs/exp1/latest.pt runs/exp1/snapshots/iter_000100.pt rule rule_defensive -n 20 --workers 4

# ユース部門相当（2 次元・高度固定）
aircombat match rule rule_defensive --mode 2d
aircombat train --config configs/youth_2d.json --out runs/youth1
```

`aircombat` コマンドの代わりに `python -m aircombat` でも動きます。

組み込みエージェントのおおよその強さ（3D、1 組 6 戦の総当たり）:

| エージェント | 内容 | レーティング | 平均スコア |
|---|---|---|---|
| `rule` | 1 機が護衛、3 機が攻撃。敵戦闘機は回避不能距離の 1.2 倍、護衛対象機は最大射程の 0.9 倍で射撃 | 1760 | 0.84 |
| `rule_defensive` | 2 機が護衛。引き付けてから撃つ | 1678 | 0.73 |
| `rule_aggressive` | 全機突撃。最大射程付近から 1 目標 2 発まで撃ち込む | 1488 | 0.53 |
| `straight` | 何もしない（直進） | 1240 | 0.20 |
| `random` | ランダム | 1227 | 0.19 |

## 対戦ルール（既定値）

公開情報にある第5回の設定を、次のようにモデル化しています。

- **編成**: 各陣営 戦闘機 4 機 ＋ 護衛対象機 1 機。護衛対象機は武装を持たず自律飛行（既定は自陣後方で旋回待機）
- **勝敗**: 相手の護衛対象機を先に撃墜した陣営の勝ち。制限時間（1200 秒）で決着しなければ引き分け。双方とも攻撃手段（残弾・飛翔中の誘導弾）が無くなった場合も引き分け
- **スコア（0.0〜1.0 の連続値）**:
  - 勝者: `0.6 + 0.3 × 命中率 + 0.1 × 速度係数`（速度係数は 900 秒以内の勝利で 1、そこから制限時間で 0 まで線形）
  - 敗者: `1 − 勝者の得点`、引き分け: 0.5
  - 命中率ボーナス（最大 0.3）・速度ボーナス（最大 0.1、900 秒以内で満点）・敗者は 1 − 勝者の得点、は公開情報どおり。**勝ちの基本点 0.6 は満点が 1.0 になるように置いた推定値**です（`score` 設定で変更可）
- **センサー**:
  - レーダー: 探知距離 100 km（護衛対象機は被探知性が高く約 141 km）、視野 ±60°。陣営内でデータリンク共有され、見えている敵だけが観測に入る（部分観測）
  - IRST: 40 km・±90°。方位だけ分かる（距離は分からない）
  - 誘導弾警報 (MWS): 自機を狙う誘導弾を 30 km 以内で探知。方位のみ
  - 見失った敵は 30 秒間、最後の速度で推測した位置で残る
- **誘導弾**: 各機 4 発。ブースト後は空気抵抗で減速し（高高度ほど遠くまで届く）、比例航法で誘導。発射母機側のレーダー航跡で中間誘導し、20 km 以内でアクティブシーカーが捕捉すると終末誘導。横機動でも減速するため、遠距離から撃たれた誘導弾は背を向けて逃げれば避けられる
  - 高度 10 km・280 m/s 同士の目安: 正面から約 55 km、真後ろから約 18 km まで届く
- **機体**: 質点モデル（最大 7G 旋回、旋回・上昇で速度を失う）。機体性能は ±10% の個体差あり
- **空域**: x ±100 km × y ±75 km、高度 0〜20 km。空域外に出た戦闘機・地面に落ちた戦闘機は失われる
- **安全装置**: 高度 2,000 m 未満での降下と、空域端 5 km 以内での外向き飛行はコマンドを上書きして防ぐ（公式サンプルの高度維持・空域外防止に相当。`altitude_guard` / `boundary_guard` を 0 で無効）
- **モード**: `3d`（オープン部門相当）/ `2d`（ユース部門相当。高度 10 km 固定の平面）

すべての値は `aircombat/config.py` の dataclass に説明付きで定義してあり、JSON で上書きできます。

```json
{
  "env": {
    "scenario": {"mode": "3d", "time_limit": 1200},
    "fighter": {"num_missiles": 6},
    "escort": {"behavior": "advance", "evade_range": 30000},
    "score": {"win_base": 0.6}
  }
}
```

## 環境の使い方（Python）

```python
from aircombat import AirCombatEnv
from aircombat.agents import make_agent

env = AirCombatEnv()                      # EnvConfig か dict で設定を渡せる
obs = env.reset(seed=0)                   # {0: 青の TeamObs, 1: 赤の TeamObs}
blue, red = make_agent("rule"), make_agent("random")
blue.reset(0, env.cfg); red.reset(1, env.cfg)
done = False
while not done:
    actions = {0: blue.act(obs[0]), 1: red.act(obs[1])}   # 各 (4, 4) の整数配列
    obs, rewards, done, info = env.step(actions)
print(info["outcome"].to_dict())          # 勝者・理由・スコア・発射数・命中数
```

### 観測（`TeamObs`）

各陣営には **自陣営が知り得る情報だけ** を、**陣営座標**（どちらの陣営でも「敵が +x 方向」）で渡します。z 軸まわりの回転なので左右が入れ替わらず、同じ方策をそのまま青にも赤にも使えます（自己対戦の前提）。

| 属性 | 形 | 内容 |
|---|---|---|
| `self_feats` | (4, 24) | 自機の位置・速度・姿勢・残弾・射撃可否・空域端までの距離・経過時間・性能（個体差）など |
| `ally_feats` | (4, 4, 12) | 他の味方戦闘機 3 機＋自陣営の護衛対象機の相対位置・速度・残弾・被警報数 |
| `enemy_feats` | (4, 5, 26) | 敵スロット（敵戦闘機を推定距離順、最後が敵護衛対象機）。航跡の有無・方位のみか・経過時間・相対位置/速度・接近率・アスペクト角・射撃可否・飛翔中の味方誘導弾数・射程推定との比 |
| `mws_feats` | (4, 4, 4) | 自機を狙う誘導弾の方位 |
| `action_mask` | (4, 22) | 各行動が選べるか（撃墜された機体・射撃できない目標は無効） |
| `view` | — | ルールベース向けの生データ（陣営座標・SI 単位）。射程推定 `range_max` / `range_ne`（回避不能距離）も入っている |

### 行動

陣営ごとに `(4, 4)` の整数配列。各行が 1 機分で `[旋回, 経路角, スロットル, 射撃]`:

| ヘッド | 選択肢 |
|---|---|
| 旋回 | 最大旋回率に対する割合 `[-1, -0.5, -0.2, 0, 0.2, 0.5, 1]`（正が左旋回） |
| 経路角 | 目標経路角 `[-30, -10, 0, 10, 30]` 度（2d では 0 のみ） |
| スロットル | `[減速, 速度維持, 増速]` |
| 射撃 | 0 = 撃たない、s = 敵スロット s−1 を撃つ（`TeamObs.fire_action()` で敵の番号から変換できる） |

### 報酬（学習用）

終局時に `2 × (スコア − 0.5)`、途中で敵戦闘機の撃墜 +0.05・味方戦闘機の喪失 −0.05（`reward` 設定で変更可）。スコアそのものではなく学習用の値です。

### 他の強化学習ライブラリから使う（Gymnasium 互換）

`SingleTeamEnv` は片方の陣営だけを外から操作し、相手は固定のエージェントが動かす Gymnasium 互換の環境です（`pip install -e ".[gym]"`）。Stable-Baselines3 などにそのまま渡せ、`action_masks()` は sb3-contrib の MaskablePPO が使う形の行動マスクを返します。

```python
from aircombat.wrappers import SingleTeamEnv

env = SingleTeamEnv(opponent="rule", team="random")   # 相手はエージェント名・モデルのパス・Agent インスタンス
obs, info = env.reset(seed=0)                          # obs は "self" / "ally" / "enemy" / "mws" / "action_mask" の辞書
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## 自己対戦学習の仕組み

### 模倣学習による初期化（`aircombat pretrain`）

ゼロから始めた方策は、最初のうち遠距離から誘導弾を撃ち尽くしてしまい、勝ち方を見つけるまでに長い時間がかかります。
`pretrain` は教師（既定は `rule`）をいろいろな相手と対戦させて教師側の観測と行動を記録し、同じネットワークに行動を真似させます（各行動ヘッドの交差エントロピー）。価値関数も割引収益で同時に学習するので、そのまま `train --init` に渡すと、ルールベース並みの強さから自己対戦を始められます（`selfplay/imitation.py`）。

### 自己対戦の強化学習（`aircombat train`）

`aircombat train` は次を繰り返します（`aircombat/selfplay/`）。

1. **マッチメイク**（`league.py`）: 1 イテレーションあたり `episodes_per_iter` 戦分の相手を抽選
   - 確率 `self_play_prob`（既定 0.35）で最新の自分自身（両陣営のデータを学習に使う）
   - それ以外はリーグ（固定のルールベース＋過去のスナップショット）から **PFSP** で抽選。重みは `(1 − 学習者の対戦スコア)^α` で、勝てていない相手ほど選ばれやすい
2. **収集**（`rollout.py`）: ワーカープロセスが複数の環境を同時に進め、学習者の推論を環境をまたいでまとめて行う
3. **更新**（`ppo.py`）: 陣営単位の報酬・価値から GAE で求めた利得を 4 機で共有し、方策比は機体ごとに取る PPO（MAPPO 方式）。撃墜された機体のステップは方策の損失から除く。模倣学習の重みから始めるときは、初期方策からの KL ダイバージェンスを損失に加えて（`ppo.anchor_coef`、係数は線形に減衰）、ノイズの多い更新で方策が崩れるのを防ぐ（AlphaStar の教師あり方策への KL と同じ考え方）
4. **リーグ更新**: 対戦結果で学習者の対戦スコアの移動平均と **Glicko-2 レーティング** を更新。`snapshot_interval` ごとに現在の方策をスナップショットとしてリーグに追加（古いものから `max_snapshots` 個まで保持）
5. **評価**: `eval_interval` ごとに固定の相手（既定 `rule`, `rule_defensive`）と同じシードで対戦して平均スコアを記録し、最良のモデルを `best.pt` に保存。`eval_deterministic: true` なら学習者は最も確率の高い行動で戦う。`replay_interval` ごとにリプレイ HTML を保存

ネットワーク（`model.py`、約 35 万パラメータ）は自機・味方・敵・誘導弾警報をそれぞれ埋め込み、自機を query にした注意機構で集約します。陣営 4 機の埋め込みの平均を各機に足して連携させ（1 つのエージェントが 4 機すべてを操縦する前提）、射撃ヘッドは敵スロットを指すポインタ型です。

出力（`--out` のディレクトリ）:

| ファイル | 内容 |
|---|---|
| `latest.pt` / `best.pt` | そのままエージェントとして使えるモデル（`aircombat match runs/exp1/latest.pt rule`） |
| `snapshots/iter_XXXXXX.pt` | リーグに入れたスナップショット |
| `checkpoint.pt` | 再開用（モデル・最適化器・リーグ・乱数） |
| `metrics.jsonl` | イテレーションごとの相手別スコア・PPO の統計・リーグ表・評価結果 |
| `replays/iter_XXXXXX.html` | 評価対戦のリプレイ |

設定ファイル:

| ファイル | 用途 |
|---|---|
| `configs/default.json` | オープン部門相当（3D）の標準設定（ゼロから学習） |
| `configs/finetune.json` | 模倣学習の重みから始める設定（小さい学習率、初期方策への KL 正則化、射撃ヘッドのエントロピーボーナスなし、決定的評価） |
| `configs/youth_2d.json` | ユース部門相当（2D） |
| `configs/quick.json` | 動作確認用の小さい設定（数分で終わる） |

速度の目安（4 コア CPU）: 環境単体で約 450 ステップ/秒（ルールベース同士、1 ステップ = 1 秒）。`configs/default.json` の学習は 1 イテレーション（32 戦、約 4 万ステップ）あたり約 1 分です。

### 学習の実例（4 コア CPU での短時間の試行）

| 段階 | かかった時間 | 結果 |
|---|---|---|
| ゼロから（`default.json`） | 6 イテレーション（約 6 分） | 遠距離から 16 発を撃ち尽くして命中 0。ルールベースには全敗（この段階を抜けるには長時間の学習が必要） |
| 模倣学習（`pretrain`、教師 `rule`・160 戦） | 約 5 分 | 教師との一致率: 旋回 0.72・経路角 0.98・スロットル 0.86・撃った場面の射撃 0.80。決定的に動かすと総当たり戦で教師とほぼ同じ強さ（レーティング 1605、`rule` は 1590） |
| 自己対戦で強化（`finetune.json`、KL 正則化あり） | 29 イテレーション（約 45 分） | 崩れずに維持。評価（決定的・各 24 戦）は `rule` 相手 0.46 → 0.46〜0.53、`rule_defensive` 相手 0.69 → 0.56〜0.66 で、誤差（24 戦で ±0.06 程度）の範囲内の横ばい |
| 参考: KL 正則化なし | 25 イテレーション | 初期方策から離れて弱くなった（`rule` 相手 0.40 → 0.21、命中率 0.08 → 0.02） |

ルールベースをはっきり超えるには、数百イテレーション以上（数時間〜）の学習が必要な見込みです。評価の試合数が少ないとばらつきが大きいので、比較には `aircombat eval -n 24` 以上を使ってください。

## 自作のエージェントを対戦させる

`Agent` を継承して `act()` を実装し、`パッケージ.モジュール:クラス名` で指定します。

```python
# my_agents.py
import math

import numpy as np
from aircombat.agents.base import Agent
from aircombat.obs import NEUTRAL_ACTION, TURN_OPTIONS


class Kamikaze(Agent):
    """敵の護衛対象機へまっすぐ向かい、射程推定の 9 割まで近づいたら 1 発ずつ撃つ。"""

    name = "kamikaze"

    def act(self, obs):
        v = obs.view  # 陣営座標の生データ（敵は自陣営が探知しているものだけ）
        esc = int(np.flatnonzero(v.enemy_is_escort)[0])
        out = np.tile(np.array(NEUTRAL_ACTION), (len(v.alive), 1))
        for e in range(len(v.alive)):
            goal = v.enemy_pos[esc] if v.enemy_known[esc] else np.array([80_000.0, 0.0, 0.0])
            err = math.atan2(goal[1] - v.pos[e, 1], goal[0] - v.pos[e, 0]) - v.heading[e]
            err = (err + math.pi) % (2 * math.pi) - math.pi
            out[e, 0] = int(np.argmin(np.abs(TURN_OPTIONS - np.clip(err, -1, 1))))
            dist = np.linalg.norm(v.enemy_pos[esc] - v.pos[e])
            # 撃ち過ぎると命中率ボーナスが減るので、飛翔中の味方誘導弾が無いときだけ撃つ
            if v.launchable[e, esc] and dist < 0.9 * v.range_max[e, esc] and v.missiles_on[esc] == 0:
                out[e, 3] = obs.fire_action(e, esc)
        return out
```

```bash
aircombat match my_agents:Kamikaze rule -n 4 --swap
aircombat eval my_agents:Kamikaze rule rule_defensive -n 10
```

リーグの固定の相手にも加えられます（`train.league.initial_opponents` に `"my_agents:Kamikaze"` を追加）。

## 公式シミュレータで使うには

このリポジトリのシミュレータは練習・研究用です。公式シミュレータで自己対戦学習をする場合は、次の部分をそのまま流用できます。

- `selfplay/league.py`: 相手の抽選（自己対戦／PFSP）、スナップショットの管理、Glicko-2 による追跡
- `selfplay/glicko2.py`: 公式の定量評価と同じ Glicko-2（Glickman の計算例と一致することをテストで確認済み）
- `evaluate.py` の集計部分: 総当たり戦の対戦表とレーティング

`rollout.py` の `RolloutWorker.run()` が環境とやり取りする部分（`reset` / `step` と観測の変換）を公式シミュレータの API に置き換えれば、`trainer.py` の学習ループはそのまま使えます。

## 開発

```bash
pytest            # 45 件、約 20 秒（gymnasium が無ければラッパーのテストは飛ばす）
```

```
aircombat/
├── config.py            # 全パラメータ（dataclass、JSON で上書き可）
├── geometry.py          # 座標変換
├── sim/                 # シミュレータ本体
│   ├── core.py          #   機体・誘導弾・センサー・データリンク
│   ├── scenario.py      #   初期配置・機体の個体差
│   └── missile_range.py #   射程推定テーブル
├── rules.py             # 勝敗判定・スコア
├── obs.py               # 観測・行動の定義
├── env.py               # 二陣営の対戦環境
├── match.py             # 対戦の実行・リプレイ記録
├── wrappers.py          # Gymnasium 互換ラッパー
├── evaluate.py          # 総当たり戦・Glicko-2
├── agents/              # ランダム・直進・ルールベース
├── selfplay/            # 自己対戦学習（モデル・PPO・ロールアウト・リーグ・Glicko-2・模倣学習・学習ループ）
├── viewer/              # リプレイ HTML
└── cli.py               # コマンドライン
```

## 参考

- [第5回 空戦AIチャレンジに係るお知らせ（防衛装備庁）](https://www.mod.go.jp/atla/kousouken/aichall_the5th.html)
- [SIGNATE「第5回 空戦AIチャレンジ」開催のプレスリリース](https://prtimes.jp/main/html/rd/p/000000327.000038674.html)
- Mark E. Glickman, *Example of the Glicko-2 system*
- AlphaStar の PFSP（Prioritized Fictitious Self-Play）
