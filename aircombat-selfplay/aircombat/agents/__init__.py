"""エージェントの生成。

make_agent() に渡す指定子:
- "random" / "straight" / "rule" / "rule_aggressive" / "rule_defensive"
- 学習済みモデルのパス（*.pt）: 学習した方策で行動する（PyTorch が必要）
- "パッケージ.モジュール:クラス名": 自作のエージェント（引数なしで生成できること）
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .base import Agent
from .rule_based import RuleBasedAgent, aggressive, defensive
from .simple import RandomAgent, StraightAgent

BUILTIN = {
    "random": RandomAgent,
    "straight": StraightAgent,
    "rule": RuleBasedAgent,
    "rule_aggressive": aggressive,
    "rule_defensive": defensive,
}


def make_agent(spec: str, deterministic: bool = False) -> Agent:
    if spec in BUILTIN:
        return BUILTIN[spec]()
    if spec.endswith(".pt") or Path(spec).suffix == ".pt":
        from ..selfplay.policy_agent import PolicyAgent

        return PolicyAgent.load(spec, deterministic=deterministic)
    if ":" in spec:
        module, cls = spec.split(":", 1)
        return getattr(importlib.import_module(module), cls)()
    raise ValueError(f"unknown agent spec: {spec!r} (choose from {sorted(BUILTIN)} or a .pt file)")


__all__ = ["Agent", "RandomAgent", "StraightAgent", "RuleBasedAgent", "make_agent", "BUILTIN"]
