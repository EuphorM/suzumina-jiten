"""学習済みネットワークで行動するエージェント。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..agents.base import Agent
from ..obs import TeamObs
from .model import PolicyNet, load_checkpoint, obs_to_arrays, stack_obs, to_tensors


class PolicyAgent(Agent):
    def __init__(self, model: PolicyNet, deterministic: bool = False, name: str = "policy"):
        self.model = model
        self.deterministic = deterministic
        self.name = name

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = False) -> "PolicyAgent":
        model, _ = load_checkpoint(path)
        return cls(model, deterministic=deterministic, name=Path(path).stem)

    def act(self, obs: TeamObs) -> np.ndarray:
        t = to_tensors(stack_obs([obs_to_arrays(obs)]))
        actions, _, _ = self.model.act(t, deterministic=self.deterministic)
        return actions[0].numpy()
