"""空戦AIチャレンジ（第5回ルール相当）の自己対戦環境。"""

from .config import EnvConfig
from .env import AirCombatEnv

__all__ = ["AirCombatEnv", "EnvConfig"]
__version__ = "0.1.0"
