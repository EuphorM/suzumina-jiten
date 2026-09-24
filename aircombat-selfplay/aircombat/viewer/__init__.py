"""リプレイ JSON から単体で開ける HTML ビューアを作る。"""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = Path(__file__).with_name("template.html")
_PLACEHOLDER = "/*__REPLAY_DATA__*/null"


def render_html(replay: dict) -> str:
    template = _TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(_PLACEHOLDER, data, 1)
