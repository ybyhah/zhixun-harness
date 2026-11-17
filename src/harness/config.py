"""配置加载与校验。

任务适配、阈值、门控参数全部走配置，避免改代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REQUIRED = ("task", "data", "model", "monitor", "gate", "loop", "record")


@dataclass
class Config:
    """扁平持有各段配置，缺省值在此集中声明。"""

    raw: dict[str, Any] = field(default_factory=dict)

    # ---- 便捷访问 ----
    @property
    def task(self) -> dict[str, Any]:
        return self.raw["task"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def monitor(self) -> dict[str, Any]:
        return self.raw["monitor"]

    @property
    def gate(self) -> dict[str, Any]:
        return self.raw["gate"]

    @property
    def loop(self) -> dict[str, Any]:
        return self.raw["loop"]

    @property
    def record(self) -> dict[str, Any]:
        return self.raw["record"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def get(self, path: str, default: Any = None) -> Any:
        """点号路径取值，例如 get("gate.min_gain")。"""
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"配置缺少必填段: {missing}")
    return Config(raw=raw)
