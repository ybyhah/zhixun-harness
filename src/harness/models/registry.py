"""模型版本库：online / candidate / history 三态管理。

门控不通过时丢弃候选；连续失败时回滚到历史最优版本。
所有版本留档，对应赛题要求的"可复现、可审计"。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .industrial import IndustrialModel


@dataclass
class Version:
    version: int
    model: IndustrialModel
    batch_index: int           # 在哪个批次产生
    action: str                # 由什么动作产生
    score: float = float("nan")  # 上线时的门控参考分

    def describe(self) -> dict:
        return {
            "version": self.version,
            "batch_index": self.batch_index,
            "action": self.action,
            "score": self.score,
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._online: Version | None = None
        self._candidate: Version | None = None
        self._history: list[Version] = []
        self._next_version = 1
        self.consecutive_failures = 0
        self.rollbacks: list[dict] = []

    # ---- 查询 ----
    @property
    def online(self) -> IndustrialModel | None:
        return self._online.model if self._online else None

    @property
    def online_version(self) -> Version | None:
        return self._online

    @property
    def candidate(self) -> IndustrialModel | None:
        return self._candidate.model if self._candidate else None

    @property
    def history(self) -> list[Version]:
        return list(self._history)

    @property
    def has_online(self) -> bool:
        return self._online is not None

    # ---- 写入 ----
    def stage(self, model: IndustrialModel, batch_index: int, action: str) -> Version:
        """登记候选版本（尚未上线）。"""
        ver = Version(
            version=self._next_version,
            model=model,
            batch_index=batch_index,
            action=action,
        )
        self._candidate = ver
        return ver

    def promote(self, score: float = float("nan")) -> Version:
        """候选上线。旧在线版本进历史。"""
        if self._candidate is None:
            raise RuntimeError("没有候选版本可上线")
        self._candidate.score = score
        if self._online is not None:
            self._history.append(self._online)
        self._online = self._candidate
        self._candidate = None
        self._next_version += 1
        self.consecutive_failures = 0
        return self._online

    def discard(self) -> None:
        """丢弃候选，保留当前在线模型。"""
        self._candidate = None
        self._next_version += 1
        self.consecutive_failures += 1

    def bootstrap(self, model: IndustrialModel, batch_index: int) -> Version:
        """冷启动：首个在线模型，不经过门控。"""
        self.stage(model, batch_index, action="BOOTSTRAP")
        return self.promote(score=float("nan"))

    # ---- 回滚 ----
    def rollback_to_best(self, batch_index: int) -> Version | None:
        """回滚到历史上门控分最高的版本。"""
        pool = [v for v in self._history if v.score == v.score]  # 过滤 NaN
        if not pool:
            return None
        best = max(pool, key=lambda v: v.score)
        if self._online is not None and best.version == self._online.version:
            return None
        restored = Version(
            version=self._next_version,
            model=copy.deepcopy(best.model),
            batch_index=batch_index,
            action=f"ROLLBACK_TO_V{best.version}",
            score=best.score,
        )
        if self._online is not None:
            self._history.append(self._online)
        self._online = restored
        self._next_version += 1
        self.consecutive_failures = 0
        self.rollbacks.append(
            {"batch_index": batch_index, "restored_from": best.version}
        )
        return restored
