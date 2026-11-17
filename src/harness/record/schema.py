"""决策记录结构。

字段设计同时服务三个目的：
1. 门槛要求的"读历史 → 提假设 → 做实验 → 做决策"闭环证据
   （对应 history_read / hypothesis / experiments / decision 四个字段）
2. 可审计：每次更新都能回答"为什么这么做、依据是什么、结果如何"
3. 复盘与第二阶段的策略自改进（outcome 事后回填）

留有扩展位，等官方《实验记录规范》发布后再对齐字段命名。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DecisionRecord:
    batch_index: int
    trigger: str                                  # 触发原因
    drift_type: str
    evidence: dict[str, Any] = field(default_factory=dict)   # 性能/PSI/KS/误差流
    history_read: list[int] = field(default_factory=list)    # 读了哪些历史记录
    hypothesis: str | None = None                            # 假设
    experiments: list[dict[str, Any]] = field(default_factory=list)  # 实验及结果
    decision: dict[str, Any] = field(default_factory=dict)   # 最终决策
    gate_result: dict[str, Any] = field(default_factory=dict)
    model_version: int | None = None
    outcome: dict[str, Any] = field(default_factory=dict)     # 事后回填
    cost: dict[str, Any] = field(default_factory=dict)        # 训练次数/耗时/Token
    extra: dict[str, Any] = field(default_factory=dict)       # 扩展位

    # ---- 闭环完整性 ----
    def is_complete_loop(self) -> bool:
        """是否构成一次完整的"读历史→提假设→做实验→做决策"闭环。"""
        return bool(
            self.history_read
            and self.hypothesis
            and self.experiments
            and self.decision
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DecisionRecord:
        known = {f for f in DecisionRecord.__dataclass_fields__}
        payload = {k: v for k, v in d.items() if k in known}
        return DecisionRecord(**payload)
