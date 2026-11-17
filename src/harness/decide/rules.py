"""规则决策器（快通路）。

设计意图：不依赖大模型就能给出有依据的动作，保证恢复速度和"无人干预跑完"。
第二阶段的大模型决策器实现同一个 Decider 协议，主闭环无需改动。

规则来自 doc.md 第 3.3 节判定表，并带两条保护：
- 小批次保护：样本量过小只累积不更新，避免噪声触发误更新
- 升级策略：同一漂移若上一次的廉价动作被门控否决，这次换更重的动作
"""

from __future__ import annotations

import numpy as np

from ..monitor.report import DriftReport
from .actions import Action, Decision

# 每种漂移类型的动作阶梯，从便宜到贵
LADDER: dict[str, list[Action]] = {
    "LABEL": [Action.ADJUST_THRESHOLD, Action.RECALIBRATE, Action.WINDOW_RETRAIN],
    "COVARIATE_HARMFUL": [Action.WINDOW_RETRAIN, Action.FULL_RETRAIN, Action.FEATURE_SUBSET],
    "CONCEPT": [Action.WINDOW_RETRAIN, Action.FULL_RETRAIN],
    "UNTRUSTED": [Action.FULL_RETRAIN, Action.FEATURE_SUBSET],
}


class RuleDecider:
    def __init__(
        self,
        min_batch_size: int = 30,
        default_window: int = 3,
        concept_decay: float = 0.6,
        allow_offline: bool = False,
    ) -> None:
        self.min_batch_size = min_batch_size
        self.default_window = default_window
        self.concept_decay = concept_decay
        self.allow_offline = allow_offline

    def decide(self, report: DriftReport, history: list) -> Decision:
        """返回单个动作。等价于 decide_many 的首项，保持旧调用兼容。"""
        return self.decide_many(report, history, 1)[0]

    def decide_many(
        self, report: DriftReport, history: list, limit: int = 2
    ) -> list[Decision]:
        """返回一组候选动作，按成本升序。

        动机：一次只给一个候选时，被门控否决就只能等下一批再试，
        漂移迟迟恢复不了。给出多个候选让门控择优，用训练预算换恢复速度。
        """
        guard = self._guard(report)
        if guard is not None:
            return [guard]

        dtype = report.drift_type
        ladder = LADDER.get(dtype, [Action.WINDOW_RETRAIN])
        step = self._failed_attempts(history, dtype)

        if (
            dtype == "UNTRUSTED"
            and self.allow_offline
            and step >= len(ladder)
        ):
            return [
                Decision(
                    Action.OFFLINE,
                    reason="重训多次仍无法恢复，模型不可信，本批次拒判",
                )
            ]

        start = min(step, len(ladder) - 1)
        picked = ladder[start : start + max(1, limit)]
        return [
            Decision(
                act,
                params=self._params(act, report),
                reason=self._reason(act, report, step)
                + ("" if i == 0 else f"（备选 {i + 1}，门控择优）"),
            )
            for i, act in enumerate(picked)
        ]

    def _guard(self, report: DriftReport) -> Decision | None:
        """不需要更新的两种情形。"""
        if report.perf.n_samples < self.min_batch_size:
            return Decision(
                Action.NOOP,
                reason=(
                    f"本批仅 {report.perf.n_samples} 个样本，低于 "
                    f"{self.min_batch_size}，只累积数据不更新"
                ),
            )
        if report.drift_type in ("STABLE", "COVARIATE_BENIGN"):
            return Decision(
                Action.NOOP,
                reason=(
                    "性能未下降，不更新"
                    if report.drift_type == "STABLE"
                    else "输入分布已变但性能未降，模型仍适用，不更新"
                ),
            )
        return None

    # ---- 内部 ----
    @staticmethod
    def _failed_attempts(history: list, dtype: str) -> int:
        """统计最近针对同类漂移、连续被门控否决的次数，用于动作升级。"""
        n = 0
        for rec in reversed(history):
            gate = getattr(rec, "gate_result", None) or {}
            if not gate:
                continue
            if getattr(rec, "drift_type", None) != dtype:
                break
            if gate.get("accepted"):
                break
            n += 1
        return n

    def _params(self, action: Action, report: DriftReport) -> dict:
        if action in (Action.WINDOW_RETRAIN, Action.INCREMENTAL):
            return {"window": self.default_window}
        if action == Action.RECALIBRATE:
            return {"window": 2}
        if action == Action.FULL_RETRAIN:
            decay = self.concept_decay if report.drift_type == "CONCEPT" else 0.9
            return {"decay": decay}
        if action == Action.FEATURE_SUBSET:
            return {
                "window": self.default_window,
                "drop_features": list(report.dist.psi_top_features[:3]),
            }
        return {}

    @staticmethod
    def _reason(action: Action, report: DriftReport, step: int) -> str:
        head = (
            f"{report.drift_type}，性能 {report.perf.value:.4f} "
            f"较参考 {report.perf_baseline:.4f} 变化 {report.perf_delta:+.4f}"
        )
        tail = {
            Action.ADJUST_THRESHOLD: "类别占比变化为主，先调整决策偏置（成本最低）",
            Action.RECALIBRATE: "阈值调整未通过门控，改做概率重校准",
            Action.WINDOW_RETRAIN: "用最近窗口重训以贴合新分布",
            Action.FULL_RETRAIN: "窗口重训未通过门控，改用全量数据加时间衰减",
            Action.FEATURE_SUBSET: "剔除漂移最严重的特征后重训",
        }.get(action, "执行默认更新动作")
        if step > 0:
            tail += f"（同类漂移已连续 {step} 次被门控否决，动作升级）"
        return f"{head}；{tail}"
