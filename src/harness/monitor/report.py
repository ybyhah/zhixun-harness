"""漂移报告与类型判定。

判定表（doc.md 第 3.3 节）：

| 分布变化 | 性能变化 | 判定              | 建议动作             |
|---------|---------|-------------------|---------------------|
| 无      | 无       | STABLE            | 不动                 |
| 有      | 无       | COVARIATE_BENIGN  | 不动（只记录）        |
| 有      | 下降     | COVARIATE_HARMFUL | 窗口重训 / 增量       |
| 无      | 下降     | CONCEPT           | 重训并降低旧数据权重  |
| 类别占比变 | 下降   | LABEL             | 调阈值 / 重校准       |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .distribution import DistResult
from .error_stream import ErrorStreamResult
from .performance import PerfResult

DriftType = Literal[
    "STABLE",
    "COVARIATE_BENIGN",
    "COVARIATE_HARMFUL",
    "CONCEPT",
    "LABEL",
    "UNTRUSTED",
]


@dataclass
class DriftReport:
    batch_index: int
    perf: PerfResult
    dist: DistResult
    error_stream: ErrorStreamResult
    perf_baseline: float            # 参考性能（近期均值或上线时成绩）
    perf_delta: float               # 当前 - 参考，负值代表下降
    label_shift: float
    drift_type: DriftType = "STABLE"
    notes: list[str] = field(default_factory=list)

    @property
    def perf_dropped(self) -> bool:
        return self.drift_type in {"COVARIATE_HARMFUL", "CONCEPT", "LABEL", "UNTRUSTED"}

    def to_dict(self) -> dict:
        return {
            "batch_index": self.batch_index,
            "drift_type": self.drift_type,
            "perf": self.perf.to_dict(),
            "perf_baseline": self.perf_baseline,
            "perf_delta": self.perf_delta,
            "label_shift": self.label_shift,
            "dist": self.dist.to_dict(),
            "error_stream": self.error_stream.to_dict(),
            "notes": list(self.notes),
        }


def classify(
    batch_index: int,
    perf: PerfResult,
    dist: DistResult,
    err: ErrorStreamResult,
    perf_baseline: float,
    label_shift_value: float,
    perf_drop: float = 0.05,
    label_shift_threshold: float = 0.2,
    untrusted_drop: float = 0.30,
    task_type: str = "classification",
) -> DriftReport:
    notes: list[str] = []
    # 统一用 score（越大越好）计算 delta，baseline 也是 score。
    # 分类下 score==Macro-F1；回归下 score==-RMSE，方向一致，无需改符号。
    delta = (
        perf.score - perf_baseline
        if perf.score == perf.score and perf_baseline == perf_baseline
        else float("nan")
    )
    dropped = delta == delta and delta < -perf_drop
    dist_changed = dist.drifted or err.drift_detected
    # 回归：label_shift_value 复用为"目标值分布漂移"程度
    label_changed = label_shift_value > label_shift_threshold
    is_reg = task_type == "regression"

    if perf.classes_missing:
        notes.append(f"本批缺少类别 {perf.classes_missing}，指标只在出现类别上宏平均")
    if err.drift_detected:
        notes.append(f"{err.detector} 在误差流上检出突变")

    if delta == delta and delta < -untrusted_drop:
        dtype: DriftType = "UNTRUSTED"
        notes.append("性能剧烈下降，模型可信度低")
    elif not dropped:
        dtype = "COVARIATE_BENIGN" if dist_changed else "STABLE"
        if dist_changed:
            notes.append("输入分布已变但性能未降，模型仍适用，本批不更新")
    elif dist_changed:
        dtype = "COVARIATE_HARMFUL"
        notes.append(f"分布漂移致失准，PSI 最高特征 {dist.psi_top_features}")
        if label_changed:
            tgt = "目标值分布变化" if is_reg else "类别占比变化"
            notes.append(f"同时伴随{tgt} {label_shift_value:.3f}")
    elif label_changed and not is_reg:
        # 仅分类有"调阈值/重校准"这类针对标签占比的廉价动作
        dtype = "LABEL"
        notes.append(f"输入分布稳定而类别占比变化 {label_shift_value:.3f}，优先调阈值或重校准")
    else:
        dtype = "CONCEPT"
        if is_reg and label_changed:
            notes.append(f"目标值分布变化 {label_shift_value:.3f} 且性能下降，判为概念漂移")
        else:
            notes.append("输入分布未显著变化而性能下降，判为概念漂移")

    return DriftReport(
        batch_index=batch_index,
        perf=perf,
        dist=dist,
        error_stream=err,
        perf_baseline=perf_baseline,
        perf_delta=delta,
        label_shift=label_shift_value,
        drift_type=dtype,
        notes=notes,
    )
