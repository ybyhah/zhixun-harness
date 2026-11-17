"""误差流突变检测。

把"每个样本是否预测错误"当成一条 0/1 流喂给 ADWIN 或 DDM。
作用是区分渐变与突变：性能指标只能告诉你掉了多少，
误差流检测器能告诉你是哪一刻开始掉的。

依赖 frouros；若不可用则降级为空检测器，不影响闭环。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from frouros.detectors.concept_drift import ADWIN, DDM

    _HAS_FROUROS = True
except ImportError:  # pragma: no cover
    _HAS_FROUROS = False


@dataclass
class ErrorStreamResult:
    detector: str
    drift_detected: bool
    error_rate: float
    n_updates: int

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "drift_detected": self.drift_detected,
            "error_rate": self.error_rate,
            "n_updates": self.n_updates,
        }


class ErrorStreamMonitor:
    """跨批次保持状态，因此必须由闭环长期持有同一个实例。"""

    def __init__(self, kind: str = "adwin") -> None:
        self.kind = kind if (_HAS_FROUROS and kind in {"adwin", "ddm"}) else "none"
        self._det = self._make()
        self._n = 0

    def _make(self):
        if self.kind == "adwin":
            return ADWIN()
        if self.kind == "ddm":
            return DDM()
        return None

    def reset(self) -> None:
        """模型更新后调用，让检测器基于新模型重新计数。"""
        self._det = self._make()
        self._n = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> ErrorStreamResult:
        errors = (np.asarray(y_true) != np.asarray(y_pred)).astype(float)
        rate = float(errors.mean()) if errors.size else float("nan")
        if self._det is None:
            return ErrorStreamResult("none", False, rate, self._n)

        detected = False
        for e in errors:
            self._det.update(value=e)
            self._n += 1
            if getattr(self._det, "drift", False):
                detected = True
        if detected:
            # 报警后重置，避免同一次漂移持续刷警报
            self.reset()
        return ErrorStreamResult(self.kind, detected, rate, self._n)
