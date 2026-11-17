"""串行批次流：本地还原 B 包的"完成当前批次才能拿下一批"约束。

两条硬规则：
1. next_batch 只返回 X，标签在 submit 之后才给。
2. 未提交就取下一批、或重复提交，都直接抛异常。

这样"预测时用到了标签"这类错误会在接口层暴露，而不是悄悄提高分数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..datasets.base import Batch

OFFLINE = "OFFLINE"
SubmitKind = Literal["prediction", "offline"]


class StreamProtocolError(RuntimeError):
    """违反串行卡点约束。"""


@dataclass(frozen=True)
class BatchView:
    """交给系统的视图，只有特征，没有标签。"""

    index: int
    X: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass
class SubmitResult:
    index: int
    y_true: np.ndarray
    kind: SubmitKind


class BatchStream:
    def __init__(self, batches: list[Batch]) -> None:
        if not batches:
            raise ValueError("批次列表为空")
        self._batches = batches
        self._cursor = 0
        self._awaiting = False
        self.offline_batches: list[int] = []

    # ---- 状态 ----
    @property
    def n_batches(self) -> int:
        return len(self._batches)

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._batches)

    # ---- 取数 ----
    def next_batch(self) -> BatchView | None:
        if self._awaiting:
            raise StreamProtocolError(
                f"批次 {self._cursor} 尚未提交结果，不能获取下一批"
            )
        if self.exhausted:
            return None
        self._awaiting = True
        b = self._batches[self._cursor]
        return BatchView(index=b.index, X=b.X)

    # ---- 提交 ----
    def submit(self, output: np.ndarray | str) -> SubmitResult:
        """提交预测结果，或提交 OFFLINE 表示本批次不提供服务。"""
        if not self._awaiting:
            raise StreamProtocolError("当前没有待提交的批次")
        b = self._batches[self._cursor]
        if isinstance(output, str):
            if output != OFFLINE:
                raise ValueError(f"非法的提交内容: {output!r}")
            kind: SubmitKind = "offline"
            self.offline_batches.append(b.index)
        else:
            pred = np.asarray(output)
            if pred.shape[0] != b.X.shape[0]:
                raise ValueError(
                    f"预测数量 {pred.shape[0]} 与批次样本数 {b.X.shape[0]} 不一致"
                )
            kind = "prediction"
        self._awaiting = False
        self._cursor += 1
        return SubmitResult(index=b.index, y_true=b.y.copy(), kind=kind)
