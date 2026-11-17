"""任务适配协议。

不同数据集（气体传感器/轴承/发动机）通过实现 TaskAdapter 接入，
主闭环只依赖这里的抽象，不感知具体任务。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

TaskType = Literal["classification", "regression"]


@dataclass
class Batch:
    """一个批次的数据。index 从 0 开始。"""

    index: int
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


@runtime_checkable
class TaskAdapter(Protocol):
    """任务适配器。"""

    @property
    def task_type(self) -> TaskType: ...

    @property
    def metric_name(self) -> str: ...

    @property
    def class_set(self) -> np.ndarray | None:
        """分类任务的全局类别集合；回归任务返回 None。

        固定全局类别集合是为了避免某批次缺类时模型输出维度跳变。
        """
        ...

    def load_batches(self) -> list[Batch]: ...
