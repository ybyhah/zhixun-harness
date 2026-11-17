"""A3 发动机剩余寿命（RUL）任务适配器：NASA C-MAPSS。

数据：4 个子集 FD001~FD004，每行 26 列（空格分隔）：
  unit, cycle, op_setting1~3, sensor1~21
训练集每台发动机运行到失效；RUL = 该发动机最大循环数 - 当前循环数，
并按 rul_cap 做分段线性截断（早期退化不明显，远期 RUL 预测无意义）。

回归任务：task_type="regression"，class_set 为 None。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Batch, TaskType

N_COLS = 26
SETTING_COLS = [2, 3, 4]                       # op_setting 1~3（0-based）
SENSOR_COLS = list(range(5, 26))               # 21 个传感器
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
# 复杂度递增顺序：单工况单故障 → 单工况双故障 → 六工况单故障 → 六工况双故障
PROGRESSIVE_ORDER = ["FD001", "FD003", "FD002", "FD004"]


def _parse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (unit 列, cycle 列, 26 列原始矩阵)。"""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [float(x) for x in line.split()]
        rows.append(vals[:N_COLS])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr


def _rul_from_train(units: np.ndarray, cycles: np.ndarray, cap: float) -> np.ndarray:
    """训练集 RUL = 每台发动机的最大循环 - 当前循环，截断到 cap。"""
    rul = np.zeros(units.shape[0], dtype=np.float64)
    for u in np.unique(units):
        m = units == u
        rul[m] = cycles[m].max() - cycles[m]
    return np.minimum(rul, cap)


class CMAPSSAdapter:
    """实现 TaskAdapter 协议（回归）。

    批次构造：
      - progressive：FD001→FD003→FD002→FD004，每子集内按 unit 分组切 batches_per_subset 批
      - single：仅用 single_subset，按 unit 号切 n_batches 批
    """

    def __init__(
        self,
        raw_dir: str | Path,
        mode: str = "progressive",
        rul_cap: float = 125.0,
        batches_per_subset: int = 3,
        single_subset: str = "FD001",
        n_batches: int = 10,
        drop_constant_sensors: bool = True,
        seed: int = 42,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.mode = mode
        self.rul_cap = rul_cap
        self.batches_per_subset = batches_per_subset
        self.single_subset = single_subset
        self.n_batches = n_batches
        self.drop_constant_sensors = drop_constant_sensors
        self.seed = seed
        self.dropped_sensors: list[int] = []
        self._feature_cols: list[int] = SETTING_COLS + SENSOR_COLS

    @property
    def task_type(self) -> TaskType:
        return "regression"

    @property
    def metric_name(self) -> str:
        return "rmse"

    @property
    def class_set(self) -> None:
        return None

    # ---- 加载 ----
    def _subset_batches(self, subset: str, n_parts: int, start_index: int) -> list[Batch]:
        units, cycles, arr = _parse(self.raw_dir / f"train_{subset}.txt")
        rul = _rul_from_train(units, cycles, self.rul_cap)
        X = arr[:, self._feature_cols]
        uniq = np.unique(units)
        groups = np.array_split(uniq, min(n_parts, len(uniq)))
        batches = []
        for k, g in enumerate(groups):
            mask = np.isin(units, g)
            batches.append(Batch(index=start_index + k, X=X[mask], y=rul[mask]))
        return batches

    def load_batches(self) -> list[Batch]:
        if self.mode == "single":
            order, per = [self.single_subset], self.n_batches
        else:
            order, per = PROGRESSIVE_ORDER, self.batches_per_subset

        # 先确定要剔除的常数传感器（基于第一个子集的训练数据）
        if self.drop_constant_sensors:
            self._compute_constant_sensors(order[0])

        batches: list[Batch] = []
        idx = 0
        for subset in order:
            sub = self._subset_batches(subset, per, idx)
            batches.extend(sub)
            idx += len(sub)
        return batches

    def _compute_constant_sensors(self, subset: str) -> None:
        _, _, arr = _parse(self.raw_dir / f"train_{subset}.txt")
        cols = SETTING_COLS + SENSOR_COLS
        sub = arr[:, cols]
        std = sub.std(axis=0)
        keep_local = std > 1e-8
        self.dropped_sensors = [cols[i] for i in range(len(cols)) if not keep_local[i]]
        self._feature_cols = [cols[i] for i in range(len(cols)) if keep_local[i]]
