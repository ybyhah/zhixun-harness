"""A1 气体传感器漂移任务适配器。

数据：UCI Gas Sensor Array Drift，16 个传感器 x 8 个特征 = 128 维，
6 类气体（1=Ethanol 2=Ethylene 3=Ammonia 4=Acetaldehyde 5=Acetone 6=Toluene），
10 个批次按时间先后排列，天然构成漂移流。

注意：批次 3/4/5 完全没有 Toluene 样本，批次 10 每类 600 个。
因此类别集合必须全局固定，不能按批推断。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Batch, TaskType

N_FEATURES = 128
CLASS_SET = np.arange(1, 7)  # 1..6
CLASS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}


def _parse_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """解析 libsvm 风格文件：<label> <idx>:<value> ..."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 标签字段可能是 "1" 或 "1;10.0"（含浓度），只取分号前的类别
        ys.append(int(float(parts[0].split(";")[0])))
        row = np.zeros(N_FEATURES, dtype=np.float64)
        for tok in parts[1:]:
            idx, _, val = tok.partition(":")
            i = int(idx) - 1
            if not 0 <= i < N_FEATURES:
                raise ValueError(f"{path.name}:{lineno} 特征下标越界: {idx}")
            row[i] = float(val)
        xs.append(row)
    if not xs:
        raise ValueError(f"{path} 为空")
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)


class GasSensorAdapter:
    """实现 TaskAdapter 协议。"""

    def __init__(self, raw_dir: str | Path, n_batches: int = 10) -> None:
        self.raw_dir = Path(raw_dir)
        self.n_batches = n_batches

    @property
    def task_type(self) -> TaskType:
        return "classification"

    @property
    def metric_name(self) -> str:
        return "macro_f1"

    @property
    def class_set(self) -> np.ndarray:
        return CLASS_SET.copy()

    def load_batches(self) -> list[Batch]:
        batches: list[Batch] = []
        for i in range(1, self.n_batches + 1):
            path = self.raw_dir / f"batch{i}.dat"
            if not path.exists():
                raise FileNotFoundError(
                    f"缺少 {path}，请先运行 scripts/download_data.py"
                )
            X, y = _parse_file(path)
            if X.shape[1] != N_FEATURES:
                raise ValueError(f"{path.name} 特征维度异常: {X.shape}")
            batches.append(Batch(index=i - 1, X=X, y=y))
        return batches
