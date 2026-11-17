"""增量标准化：scaler 只允许用已见过的数据拟合。

这是防止"偷看未来"的关键组件。每次 partial_fit 之后统计量才更新，
transform 永远基于当前已见数据。
"""

from __future__ import annotations

import numpy as np


class IncrementalScaler:
    """基于 Welford 式累计的标准化器。"""

    def __init__(self, n_features: int) -> None:
        self._n = 0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._m2 = np.zeros(n_features, dtype=np.float64)

    @property
    def fitted(self) -> bool:
        return self._n > 0

    @property
    def n_seen(self) -> int:
        return self._n

    def copy(self) -> "IncrementalScaler":
        """冻结当前统计量的快照。

        模型必须带着"训练时的那一份"标准化参数，否则训练与推理的
        量纲不一致。第一阶段批次 2 出现负收益就是这个原因。
        """
        out = IncrementalScaler(self._mean.shape[0])
        out._n = self._n
        out._mean = self._mean.copy()
        out._m2 = self._m2.copy()
        return out

    def subset(self, cols: np.ndarray) -> "IncrementalScaler":
        """按列索引裁剪统计量，供 FEATURE_SUBSET 动作保持 scaler 与特征对齐。"""
        cols = np.asarray(cols, dtype=int)
        out = IncrementalScaler(cols.shape[0])
        out._n = self._n
        out._mean = self._mean[cols].copy()
        out._m2 = self._m2[cols].copy()
        return out

    def partial_fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        for row in X:
            self._n += 1
            delta = row - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (row - self._mean)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.fitted:
            return X
        std = np.sqrt(self._m2 / max(self._n - 1, 1))
        std[std < 1e-12] = 1.0
        return (X - self._mean) / std


def fill_missing(X: np.ndarray) -> np.ndarray:
    """缺失与非有限值统一置 0（标准化后 0 即均值）。"""
    X = np.asarray(X, dtype=np.float64).copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = 0.0
    return X
