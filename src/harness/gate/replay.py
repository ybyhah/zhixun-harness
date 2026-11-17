"""历史工况回放集。

用途：候选模型必须在历史工况上不显著退化才能上线，对应客观分里的
"稳定性与抗遗忘"5 分。

做法是分层抽样，保证每个历史批次、每个类别都有代表；容量超限时按
"越旧越容易被淘汰"的概率抽取保留，而不是直接截断，避免早期工况完全丢失。
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        per_batch: int = 60,
        capacity: int = 600,
        seed: int = 42,
    ) -> None:
        self.per_batch = per_batch
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._bid: list[np.ndarray] = []

    @property
    def empty(self) -> bool:
        return not self._X

    def __len__(self) -> int:
        return int(sum(a.shape[0] for a in self._X))

    @property
    def batch_ids(self) -> list[int]:
        return [int(b[0]) for b in self._bid]

    def add(self, X: np.ndarray, y: np.ndarray, batch_index: int) -> None:
        """从一个批次里分层抽样若干样本存入。"""
        X, y = np.asarray(X), np.asarray(y)
        if X.shape[0] == 0:
            return
        picked = self._stratified(y, self.per_batch)
        self._X.append(X[picked])
        self._y.append(y[picked])
        self._bid.append(np.full(picked.size, batch_index, dtype=int))
        self._evict()

    def dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.empty:
            empty = np.empty((0,), dtype=int)
            return np.empty((0, 0)), empty, empty
        return (
            np.vstack(self._X),
            np.concatenate(self._y),
            np.concatenate(self._bid),
        )

    # ---- 内部 ----
    def _stratified(self, y: np.ndarray, n: int) -> np.ndarray:
        """按类别均摊名额，保证稀有类不被挤掉。"""
        classes = np.unique(y)
        if n >= y.size:
            return np.arange(y.size)
        per = max(1, n // len(classes))
        chosen: list[int] = []
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            take = min(per, idx.size)
            chosen.extend(self._rng.choice(idx, size=take, replace=False).tolist())
        # 名额没用满时随机补齐
        if len(chosen) < n:
            rest = np.setdiff1d(np.arange(y.size), np.asarray(chosen, dtype=int))
            extra = min(n - len(chosen), rest.size)
            if extra > 0:
                chosen.extend(self._rng.choice(rest, size=extra, replace=False).tolist())
        return np.sort(np.asarray(chosen, dtype=int))

    def _evict(self) -> None:
        """超容量时按批次新旧加权下采样，保留全部批次的代表性。"""
        while len(self) > self.capacity and len(self._X) > 1:
            n_batches = len(self._X)
            # 越旧的批次保留比例越低，但不清零
            for i in range(n_batches):
                keep_ratio = 0.5 + 0.5 * (i / max(n_batches - 1, 1))
                cur = self._X[i].shape[0]
                target = max(1, int(cur * keep_ratio))
                if target < cur:
                    sel = self._stratified(self._y[i], target)
                    self._X[i] = self._X[i][sel]
                    self._y[i] = self._y[i][sel]
                    self._bid[i] = self._bid[i][sel]
            if len(self) > self.capacity:
                # 仍超限则丢掉最旧的一个批次
                self._X.pop(0)
                self._y.pop(0)
                self._bid.pop(0)
