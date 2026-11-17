"""漂移模拟器：在 A 包数据上注入人工漂移。

三个用途：
1. 对照实验的统一测试床（A 包自带漂移模式固定，样本有限）
2. 策略层自改进的训练环境（第二阶段）
3. 进阶加分要求的"扩展数据验证"材料

支持协变量漂移、标签漂移、概念漂移，各有突变与渐变两种速率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..datasets.base import Batch

DriftKind = Literal["covariate", "label", "concept"]
Speed = Literal["sudden", "gradual"]


@dataclass
class DriftSpec:
    kind: DriftKind
    speed: Speed = "gradual"
    start: int = 0          # 从第几个批次开始
    magnitude: float = 1.0  # 强度
    seed: int = 42


def _ramp(idx: int, spec: DriftSpec, n: int) -> float:
    """返回 [0,1] 的强度系数。"""
    if idx < spec.start:
        return 0.0
    if spec.speed == "sudden":
        return 1.0
    span = max(n - 1 - spec.start, 1)
    return min((idx - spec.start) / span, 1.0)


def _covariate(X: np.ndarray, k: float, spec: DriftSpec, rng) -> np.ndarray:
    """特征偏移 + 缩放 + 噪声，模拟传感器老化。"""
    if k <= 0:
        return X
    scale = np.sqrt(np.maximum(X.var(axis=0), 1e-12))
    offset = rng.normal(0.0, 1.0, size=X.shape[1]) * scale
    gain = 1.0 + rng.normal(0.0, 0.1, size=X.shape[1])
    noise = rng.normal(0.0, 1.0, size=X.shape) * scale * 0.1
    m = k * spec.magnitude
    return X * (1.0 + m * (gain - 1.0)) + m * offset + m * noise


def _label(X: np.ndarray, y: np.ndarray, k: float, spec: DriftSpec, rng):
    """按类别重采样，改变 P(Y)。"""
    if k <= 0:
        return X, y
    classes = np.unique(y)
    weights = rng.dirichlet(np.ones(len(classes)) * max(1.0 - k * spec.magnitude, 0.05))
    keep: list[int] = []
    for cls, w in zip(classes, weights):
        idx = np.flatnonzero(y == cls)
        n_keep = max(int(round(len(idx) * float(w) * len(classes))), 1)
        n_keep = min(n_keep, len(idx))
        keep.extend(rng.choice(idx, size=n_keep, replace=False).tolist())
    keep_arr = np.sort(np.asarray(keep, dtype=int))
    return X[keep_arr], y[keep_arr]


def _concept(X: np.ndarray, y: np.ndarray, k: float, spec: DriftSpec, rng):
    """在特征空间的一个子区域内交换两个类别的标签，改变 P(Y|X)。"""
    if k <= 0:
        return y
    classes = np.unique(y)
    if len(classes) < 2:
        return y
    a, b = rng.choice(classes, size=2, replace=False)
    feat = int(rng.integers(0, X.shape[1]))
    thresh = np.median(X[:, feat])
    region = X[:, feat] > thresh
    frac = min(k * spec.magnitude, 1.0)
    y_new = y.copy()
    for src, dst in ((a, b), (b, a)):
        cand = np.flatnonzero(region & (y == src))
        if cand.size == 0:
            continue
        n_flip = int(round(cand.size * frac))
        if n_flip > 0:
            y_new[rng.choice(cand, size=n_flip, replace=False)] = dst
    return y_new


def inject(batches: list[Batch], specs: list[DriftSpec]) -> list[Batch]:
    """按 specs 在批次序列上注入漂移，返回新序列（不修改入参）。"""
    n = len(batches)
    out: list[Batch] = []
    for i, b in enumerate(batches):
        X, y = b.X.copy(), b.y.copy()
        for spec in specs:
            rng = np.random.default_rng(spec.seed + i)
            k = _ramp(i, spec, n)
            if spec.kind == "covariate":
                X = _covariate(X, k, spec, rng)
            elif spec.kind == "label":
                X, y = _label(X, y, k, spec, rng)
            elif spec.kind == "concept":
                y = _concept(X, y, k, spec, rng)
        out.append(Batch(index=b.index, X=X, y=y))
    return out
