"""与官方 `baselines/train.py` 的 `metrics()` 完全一致的纯函数版指标。

为什么不直接调官方 `metrics(model, data)`：它内部每次都 `model.predict`，
bootstrap 要重采样几百次，不能每次重新推理。所以这里把**公式**抽成纯函数，
输入已经算好的 pred 与 y，再按下标子集计算。

口径必须与官方逐位一致，否则门控优化的东西和平台打分的东西不是一回事：
分类 Macro-F1 在 `set(expected) ∪ set(argmax(pred))` 上宏平均
（**不是只看真实出现的类**），这一点由 tests/test_significance.py 强制断言。
"""

from __future__ import annotations

import numpy as np


def classification_macro_f1(pred: np.ndarray, y: np.ndarray,
                            idx: np.ndarray | None = None) -> float:
    """pred 为 (n, K) 概率；完全照抄官方公式。"""
    labels = np.asarray(pred).argmax(axis=1)
    expected = np.asarray(y)
    if idx is not None:
        labels, expected = labels[idx], expected[idx]
    if expected.size == 0:
        return float("nan")
    observed = sorted(set(expected.tolist()) | set(labels.tolist()))
    f1 = []
    for label in observed:
        denom = np.sum(labels == label) + np.sum(expected == label)
        # 官方实现未对分母做保护；分母为 0 只可能在 label 不属于并集时出现，
        # 而 observed 来自并集，故此处分母必 > 0。保留断言式写法避免静默偏差。
        f1.append(2 * np.sum((labels == label) & (expected == label)) / denom)
    return float(np.mean(f1))


def regression_rmse(pred: np.ndarray, y: np.ndarray,
                    idx: np.ndarray | None = None) -> float:
    p, e = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if idx is not None:
        p, e = p[idx], e[idx]
    if e.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - e) ** 2)))


def score(pred: np.ndarray, y: np.ndarray, task: str,
          idx: np.ndarray | None = None) -> float:
    """统一成"越大越好"：分类取 Macro-F1，回归取 -RMSE。"""
    if task == "classification":
        return classification_macro_f1(pred, y, idx)
    r = regression_rmse(pred, y, idx)
    return -r if r == r else float("nan")
