"""把官方 validation_after 统一成"越大越好"分数，并复用 harness.metrics 汇总。

分类主指标 Macro-F1（越大越好）；回归主指标 RMSE（越小越好）→ 统一 score = -RMSE。
"""

from __future__ import annotations

from harness import metrics
from .package import OfficialPackage


def round_value(pkg: OfficialPackage, validation_after: dict) -> float:
    """官方口径的主指标原始值（分类 Macro-F1，回归 RMSE）。缺失返回 NaN。"""
    if not validation_after:
        return float("nan")
    key = "macro_f1" if pkg.task == "classification" else "rmse"
    return float(validation_after.get(key, float("nan")))


def to_score(pkg: OfficialPackage, value: float) -> float:
    """统一成越大越好。"""
    if value != value:
        return float("nan")
    return value if pkg.higher_better else -value


def summarize_rounds(pkg: OfficialPackage, values: list[float], *,
                     seconds: float, trainings: int, promotions: int):
    """跨轮汇总：综合、最差阶段、恢复。全部用统一 score。"""
    scores = [to_score(pkg, v) for v in values]
    return metrics.summarize(
        scores, trainings=trainings, seconds=seconds,
        worst_window=2, promotions=promotions,
    )


def nasa_asym_score(y_true, y_pred) -> float:
    """NASA C-MAPSS 非对称分，越低越好；高估剩余寿命（d>0）惩罚更重。

    官方 train.py 的 metrics 只给 mae/rmse，平台会另报此项，故单独实现。
    注意：这是 **C-MAPSS（循环）专用**口径。`exp(d/10)` 对 B 的分钟量级（可达上千）
    会溢出成无意义的天文数字，不要用于 B。
    """
    import numpy as np
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def over_estimate_rate(y_true, y_pred) -> float:
    """高估比例：预测大于真值的样本占比（对应 C 的"减少寿命高估"）。"""
    import numpy as np
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(d > 0)) if d.size else float("nan")
