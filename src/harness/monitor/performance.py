"""性能指标，分类与回归共用。

分类缺类处理：A1 的批次 3/4/5 没有 Toluene 样本。Macro-F1 若按全局 6 类计算，
缺失类别会贡献 0 分，把指标压低。因此只在"本批出现过的类别"上做宏平均。

回归分数方向：门控、参照线、恢复速度全部假设"分数越大越好"。回归 RMSE 越小越好，
为避免符号散落各处，**统一用 `score = -RMSE` 对外暴露**，仅在报告层转回正 RMSE。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ---- 回归指标 ----

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def asym_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """C-MAPSS 官方非对称评分，越低越好。

    预测偏晚（RUL 高估，d>=0）惩罚更重，因为漏报故障更危险。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


@dataclass
class PerfResult:
    metric_name: str
    value: float                 # 该任务的主指标原始值（分类=Macro-F1；回归=RMSE）
    score: float                 # 统一分数，越大越好（分类=value；回归=-RMSE）
    n_samples: int
    # 分类字段
    accuracy: float = float("nan")
    macro_f1: float = float("nan")
    classes_present: list[int] = field(default_factory=list)
    classes_missing: list[int] = field(default_factory=list)
    # 关键类召回（如 A2 的严重退化类）；该类未出现或未配置时为 NaN。漏报率 = 1 - 该值
    critical_recall: float = float("nan")
    # 回归字段
    rmse: float = float("nan")
    asym: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "metric": self.metric_name,
            "value": self.value,
            "score": self.score,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "classes_present": self.classes_present,
            "classes_missing": self.classes_missing,
            "critical_recall": self.critical_recall,
            "rmse": self.rmse,
            "asym": self.asym,
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_set: np.ndarray | None,
    metric: str = "macro_f1",
    task_type: str = "classification",
    critical_class: int | None = None,
) -> PerfResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return PerfResult(metric, float("nan"), float("nan"), 0)

    if task_type == "regression":
        r = rmse(y_true, y_pred)
        a = asym_score(y_true, y_pred)
        value = a if metric == "asym" else r
        return PerfResult(
            metric_name=metric,
            value=value,
            score=-r,                       # 统一：越大越好
            n_samples=int(y_true.shape[0]),
            rmse=r,
            asym=a,
        )

    # 分类
    present = np.unique(y_true)
    missing = (
        np.setdiff1d(class_set, present) if class_set is not None else np.array([])
    )
    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(
        f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    )
    value = mf1 if metric == "macro_f1" else acc
    crit = float("nan")
    if critical_class is not None:
        mask = y_true == critical_class
        if mask.any():
            crit = float(np.mean(y_pred[mask] == critical_class))
    return PerfResult(
        metric_name=metric,
        value=value,
        score=value,                        # 分类本身越大越好
        n_samples=int(y_true.shape[0]),
        accuracy=acc,
        macro_f1=mf1,
        classes_present=present.tolist(),
        classes_missing=missing.tolist(),
        critical_recall=crit,
    )


def score_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_set,
    metric,
    task_type: str = "classification",
) -> float:
    """门控用的统一单值打分，越大越好。模型未训练或数据为空返回 NaN。"""
    if model is None or X is None or len(y) == 0:
        return float("nan")
    try:
        pred = model.predict(X)
    except RuntimeError:
        return float("nan")
    return evaluate(y, pred, class_set, metric, task_type).score
