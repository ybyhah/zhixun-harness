"""工业模型封装。

选型说明：默认后端用 sklearn 的 HistGradientBoosting，与 LightGBM 同属直方图
梯度提升，CPU 上速度相当，但不依赖 libomp，装包更省事，也更契合"镜像 2GB、
离线、4 核 8G 无 GPU"的评测环境。LightGBM 作为可选后端保留。
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


def decay_weights(batch_ids: np.ndarray, current: int, decay: float) -> np.ndarray:
    """时间衰减权重：越旧的批次权重越低。

    用衰减而不是直接丢弃旧数据，是为了在适应新分布的同时抑制遗忘。
    """
    age = np.maximum(current - np.asarray(batch_ids, dtype=np.float64), 0.0)
    return np.power(float(decay), age)


@dataclass
class TrainBudget:
    """训练预算。

    动机：全量重训时样本会累积到上万条，固定 max_iter 会让单批耗时失控
    （第一阶段实测 19.4s，而评测环境只有 4 核）。按样本量反比缩放迭代数，
    让单次训练的计算量大致恒定。
    """

    enabled: bool = False
    small_n: int = 1200
    min_iter: int = 200
    floor_iter: int = 40
    max_train_samples: int = 6000

    @staticmethod
    def from_config(cfg) -> "TrainBudget":
        raw = cfg.raw.get("budget", {}) if hasattr(cfg, "raw") else {}
        return TrainBudget(
            enabled=bool(raw.get("enabled", False)),
            small_n=int(raw.get("small_n", 1200)),
            min_iter=int(raw.get("min_iter", 200)),
            floor_iter=int(raw.get("floor_iter", 40)),
            max_train_samples=int(raw.get("max_train_samples", 6000)),
        )


def adaptive_params(base: dict, n_samples: int, budget: TrainBudget) -> dict:
    """按样本量自适应调整迭代数。"""
    p = dict(base)
    if not budget.enabled or n_samples <= 0:
        return p
    ceiling = int(p.get("max_iter", budget.min_iter))
    if n_samples <= budget.small_n:
        p["max_iter"] = min(ceiling, budget.min_iter)
    else:
        scaled = int(budget.min_iter * budget.small_n / n_samples)
        p["max_iter"] = max(budget.floor_iter, min(ceiling, scaled))
    return p


def subsample(
    y: np.ndarray, limit: int, seed: int = 42, stratified: bool = True
) -> np.ndarray:
    """下采样的行索引。

    stratified=True（分类）：按类别分层，每类至少保留 1 条，避免稀有类被抽空。
    stratified=False（回归）：直接随机抽样。
    """
    n = int(y.shape[0])
    if limit <= 0 or n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if not stratified:
        return np.sort(rng.choice(n, size=limit, replace=False))
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum((counts / counts.sum() * limit).astype(int), 1)
    picked: list[int] = []
    for cls, q in zip(classes, quota):
        idx = np.flatnonzero(y == cls)
        picked.extend(rng.choice(idx, size=min(int(q), idx.size), replace=False).tolist())
    return np.sort(np.asarray(picked, dtype=int))


@dataclass
class TrainStats:
    n_samples: int = 0
    n_features: int = 0
    seconds: float = 0.0
    classes_seen: int = 0
    n_samples_available: int = 0   # 下采样前的样本量
    max_iter_used: int = 0


@dataclass
class IndustrialModel:
    """工业模型，支持分类与回归。

    分类：类别集合全局固定，避免输出维度随批次跳变。
    回归：class_set 传 None，predict 返回连续值，禁用 proba/校准/类别偏置。
    """

    class_set: np.ndarray | None
    backend: str = "hist"
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    task_type: str = "classification"

    scaler: Any = None          # 训练时的标准化参数快照，随模型一起走
    feature_mask: Any = None    # FEATURE_SUBSET 保留的列索引，推理时自动套用
    _est: Any = field(default=None, repr=False)
    _calibrated: bool = False
    _class_bias: np.ndarray | None = field(default=None, repr=False)
    budget: TrainBudget = field(default_factory=TrainBudget)
    stats: TrainStats = field(default_factory=TrainStats)

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def _prep(self, X: np.ndarray) -> np.ndarray:
        """统一入口：先套用特征子集，再做标准化。训练与推理共用，保证一致。"""
        if self.feature_mask is not None:
            X = X[:, self.feature_mask]
        return self.scaler.transform(X) if self.scaler is not None else X

    # ---- 构建 ----
    def _make_estimator(self, params: dict):
        if self.is_regression:
            if self.backend == "lgbm":
                from lightgbm import LGBMRegressor

                return LGBMRegressor(random_state=self.seed, verbose=-1, **params)
            return HistGradientBoostingRegressor(random_state=self.seed, **params)
        if self.backend == "lgbm":
            from lightgbm import LGBMClassifier  # 延迟导入，未装也不影响默认后端

            return LGBMClassifier(random_state=self.seed, verbose=-1, **params)
        return HistGradientBoostingClassifier(random_state=self.seed, **params)

    @property
    def fitted(self) -> bool:
        return self._est is not None

    # ---- 训练 ----
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> TrainStats:
        t0 = time.perf_counter()
        n_available = int(X.shape[0])

        if self.budget.enabled:
            # 回归无类别概念，随机下采样；分类分层下采样
            sel = subsample(
                y, self.budget.max_train_samples, self.seed,
                stratified=not self.is_regression,
            )
        else:
            sel = np.arange(n_available)
        if sel.shape[0] != n_available:
            X, y = X[sel], y[sel]
            if sample_weight is not None:
                sample_weight = sample_weight[sel]

        X = self._prep(X)
        params = adaptive_params(self.params, int(X.shape[0]), self.budget)
        est = self._make_estimator(params)
        est.fit(X, y, sample_weight=sample_weight)
        self._est = est
        self._calibrated = False
        self._class_bias = None
        self.stats = TrainStats(
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            seconds=time.perf_counter() - t0,
            classes_seen=0 if self.is_regression else int(np.unique(y).size),
            n_samples_available=n_available,
            max_iter_used=int(params.get("max_iter", 0)),
        )
        return self.stats

    def calibrate(self, X: np.ndarray, y: np.ndarray, method: str = "sigmoid") -> None:
        """概率校准。成本远低于重训，适合只有打分偏移的情形。仅分类可用。"""
        if self.is_regression:
            raise RuntimeError("回归模型不支持概率校准")
        if not self.fitted:
            raise RuntimeError("模型未训练，无法校准")
        try:
            # sklearn >= 1.6：冻结已训练模型后再校准
            from sklearn.frozen import FrozenEstimator

            cal = CalibratedClassifierCV(FrozenEstimator(self._est), method=method)
        except ImportError:
            cal = CalibratedClassifierCV(self._est, method=method, cv="prefit")
        cal.fit(self._prep(X), y)
        self._est = cal
        self._calibrated = True

    def set_class_bias(self, bias: np.ndarray | None) -> None:
        """类别偏置（对数空间），用于标签漂移时调整决策倾向。仅分类可用。"""
        if bias is None:
            self._class_bias = None
            return
        if self.is_regression:
            raise RuntimeError("回归模型不支持类别偏置")
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != self.class_set.shape:
            raise ValueError("偏置长度与全局类别数不一致")
        self._class_bias = bias

    # ---- 推理 ----
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回按全局类别集合对齐的概率矩阵，未见类别列为 0。仅分类可用。"""
        if self.is_regression:
            raise RuntimeError("回归模型没有类别概率")
        if not self.fitted:
            raise RuntimeError("模型未训练")
        Xt = self._prep(X)
        raw = self._est.predict_proba(Xt)
        out = np.zeros((Xt.shape[0], self.class_set.size), dtype=np.float64)
        pos = {int(c): i for i, c in enumerate(self.class_set)}
        for j, cls in enumerate(self._est.classes_):
            out[:, pos[int(cls)]] = raw[:, j]
        if self._class_bias is not None:
            out = out * np.exp(self._class_bias)[None, :]
        total = out.sum(axis=1, keepdims=True)
        total[total < 1e-12] = 1.0
        return out / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("模型未训练")
        if self.is_regression:
            return np.asarray(self._est.predict(self._prep(X)), dtype=np.float64)
        proba = self.predict_proba(X)
        return self.class_set[np.argmax(proba, axis=1)]

    # ---- 持久化 ----
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> IndustrialModel:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
