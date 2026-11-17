"""动作空间与执行器。

按成本从低到高排列，决策器优先选便宜的动作。
"NOOP" 是一等公民：赛题明确"不更新也可以是正确的系统决策"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..models.industrial import IndustrialModel, decay_weights


class Action(str, Enum):
    NOOP = "NOOP"
    ADJUST_THRESHOLD = "ADJUST_THRESHOLD"
    RECALIBRATE = "RECALIBRATE"
    INCREMENTAL = "INCREMENTAL"
    WINDOW_RETRAIN = "WINDOW_RETRAIN"
    FULL_RETRAIN = "FULL_RETRAIN"
    FEATURE_SUBSET = "FEATURE_SUBSET"
    OFFLINE = "OFFLINE"


# 相对成本，用于资源效率统计与决策排序
ACTION_COST = {
    Action.NOOP: 0,
    Action.ADJUST_THRESHOLD: 1,
    Action.RECALIBRATE: 2,
    Action.INCREMENTAL: 3,
    Action.WINDOW_RETRAIN: 4,
    Action.FULL_RETRAIN: 5,
    Action.FEATURE_SUBSET: 4,
    Action.OFFLINE: 0,
}


@dataclass
class Decision:
    action: Action
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    hypothesis: str | None = None   # 规则版为 None；大模型版填写

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "params": self.params,
            "reason": self.reason,
            "hypothesis": self.hypothesis,
            "cost": ACTION_COST[self.action],
        }


@runtime_checkable
class Decider(Protocol):
    """规则决策器与第二阶段的大模型决策器实现同一协议。"""

    def decide(self, report, history: list) -> Decision: ...


@dataclass
class TrainingData:
    """历史数据池，按批次编号保存，供窗口与衰减训练使用。"""

    X: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)
    batch_ids: list[int] = field(default_factory=list)

    def add(self, X: np.ndarray, y: np.ndarray, batch_index: int) -> None:
        self.X.append(np.asarray(X))
        self.y.append(np.asarray(y))
        self.batch_ids.append(batch_index)

    @property
    def n_batches(self) -> int:
        return len(self.X)

    def window(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """最近 k 个批次。"""
        k = max(1, min(k, self.n_batches))
        Xs, ys, ids = self.X[-k:], self.y[-k:], self.batch_ids[-k:]
        return self._stack(Xs, ys, ids)

    def window_for_samples(self, k_pref: int, min_samples: int) -> int:
        """从 k_pref 开始向前扩展窗口，直到凑够 min_samples 或用完全部批次。

        固定批次数的窗口在早期会样本不足（第一阶段批次 2 的更新因此是负收益）。
        """
        k = max(1, min(k_pref, self.n_batches))
        if min_samples <= 0:
            return k
        while k < self.n_batches and self.window(k)[0].shape[0] < min_samples:
            k += 1
        return k

    def all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._stack(self.X, self.y, self.batch_ids)

    def split_last(
        self, val_frac: float = 0.3, seed: int = 42, val_mode: str = "in_batch"
    ) -> tuple["TrainingData", np.ndarray, np.ndarray]:
        """切出门控验证集。返回 (剔除验证部分后的数据池, val_X, val_y)。

        门控必须在候选"没训练过"的数据上比较。此前门控用的是候选刚训练过的
        窗口，分数虚高到 1.0，三道检查形同虚设。

        val_mode：
          - in_batch（默认）：从最近批次按类别分层切 val_frac 作验证，其余连同
            更早批次进训练池。验证集与训练数据同分布，门控分偏乐观。
          - holdout_recent：训练池为"截至上一批"的全部历史，验证集为整个最近
            批次。候选完全没见过验证集，且验证集是手上最新、最可能已漂移的数据，
            是对"下一批"更接近的代理（walk-forward 时序验证）。只有 1 个批次时
            无"更早批次"可训练，回退 in_batch。
          - holdout_refit：切分口径同 holdout_recent（probe 诚实门控）；上线前
            由闭环对选中动作在全窗口重训，兼顾诚实门控与最新批适应。此处切分
            逻辑与 holdout_recent 完全一致，重训在 loop 层完成。
        """
        if self.n_batches == 0:
            return self, np.empty((0, 0)), np.empty((0,), dtype=int)

        if val_mode not in ("in_batch", "holdout_recent", "holdout_refit"):
            raise ValueError(f"未知的 val_mode: {val_mode}")

        if val_mode in ("holdout_recent", "holdout_refit") and self.n_batches >= 2:
            trimmed = TrainingData(
                X=list(self.X[:-1]), y=list(self.y[:-1]),
                batch_ids=list(self.batch_ids[:-1]),
            )
            return trimmed, self.X[-1], self.y[-1]

        X_last, y_last = self.X[-1], self.y[-1]
        rng = np.random.default_rng(seed)
        val_idx: list[int] = []
        for cls in np.unique(y_last):
            idx = np.flatnonzero(y_last == cls)
            k = min(max(int(round(idx.size * val_frac)), 1), max(idx.size - 1, 1))
            val_idx.extend(rng.choice(idx, size=k, replace=False).tolist())
        val_mask = np.zeros(y_last.size, dtype=bool)
        val_mask[np.asarray(val_idx, dtype=int)] = True

        trimmed = TrainingData(
            X=list(self.X[:-1]) + [X_last[~val_mask]],
            y=list(self.y[:-1]) + [y_last[~val_mask]],
            batch_ids=list(self.batch_ids),
        )
        return trimmed, X_last[val_mask], y_last[val_mask]

    @staticmethod
    def _stack(Xs, ys, ids):
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        bid = np.concatenate([np.full(len(a), i) for a, i in zip(ys, ids)])
        return X, y, bid


class ActionExecutor:
    """执行动作，产出候选模型。不负责决定是否上线（那是门控的事）。"""

    def __init__(
        self,
        class_set: np.ndarray,
        backend: str,
        params: dict,
        seed: int,
        budget=None,
        min_window_samples: int = 0,
        scaler_factory=None,
        task_type: str = "classification",
    ):
        self.class_set = class_set
        self.backend = backend
        self.params = params
        self.seed = seed
        self.budget = budget
        self.min_window_samples = min_window_samples
        # 每个候选模型都带一份"训练时点"的标准化参数快照
        self.scaler_factory = scaler_factory
        self.task_type = task_type
        self.train_calls = 0

    def _new_model(self) -> IndustrialModel:
        from ..models.industrial import TrainBudget

        return IndustrialModel(
            class_set=self.class_set,
            backend=self.backend,
            params=dict(self.params),
            seed=self.seed,
            task_type=self.task_type,
            budget=self.budget or TrainBudget(),
            scaler=self.scaler_factory() if self.scaler_factory else None,
        )

    def _window(self, pool: "TrainingData", k_pref: int) -> int:
        """按最小样本量扩展窗口，避免早期批次少时训出过弱的候选。"""
        return pool.window_for_samples(k_pref, self.min_window_samples)

    def execute(
        self,
        decision: Decision,
        pool: TrainingData,
        online: IndustrialModel | None,
        current_batch: int,
        decay: float,
    ) -> IndustrialModel | None:
        """返回候选模型；NOOP 与 OFFLINE 返回 None。"""
        act = decision.action
        if act in (Action.NOOP, Action.OFFLINE):
            return None

        if act == Action.ADJUST_THRESHOLD:
            if online is None:
                return None
            model = self._clone_online(online)
            X, y, _ = pool.window(1)
            model.set_class_bias(self._prior_bias(y))
            return model

        if act == Action.RECALIBRATE:
            if online is None:
                return None
            model = self._clone_online(online)
            X, y, _ = pool.window(int(decision.params.get("window", 2)))
            try:
                model.calibrate(X, y)
            except Exception:
                return None
            return model

        if act == Action.INCREMENTAL:
            k = self._window(pool, int(decision.params.get("window", 2)))
            X, y, bid = pool.window(k)
            return self._fit(X, y, bid, current_batch, decay)

        if act == Action.WINDOW_RETRAIN:
            k = self._window(pool, int(decision.params.get("window", 3)))
            X, y, bid = pool.window(k)
            return self._fit(X, y, bid, current_batch, decay)

        if act == Action.FULL_RETRAIN:
            X, y, bid = pool.all()
            d = float(decision.params.get("decay", decay))
            return self._fit(X, y, bid, current_batch, d)

        if act == Action.FEATURE_SUBSET:
            drop = list(decision.params.get("drop_features", []))
            k = self._window(pool, int(decision.params.get("window", 3)))
            X, y, bid = pool.window(k)
            if not drop:
                return self._fit(X, y, bid, current_batch, decay)
            if len(np.unique(y)) < 2 and self.task_type != "regression":
                return None
            keep = np.setdiff1d(np.arange(X.shape[1]), np.asarray(drop, dtype=int))
            model = self._new_model()
            # 特征子集由模型内部统一套用：设 feature_mask + 裁剪 scaler 到保留列，
            # fit 与 predict 都走 _prep，自动对齐，调用方始终传全维特征。
            model.feature_mask = keep
            if model.scaler is not None:
                model.scaler = model.scaler.subset(keep)
            w = decay_weights(bid, current_batch, decay)
            model.fit(X, y, sample_weight=w)
            self.train_calls += 1
            return model

        return None

    # ---- 内部 ----
    def _fit(self, X, y, bid, current_batch, decay) -> IndustrialModel | None:
        if len(np.unique(y)) < 2:
            return None          # 单类无法训练
        model = self._new_model()
        w = decay_weights(bid, current_batch, decay)
        model.fit(X, y, sample_weight=w)
        self.train_calls += 1
        return model

    def _clone_online(self, online: IndustrialModel) -> IndustrialModel:
        import copy

        return copy.deepcopy(online)

    def _prior_bias(self, y: np.ndarray) -> np.ndarray:
        """按最近批次的类别占比生成对数偏置。"""
        counts = np.array([(y == c).sum() for c in self.class_set], dtype=float)
        prior = (counts + 1.0) / (counts.sum() + len(self.class_set))
        return np.log(prior) - np.log(1.0 / len(self.class_set))
