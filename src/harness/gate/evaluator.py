"""门控：候选模型要通过三道检查才能上线。

1. 优于当前在线模型，且提升超过 min_gain（防止噪声导致的无意义切换）
2. 不低于静态基线（对应赛题排名门槛）
3. 在历史工况回放集上不显著退化（对应"稳定性与抗遗忘"）

任一不过就丢弃候选，保留在线模型；连续失败达到上限则回滚历史最优。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..monitor.performance import evaluate, score_model
from .replay import ReplayBuffer


@dataclass
class GateResult:
    accepted: bool
    reason: str
    candidate_score: float = float("nan")
    online_score: float = float("nan")
    baseline_score: float = float("nan")
    candidate_replay: float = float("nan")
    online_replay: float = float("nan")
    candidate_critical_recall: float = float("nan")
    online_critical_recall: float = float("nan")
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "candidate_score": self.candidate_score,
            "online_score": self.online_score,
            "baseline_score": self.baseline_score,
            "candidate_replay": self.candidate_replay,
            "online_replay": self.online_replay,
            "candidate_critical_recall": self.candidate_critical_recall,
            "online_critical_recall": self.online_critical_recall,
            "checks": dict(self.checks),
        }


class Gate:
    def __init__(
        self,
        class_set: np.ndarray,
        metric: str = "macro_f1",
        min_gain: float = 0.005,
        forget_tol: float = 0.02,
        task_type: str = "classification",
        critical_class: int | None = None,
        critical_recall_tol: float | None = None,
    ) -> None:
        self.class_set = class_set
        self.metric = metric
        self.min_gain = min_gain
        self.forget_tol = forget_tol
        self.task_type = task_type
        self.critical_class = critical_class
        self.critical_recall_tol = critical_recall_tol

    def _critical_recall(self, model, X, y) -> float:
        """模型在 (X,y) 上对关键类的召回；无模型/预测失败返回 NaN。"""
        if model is None or X is None or len(y) == 0:
            return float("nan")
        try:
            pred = model.predict(X)
        except RuntimeError:
            return float("nan")
        return evaluate(
            y, pred, self.class_set, self.metric, self.task_type, self.critical_class
        ).critical_recall

    def evaluate(
        self,
        candidate,
        online,
        baseline,
        recent: tuple[np.ndarray, np.ndarray],
        replay: ReplayBuffer,
    ) -> GateResult:
        Xr, yr = recent
        s_cand = score_model(candidate, Xr, yr, self.class_set, self.metric, self.task_type)
        s_online = score_model(online, Xr, yr, self.class_set, self.metric, self.task_type)
        s_base = score_model(baseline, Xr, yr, self.class_set, self.metric, self.task_type)

        res = GateResult(
            accepted=False,
            reason="",
            candidate_score=s_cand,
            online_score=s_online,
            baseline_score=s_base,
        )

        if candidate is None or s_cand != s_cand:
            res.reason = "候选模型缺失或无法评分"
            res.checks = {"beats_online": False}
            return res

        # 检查一：优于在线模型
        if s_online == s_online:
            gain = s_cand - s_online
            ok = gain >= self.min_gain
            res.checks["beats_online"] = ok
            if not ok:
                res.reason = (
                    f"提升 {gain:+.4f} 未达阈值 {self.min_gain:.4f}，不切换"
                )
                return res
        else:
            res.checks["beats_online"] = True  # 无在线模型，视为通过

        # 检查二：不低于静态基线
        if s_base == s_base:
            ok = s_cand >= s_base
            res.checks["not_below_baseline"] = ok
            if not ok:
                res.reason = f"候选 {s_cand:.4f} 低于静态基线 {s_base:.4f}"
                return res
        else:
            res.checks["not_below_baseline"] = True

        # 检查三：回放集不退化
        if not replay.empty:
            Xp, yp, _ = replay.dataset()
            if Xp.shape[0] > 0 and Xp.shape[1] == Xr.shape[1]:
                r_cand = score_model(candidate, Xp, yp, self.class_set, self.metric, self.task_type)
                r_online = score_model(online, Xp, yp, self.class_set, self.metric, self.task_type)
                res.candidate_replay = r_cand
                res.online_replay = r_online
                if r_cand == r_cand and r_online == r_online:
                    ok = r_cand >= r_online - self.forget_tol
                    res.checks["no_forgetting"] = ok
                    if not ok:
                        res.reason = (
                            f"回放集退化 {r_cand - r_online:+.4f}，超出容忍 "
                            f"{self.forget_tol:.4f}，疑似遗忘"
                        )
                        return res
                else:
                    res.checks["no_forgetting"] = True
            else:
                res.checks["no_forgetting"] = True
        else:
            res.checks["no_forgetting"] = True  # 首批无回放集，跳过

        # 检查四（可选）：关键类召回不退化（A2 严重退化漏报）
        if self.critical_class is not None and self.critical_recall_tol is not None:
            r_cand = self._critical_recall(candidate, Xr, yr)
            r_online = self._critical_recall(online, Xr, yr)
            res.candidate_critical_recall = r_cand
            res.online_critical_recall = r_online
            if r_cand == r_cand and r_online == r_online:
                ok = r_cand >= r_online - self.critical_recall_tol
                res.checks["critical_recall"] = ok
                if not ok:
                    res.reason = (
                        f"关键类召回退化 {r_cand - r_online:+.4f}，超出容忍 "
                        f"{self.critical_recall_tol:.4f}（严重退化漏报）"
                    )
                    return res
            else:
                res.checks["critical_recall"] = True  # 验证集缺关键类或无在线，跳过

        res.accepted = True
        res.reason = (
            f"三道门控通过：近期 {s_cand:.4f} vs 在线 {s_online:.4f}，"
            f"回放 {res.candidate_replay:.4f} vs {res.online_replay:.4f}"
        )
        return res
