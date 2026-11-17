"""客观分四项指标的统一实现。

口径写死在这里并有测试覆盖，E0 到 E4 全部共用，避免不同实验用不同算法
导致结论不可比。四项对应赛题客观分：漂移综合性能 15、最差阶段与恢复速度 10、
稳定性与抗遗忘 5、资源效率 5。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

NAN = float("nan")


def _clean(scores: list[float]) -> list[float]:
    return [s for s in scores if s == s]


def composite(scores: list[float]) -> float:
    """漂移综合性能：各批次指标均值。冷启动批次的 NaN 自动跳过。"""
    vals = _clean(scores)
    return float(np.mean(vals)) if vals else NAN


def worst_stage(scores: list[float], window: int = 2) -> float:
    """最差阶段：滑动窗口均值的最小值。

    用窗口而非单批最小值，是因为单批容易被样本量小的批次带偏，
    而"最差阶段"关心的是持续一段时间的低谷。
    """
    vals = _clean(scores)
    if not vals:
        return NAN
    w = max(1, min(window, len(vals)))
    means = [float(np.mean(vals[i : i + w])) for i in range(len(vals) - w + 1)]
    return float(np.min(means))


def recovery(
    scores: list[float], tol: float = 0.05, penalty: int | None = None
) -> tuple[list[int], float]:
    """恢复速度：跌破"此前最佳 - tol"后用了几个批次回到该水平。

    未恢复的事件记为惩罚值（默认总批次数）而不是丢弃，
    否则"一直没救回来"反而会让均值看起来更好。
    """
    vals = _clean(scores)
    pen = penalty if penalty is not None else max(len(vals), 1)
    events: list[int] = []
    best = float("-inf")
    pending: int | None = None
    for i, s in enumerate(vals):
        if pending is not None:
            if s >= best - tol:
                events.append(i - pending)
                pending = None
        elif s < best - tol:
            pending = i
        best = max(best, s)
    if pending is not None:
        events.append(-1)
    scored = [pen if e < 0 else e for e in events]
    return events, (float(np.mean(scored)) if scored else 0.0)


def forgetting(replay_pairs: list[tuple[float, float]]) -> float:
    """稳定性与抗遗忘：每次上线前后在回放集上的分数差，取最差一次。

    入参是 (上线前分数, 上线后分数) 列表。返回值 <= 0 表示有退化，
    越接近 0 越好。无可用数据返回 NaN。
    """
    deltas = [
        after - before
        for before, after in replay_pairs
        if before == before and after == after
    ]
    return float(np.min(deltas)) if deltas else NAN


def resource_efficiency(
    gain: float, trainings: int, seconds: float
) -> tuple[float, float]:
    """资源效率：单位性能增益消耗的训练次数与秒数。

    增益为零或负时返回 NaN —— 没有增益谈效率没有意义。
    """
    if gain != gain or gain <= 0:
        return NAN, NAN
    return float(trainings / gain), float(seconds / gain)


@dataclass
class MetricSummary:
    composite: float = NAN
    worst_stage: float = NAN
    recovery_events: list[int] = field(default_factory=list)
    recovery_mean: float = NAN
    forgetting: float = NAN
    trainings: int = 0
    seconds: float = NAN
    gain_vs_baseline: float = NAN
    trainings_per_gain: float = NAN
    seconds_per_gain: float = NAN
    promotions: int = 0
    offline_batches: int = 0
    rollbacks: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def recovery_text(self) -> str:
        if not self.recovery_events:
            return "无跌落"
        return ",".join("未恢复" if e < 0 else str(e) for e in self.recovery_events)


def summarize(
    scores: list[float],
    baseline_composite: float = NAN,
    trainings: int = 0,
    seconds: float = NAN,
    replay_pairs: list[tuple[float, float]] | None = None,
    tol: float = 0.05,
    worst_window: int = 2,
    promotions: int = 0,
    offline_batches: int = 0,
    rollbacks: int = 0,
) -> MetricSummary:
    comp = composite(scores)
    events, rec_mean = recovery(scores, tol=tol)
    gain = (
        comp - baseline_composite
        if comp == comp and baseline_composite == baseline_composite
        else NAN
    )
    t_per, s_per = resource_efficiency(gain, trainings, seconds)
    return MetricSummary(
        composite=comp,
        worst_stage=worst_stage(scores, worst_window),
        recovery_events=events,
        recovery_mean=rec_mean,
        forgetting=forgetting(replay_pairs or []),
        trainings=trainings,
        seconds=seconds,
        gain_vs_baseline=gain,
        trainings_per_gain=t_per,
        seconds_per_gain=s_per,
        promotions=promotions,
        offline_batches=offline_batches,
        rollbacks=rollbacks,
    )
