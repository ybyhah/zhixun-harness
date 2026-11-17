"""上线门控：官方验证指标较当前最优提升达阈值，且不低于官方基线，才采纳。

门控对象是官方续训产出的 checkpoint；分数用统一口径（越大越好）。
未通过则不上线，下一轮从上一个被采纳的 checkpoint 继续（回滚语义）。

可选的**统计显著性模式**：用配对 bootstrap 判断"候选是否显著优于当前最优"，
避免在验证集噪声上做上线决策（实测 A 包有 +0.0006 的提升对 ±0.0086 的配对噪声）。
显著性不可计算时自动退回普通数值比较，并在 reason 标注降级——门控绝不中断闭环。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metric import to_score
from .package import OfficialPackage
from .significance import Significance, paired_bootstrap


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    candidate_score: float
    best_score: float
    baseline_score: float
    significance: dict | None = None     # 配对 bootstrap 结果（显著性模式下）
    degraded: bool = False               # 显著性不可用，已退回普通比较
    inconclusive: bool = False           # 统计上无法区分

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason,
                "candidate_score": self.candidate_score,
                "best_score": self.best_score,
                "baseline_score": self.baseline_score,
                "significance": self.significance,
                "degraded": self.degraded,
                "inconclusive": self.inconclusive}


class OfficialGate:
    def __init__(self, pkg: OfficialPackage, baseline_value: float,
                 min_gain: float = 0.0, *, mode: str = "plain",
                 bootstrap: int = 500, p_threshold: float = 0.95) -> None:
        if mode not in ("plain", "significance"):
            raise ValueError(f"未知门控模式 {mode}")
        self.pkg = pkg
        self.baseline_score = to_score(pkg, baseline_value)
        self.min_gain = min_gain
        self.mode = mode
        self.bootstrap = bootstrap
        self.p_threshold = p_threshold

    def evaluate(self, candidate_value: float, best_value: float, *,
                 pred_cand: np.ndarray | None = None,
                 pred_best: np.ndarray | None = None,
                 y: np.ndarray | None = None,
                 seed: int = 42,
                 cand_steps: int | None = None,
                 best_steps: int | None = None) -> GateDecision:
        cand = to_score(self.pkg, candidate_value)
        best = to_score(self.pkg, best_value)
        base = self.baseline_score
        if cand != cand:
            return GateDecision(False, "候选验证指标缺失", cand, best, base)
        # 门槛一：不低于官方基线（C10）。这是阈值而非比较，始终用普通数值判定。
        if base == base and cand < base:
            return GateDecision(False, f"候选 {candidate_value:.4f} 低于官方基线",
                                cand, best, base)

        # 门槛二：优于当前最优
        if best != best:                      # 还没有在线最优 → 直接接受
            return GateDecision(True, "无当前最优，接受首个候选", cand, best, base)

        if self.mode == "plain":
            if cand < best + self.min_gain:
                return GateDecision(
                    False, f"较最优提升 {cand - best:+.4f} 未达阈值 {self.min_gain}",
                    cand, best, base)
            return GateDecision(True, "通过：不低于基线且较最优有提升", cand, best, base)

        # 显著性模式
        sig = None
        if pred_cand is not None and pred_best is not None and y is not None:
            sig = paired_bootstrap(pred_cand, pred_best, y, self.pkg.task,
                                   bootstrap=self.bootstrap, seed=seed)
        if sig is None:                        # 不可用 → 降级普通比较
            ok = cand >= best + self.min_gain
            return GateDecision(
                ok, f"{'通过' if ok else '未达阈值'}（显著性不可用，已退回数值比较）",
                cand, best, base, degraded=True)

        payload = sig.to_dict()
        if sig.p_better >= self.p_threshold:
            return GateDecision(
                True,
                f"显著优于当前最优：P(更好)={sig.p_better:.1%}，"
                f"95%CI [{sig.ci_low:+.4f},{sig.ci_high:+.4f}]",
                cand, best, base, significance=payload)

        # 统计上无法区分 = 差值的 95% 置信区间跨过 0（两个方向都不能排除）。
        # 注意不能用 p_better 居中来判定：两个模型完全相同时 p_better=0 而非 0.5。
        inconclusive = sig.ci_low <= 0.0 <= sig.ci_high
        if (inconclusive and cand_steps is not None and best_steps is not None
                and cand_steps < best_steps):
            return GateDecision(
                True,
                f"统计上无法区分（P={sig.p_better:.1%}），但候选步数更少 "
                f"({cand_steps} < {best_steps})，取更不易过拟合者",
                cand, best, base, significance=payload, inconclusive=True)
        return GateDecision(
            False,
            f"提升不显著：P(更好)={sig.p_better:.1%} < {self.p_threshold:.0%}，"
            f"95%CI [{sig.ci_low:+.4f},{sig.ci_high:+.4f}]",
            cand, best, base, significance=payload, inconclusive=inconclusive)
