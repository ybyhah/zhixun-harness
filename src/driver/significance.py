"""配对 bootstrap：判断候选是否**统计显著地**优于当前最优。

为什么要配对：两个模型在同一批验证样本上评估，差值的方差远小于各自估计的方差。
实测 A 包：单点 bootstrap 标准差 ±0.0249，而配对差值标准差只有 ±0.0086。
用单点方差会把真实提升误判成噪声，用配对才是比较决策的正确口径。

工业级约束：
  - 不重新推理（只在已算好的 pred 上重采样）
  - RNG 由 (seed, round, cand) 确定性派生 → 可复现
  - 任何退化输入/异常都返回 None，由上层降级为普通数值比较，绝不抛错（C1 无人干预）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .official_metric import score


@dataclass
class Significance:
    p_better: float          # 候选更好的 bootstrap 比例
    ci_low: float            # 差值 95% 置信区间下界（正=更好）
    ci_high: float
    mean_diff: float
    bootstrap: int
    seed: int

    def to_dict(self) -> dict:
        return {"p_better": self.p_better, "ci_low": self.ci_low,
                "ci_high": self.ci_high, "mean_diff": self.mean_diff,
                "bootstrap": self.bootstrap, "seed": self.seed}


def derive_seed(base_seed: int, round_index: int, candidate: int = 0) -> int:
    """确定性派生，避免用全局随机。"""
    return int(np.random.SeedSequence([int(base_seed), int(round_index),
                                       int(candidate)]).generate_state(1)[0])


def paired_bootstrap(pred_cand: np.ndarray, pred_best: np.ndarray, y: np.ndarray,
                     task: str, *, bootstrap: int = 500, seed: int = 42,
                     ) -> Significance | None:
    """返回候选相对当前最优的显著性；无法计算时返回 None（上层降级）。"""
    try:
        y = np.asarray(y)
        a, b = np.asarray(pred_cand), np.asarray(pred_best)
        n = y.shape[0]
        if n == 0 or a.shape[0] != n or b.shape[0] != n or bootstrap < 1:
            return None
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return None
        rng = np.random.default_rng(seed)
        diffs = np.empty(bootstrap, dtype=np.float64)
        for i in range(bootstrap):
            idx = rng.integers(0, n, n)
            sa = score(a, y, task, idx)
            sb = score(b, y, task, idx)
            if sa != sa or sb != sb:
                return None
            diffs[i] = sa - sb          # score 已统一为越大越好
        return Significance(
            p_better=float(np.mean(diffs > 0)),
            ci_low=float(np.percentile(diffs, 2.5)),
            ci_high=float(np.percentile(diffs, 97.5)),
            mean_diff=float(np.mean(diffs)),
            bootstrap=int(bootstrap),
            seed=int(seed),
        )
    except Exception:  # noqa: BLE001 —— 门控绝不能因统计计算中断闭环
        return None
