"""分布漂移信号：PSI、KS 检验、PCA 重构误差。

PSI 自己实现（几十行，可控且无额外依赖），KS 用 scipy，
多变量漂移用 PCA 重构误差——都只在参考窗口上拟合，不碰未来数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

_EPS = 1e-6


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index。分箱按参考分布的分位数切，零频用 EPS 兜底。"""
    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    qs = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, qs))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    r_frac = np.maximum(r / max(r.sum(), 1), _EPS)
    c_frac = np.maximum(c / max(c.sum(), 1), _EPS)
    return float(np.sum((c_frac - r_frac) * np.log(c_frac / r_frac)))


@dataclass
class DistResult:
    psi_mean: float
    psi_max: float
    psi_alarm_ratio: float            # 超阈值特征占比
    psi_top_features: list[int] = field(default_factory=list)
    ks_alarm_ratio: float = 0.0
    pca_ratio: float = 1.0            # 当前重构误差 / 参考重构误差
    drifted: bool = False

    def to_dict(self) -> dict:
        return {
            "psi_mean": self.psi_mean,
            "psi_max": self.psi_max,
            "psi_alarm_ratio": self.psi_alarm_ratio,
            "psi_top_features": self.psi_top_features,
            "ks_alarm_ratio": self.ks_alarm_ratio,
            "pca_ratio": self.pca_ratio,
            "drifted": self.drifted,
        }


class DistributionMonitor:
    """参考窗口由已见数据构成，可随模型更新一起刷新。"""

    def __init__(
        self,
        psi_bins: int = 10,
        psi_threshold: float = 0.25,
        psi_feature_ratio: float = 0.15,
        ks_alpha: float = 0.05,
        pca_components: int = 8,
        pca_threshold: float = 2.0,
    ) -> None:
        self.psi_bins = psi_bins
        self.psi_threshold = psi_threshold
        self.psi_feature_ratio = psi_feature_ratio
        self.ks_alpha = ks_alpha
        self.pca_components = pca_components
        self.pca_threshold = pca_threshold
        self._ref: np.ndarray | None = None
        self._pca: PCA | None = None
        self._ref_recon: float = float("nan")

    @property
    def has_reference(self) -> bool:
        return self._ref is not None

    def set_reference(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        self._ref = X
        n_comp = int(min(self.pca_components, X.shape[1], max(X.shape[0] - 1, 1)))
        if n_comp >= 1 and X.shape[0] > n_comp:
            self._pca = PCA(n_components=n_comp, random_state=0).fit(X)
            self._ref_recon = self._recon_error(X)
        else:
            self._pca = None
            self._ref_recon = float("nan")

    def _recon_error(self, X: np.ndarray) -> float:
        if self._pca is None:
            return float("nan")
        back = self._pca.inverse_transform(self._pca.transform(X))
        return float(np.mean(np.sum((X - back) ** 2, axis=1)))

    def compare(self, X: np.ndarray) -> DistResult:
        if self._ref is None:
            return DistResult(0.0, 0.0, 0.0, [], 0.0, 1.0, False)
        X = np.asarray(X, dtype=np.float64)
        n_feat = self._ref.shape[1]

        psis = np.array(
            [psi(self._ref[:, j], X[:, j], self.psi_bins) for j in range(n_feat)]
        )
        alarm = psis > self.psi_threshold
        top = np.argsort(psis)[::-1][:5].tolist()

        ks_alarm = 0
        for j in range(n_feat):
            if ks_2samp(self._ref[:, j], X[:, j]).pvalue < self.ks_alpha:
                ks_alarm += 1

        pca_ratio = 1.0
        if self._pca is not None and self._ref_recon == self._ref_recon:
            cur = self._recon_error(X)
            pca_ratio = float(cur / max(self._ref_recon, 1e-12))

        alarm_ratio = float(alarm.mean())
        drifted = bool(
            alarm_ratio > self.psi_feature_ratio or pca_ratio > self.pca_threshold
        )
        return DistResult(
            psi_mean=float(psis.mean()),
            psi_max=float(psis.max()),
            psi_alarm_ratio=alarm_ratio,
            psi_top_features=[int(i) for i in top],
            ks_alarm_ratio=float(ks_alarm / n_feat),
            pca_ratio=pca_ratio,
            drifted=drifted,
        )


def label_shift(ref_y: np.ndarray, cur_y: np.ndarray, class_set: np.ndarray) -> float:
    """类别占比的 L1 距离，衡量 P(Y) 变化。"""
    def mix(y):
        counts = np.array([(np.asarray(y) == c).sum() for c in class_set], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts
    return float(np.abs(mix(cur_y) - mix(ref_y)).sum())


def target_shift(ref_y: np.ndarray, cur_y: np.ndarray) -> float:
    """回归目标值分布漂移，用 KS 统计量（0~1，越大漂移越强）。"""
    ref_y = np.asarray(ref_y, dtype=np.float64)
    cur_y = np.asarray(cur_y, dtype=np.float64)
    if ref_y.size == 0 or cur_y.size == 0:
        return 0.0
    return float(ks_2samp(ref_y, cur_y).statistic)
