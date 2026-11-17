"""A2 轴承退化识别任务适配器：XJTU-SY 滚动轴承加速寿命数据。

输入是 scripts/extract_xjtu_features.py 产出的特征缓存（每分钟一行，40 维）。
标签：0=Health / 1=Degrading / 2=Severe。退化起点 FPT 用水平通道 RMS 的
3σ 连续超限确定，找不到时回退到固定寿命分数。标签用到整条寿命，这是离线构造
真值，对应评测流"每批自带标签"，预测时并不可见。

批次流：
  - condition：35Hz → 37.5Hz → 40Hz，每工况内一个轴承一批（工况切换漂移）
  - within：只取 single_condition 一个工况，一个轴承一批（弱漂移对照）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Batch, TaskType

CLASS_SET = np.array([0, 1, 2])


def find_fpt(rms: np.ndarray, base_frac: float = 0.1, k_sigma: float = 3.0,
             consecutive: int = 3, min_base: int = 10) -> int | None:
    """RMS 连续 consecutive 点超过基线段 mean + k·std 的首个下标；找不到返回 None。"""
    n = len(rms)
    n_base = max(min_base, int(n * base_frac))
    if n <= n_base + consecutive:
        return None
    base = rms[:n_base]
    over = rms > base.mean() + k_sigma * base.std()
    for i in range(n_base, n - consecutive + 1):
        if over[i:i + consecutive].all():
            return i
    return None


def label_bearing(rms: np.ndarray, life_frac: np.ndarray, severe_frac: float = 0.9,
                  fallback_fpt: float = 0.6, severe_after_fpt: bool = True,
                  **fpt_kwargs) -> tuple[np.ndarray, float, str]:
    """按时间顺序的单个轴承样本打标签。返回 (标签, FPT 寿命分数, FPT 来源)。

    severe_after_fpt=True 时，Severe 起点取 max(severe_frac, FPT)。
    实测 2_1、3_1、3_3、3_4 属于突发失效：FPT 在寿命 92% 之后，寿命 90% 处 RMS
    仍与健康段相同。若按纯寿命分数标 Severe，这一段"看起来健康却标严重"，
    属于不可学的标签噪声。设为 False 则退回纯寿命分数口径（FPT 截断到 severe_frac）。
    """
    i = find_fpt(rms, **fpt_kwargs)
    if i is None:
        fpt_frac, source = fallback_fpt, "fallback"
    else:
        fpt_frac, source = float(life_frac[i]), "changepoint"
    if severe_after_fpt:
        severe_start = max(severe_frac, fpt_frac)
    else:
        fpt_frac = min(fpt_frac, severe_frac)
        severe_start = severe_frac
    y = np.where(life_frac >= severe_start, 2, np.where(life_frac >= fpt_frac, 1, 0))
    return y.astype(int), fpt_frac, source


class XJTUAdapter:
    """实现 TaskAdapter 协议（分类）。"""

    def __init__(self, cache_path: str | Path, stream_mode: str = "condition",
                 single_condition: int = 0, severe_frac: float = 0.9,
                 fallback_fpt: float = 0.6, severe_after_fpt: bool = True,
                 fpt_kwargs: dict | None = None) -> None:
        if stream_mode not in ("condition", "within"):
            raise ValueError(f"未知的 stream_mode: {stream_mode}")
        self.cache_path = Path(cache_path)
        self.stream_mode = stream_mode
        self.single_condition = single_condition
        self.severe_frac = severe_frac
        self.fallback_fpt = fallback_fpt
        self.severe_after_fpt = severe_after_fpt
        self.fpt_kwargs = dict(fpt_kwargs or {})
        self.label_info: dict[str, dict] = {}
        self.batch_bearings: list[str] = []

    @property
    def task_type(self) -> TaskType:
        return "classification"

    @property
    def metric_name(self) -> str:
        return "macro_f1"

    @property
    def class_set(self) -> np.ndarray:
        return CLASS_SET

    def load_batches(self) -> list[Batch]:
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"特征缓存不存在：{self.cache_path}。"
                "请先运行 python3 scripts/extract_xjtu_features.py"
            )
        d = np.load(self.cache_path, allow_pickle=False)
        X, bearing, cond = d["X"].astype(np.float64), d["bearing"], d["condition"]
        minute, life = d["minute"], d["life"]
        rms_col = list(d["feature_names"]).index("h_rms")

        conds = [self.single_condition] if self.stream_mode == "within" else [0, 1, 2]
        batches: list[Batch] = []
        self.batch_bearings = []
        for c in conds:
            for b in sorted(np.unique(bearing[cond == c])):
                m = np.where(bearing == b)[0]
                m = m[np.argsort(minute[m])]
                frac = minute[m] / float(life[m][0])
                y, fpt, src = label_bearing(
                    X[m, rms_col], frac, self.severe_frac, self.fallback_fpt,
                    self.severe_after_fpt, **self.fpt_kwargs,
                )
                self.label_info[str(b)] = {
                    "fpt_frac": fpt, "source": src, "n": int(len(m)),
                    "counts": np.bincount(y, minlength=3).tolist(),
                }
                batches.append(Batch(index=len(batches), X=X[m], y=y))
                self.batch_bearings.append(str(b))
        return batches
