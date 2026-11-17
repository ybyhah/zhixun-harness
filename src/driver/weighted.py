"""改训练损失：子类重写 fit，运行时注入官方 train.py。

只改"每个 batch 的 loss 与梯度"，其余（save/load/predict/validate、Adam、梯度裁剪、
steps/epochs 计数）全部继承官方实现，因此产出的 checkpoint 与官方格式完全一致，
官方未修改的 predict.py 能原样加载。不修改 base/ 里任何官方文件。

自证：weighted_ce 的权重全为 1、asym 的 alpha=1 时，与官方损失数值等价。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .migrate import _load_official_model_cls

LOSSES = ("official", "weighted_ce", "asym", "huber")


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """w_c = N / (K * n_c)，再归一化到均值 1（不改变有效学习率量级）。"""
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes, dtype=np.float64)
    seen = counts > 0
    w[seen] = y.size / (seen.sum() * counts[seen])
    w[~seen] = 0.0
    mean = w[seen].mean() if seen.any() else 1.0
    return w / (mean if mean > 0 else 1.0)


def make_weighted_model(pkg_root: str | Path, loss: str = "official",
                        alpha: float = 1.0, delta: float = 1.0):
    """返回 WeightedModel 类（基类取自官方包内的 Model）。"""
    if loss not in LOSSES:
        raise ValueError(f"未知损失 {loss}，可选 {LOSSES}")
    if alpha <= 0 or delta <= 0:
        raise ValueError("alpha/delta 必须为正")
    Base = _load_official_model_cls(pkg_root)

    class WeightedModel(Base):
        LOSS, ALPHA, DELTA = loss, alpha, delta

        def fit(self, x, y, epochs):
            if type(epochs) is not int or not 1 <= epochs <= 10000:
                raise ValueError("epochs must be between 1 and 10000")
            x = self.normalized(x)
            if y.shape != (len(x),) or not len(x):
                raise ValueError("target shape mismatch")
            classification = self.meta["task"] == "classification"
            if classification:
                if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                    raise ValueError("unknown class")
                target = y
                cw = (class_weights(y, len(self.meta["classes"]))
                      if self.LOSS == "weighted_ce" else None)
            else:
                target = (y - self.meta["target_mean"]) / self.meta["target_scale"]
                cw = None

            epoch_losses = []
            for _ in range(epochs):
                rng = np.random.default_rng(
                    np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
                order = rng.permutation(len(x))
                loss_sum = 0.0
                for start in range(0, len(order), self.meta["batch_size"]):
                    idx = order[start:start + self.meta["batch_size"]]
                    batch, expected = x[idx], target[idx]
                    hidden, logits = self.forward(batch)
                    n = len(idx)
                    if classification:
                        logits -= logits.max(axis=1, keepdims=True)
                        probs = np.exp(logits)
                        probs /= probs.sum(axis=1, keepdims=True)
                        picked = np.maximum(probs[np.arange(n), expected], 1e-300)
                        sw = None if cw is None else cw[expected]
                        if sw is None:
                            loss_sum -= float(np.log(picked).sum())
                        else:
                            loss_sum -= float(np.sum(sw * np.log(picked)))
                        delta_out = probs
                        delta_out[np.arange(n), expected] -= 1
                        # 先除以 batch 再乘权重：权重全为 1 时与官方实现逐位一致
                        delta_out /= n
                        if sw is not None:
                            delta_out *= sw[:, None]
                    else:
                        error = logits[:, 0] - expected
                        if self.LOSS == "asym":
                            sw = np.where(error > 0, self.ALPHA, 1.0)
                            loss_sum += float(np.sum(sw * error ** 2))
                            grad = 2 * sw * error / n
                        elif self.LOSS == "huber":
                            a = np.abs(error)
                            quad = a <= self.DELTA
                            loss_sum += float(np.sum(np.where(
                                quad, error ** 2, 2 * self.DELTA * a - self.DELTA ** 2)))
                            grad = np.where(quad, 2 * error,
                                            2 * self.DELTA * np.sign(error)) / n
                        else:
                            loss_sum += float(np.sum(error ** 2))
                            grad = 2 * error / n
                        delta_out = grad[:, None]
                    gradients = {"w2": hidden.T @ delta_out,
                                 "b2": delta_out.sum(axis=0)}
                    hidden_delta = (delta_out @ self.params["w2"].T) * (1 - hidden ** 2)
                    gradients.update(w1=batch.T @ hidden_delta,
                                     b1=hidden_delta.sum(axis=0))
                    norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                    if not math.isfinite(norm):
                        raise ValueError("nonfinite gradient")
                    self.meta["steps"] += 1
                    for name, gradient in gradients.items():
                        gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                        self.first[name] = .9 * self.first[name] + .1 * gradient
                        self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                        first = self.first[name] / (1 - .9 ** self.meta["steps"])
                        second = self.second[name] / (1 - .999 ** self.meta["steps"])
                        self.params[name] -= (self.meta["learning_rate"] * first
                                              / (np.sqrt(second) + 1e-8))
                    if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                        raise ValueError("nonfinite model state")
                self.meta["epochs"] += 1
                epoch_losses.append(loss_sum / len(x))
            self.validate()
            return epoch_losses

    return WeightedModel


LAUNCHER = '''"""运行时注入：只替换官方 train.py 使用的 Model 类，其余校验与报告全部复用。"""
import sys
sys.path.insert(0, {pkg_root!r})
sys.path.insert(0, {src_root!r})
from baselines import train as T
from driver.weighted import make_weighted_model
T.Model = make_weighted_model({pkg_root!r}, loss={loss!r}, alpha={alpha!r}, delta={delta!r})
T.main()
'''


def write_launcher(path: str | Path, pkg_root: str | Path, src_root: str | Path,
                   loss: str, alpha: float, delta: float = 1.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LAUNCHER.format(pkg_root=str(Path(pkg_root).resolve()),
                                    src_root=str(Path(src_root).resolve()),
                                    loss=loss, alpha=alpha, delta=delta))
    return path
