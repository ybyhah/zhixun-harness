"""函数保持式加宽（Net2Net）+ 权重迁移。

官方 TRAINING_GUIDE 允许改网络结构，但要求"显式实现兼容的权重迁移，不能将一个新
模型声明为原模型续训"。这里的做法让迁移后的模型与原模型**输出逐位一致**：
  w1: 新列用 σ=eps 的小随机初始化（打破对称）
  b1: 新位置 0
  w2: **新行置 0** —— 新隐藏单元初始不影响任何输出
  Adam 一阶/二阶矩：按槽位迁移，新槽置 0
  mean/scale/target_mean/target_scale/seed/steps/epochs：原样继承
因此"迁移后指标 == 原模型指标"可以作为真续训的自证，见 assert_function_preserved。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MAX_PARAMETERS = 100000      # 官方 model.py 的上限


def _load_official_model_cls(pkg_root: Path):
    """从官方包里加载 Model 类（不复制官方代码，直接用包内实现）。"""
    path = Path(pkg_root) / "baselines" / "model.py"
    spec = importlib.util.spec_from_file_location("official_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("official_model", module)
    spec.loader.exec_module(module)
    return module.Model


def widen_checkpoint(pkg_root: str | Path, src: str | Path, dst: str | Path,
                     new_hidden: int, *, seed: int = 42, eps: float = 1e-3):
    """把 src 检查点加宽到 new_hidden 并写到 dst。返回 (新参数量, Model 类)。"""
    Model = _load_official_model_cls(pkg_root)
    model = Model.load(src)
    old = int(model.meta["hidden"])
    if new_hidden <= old:
        raise ValueError(f"new_hidden({new_hidden}) 必须大于原 hidden({old})")
    if not 1 <= new_hidden <= 128:
        raise ValueError("官方约束 1 <= hidden <= 128")
    out = len(model.meta["classes"]) if model.meta["task"] == "classification" else 1
    n_in = len(model.meta["paths"])
    count = n_in * new_hidden + new_hidden + new_hidden * out + out
    if count >= MAX_PARAMETERS:
        raise ValueError(f"加宽后参数量 {count} 超官方上限 {MAX_PARAMETERS}")

    rng = np.random.default_rng(seed)

    def grow(name: str, arr: np.ndarray, fill_new: bool) -> np.ndarray:
        if name == "w1":
            new = np.zeros((arr.shape[0], new_hidden))
            new[:, :old] = arr
            if fill_new:
                new[:, old:] = rng.normal(0.0, eps, (arr.shape[0], new_hidden - old))
            return new
        if name == "b1":
            new = np.zeros(new_hidden)
            new[:old] = arr
            return new
        if name == "w2":
            new = np.zeros((new_hidden, out))    # 新行为 0：初始不改变输出
            new[:old] = arr
            return new
        return arr.copy()                        # b2 不变

    model.params = {k: grow(k, v, True) for k, v in model.params.items()}
    model.first = {k: grow(k, v, False) for k, v in model.first.items()}
    model.second = {k: grow(k, v, False) for k, v in model.second.items()}
    model.meta["hidden"] = new_hidden
    model.meta["parameter_count"] = int(sum(v.size for v in model.params.values()))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    model.save(dst)                              # save 内部会 validate
    return model.meta["parameter_count"], Model


def assert_function_preserved(pkg_root: str | Path, src: str | Path, dst: str | Path,
                              x: np.ndarray, *, atol: float = 1e-10,
                              rtol: float = 1e-8) -> float:
    """自证迁移是真续训：加宽前后在同一输入上的预测必须一致。返回最大绝对偏差。

    不要求逐位相等：`hidden @ w2` 的求和长度由 32 变成 64（多出的项恰为 0），
    BLAS 的分块与求和顺序会变，浮点重结合会带来 ~1e-16 量级的差异。
    数学上新单元对输出的贡献严格为 0（w2 新行置 0）。
    """
    Model = _load_official_model_cls(pkg_root)
    before = Model.load(src).predict(x)
    after = Model.load(dst).predict(x)
    if before.shape != after.shape:
        raise ValueError("迁移前后输出形状不一致")
    delta = float(np.max(np.abs(after - before)))
    if not np.allclose(before, after, rtol=rtol, atol=atol):
        raise ValueError(f"迁移不是函数保持的：最大偏差 {delta:.3e}")
    return delta
