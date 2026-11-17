import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.migrate import (
    _load_official_model_cls, assert_function_preserved, widen_checkpoint,
)
from driver.package import load_package

PKG_A = ROOT / "base" / "cpu-a-guide-20261110-v1"
has_a = PKG_A.is_dir()
pytestmark = pytest.mark.skipif(not has_a, reason="需要 base/A 包")


@pytest.fixture
def ckpt(tmp_path):
    return PKG_A / "checkpoint.npz", tmp_path / "wide.npz"


def _sample_x(n=16):
    Model = _load_official_model_cls(PKG_A)
    m = Model.load(PKG_A / "checkpoint.npz")
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, len(m.meta["paths"])))


def test_widen_is_function_preserving(ckpt):
    src, dst = ckpt
    widen_checkpoint(PKG_A, src, dst, 64)
    delta = assert_function_preserved(PKG_A, src, dst, _sample_x())
    assert delta < 1e-12          # 仅浮点重结合级别的差异


def test_widen_updates_meta_and_shapes(ckpt):
    src, dst = ckpt
    count, Model = widen_checkpoint(PKG_A, src, dst, 64)
    m = Model.load(dst)
    assert m.meta["hidden"] == 64
    assert m.meta["parameter_count"] == count == 128 * 64 + 64 + 64 * 6 + 6
    assert m.params["w1"].shape == (128, 64)
    assert m.params["w2"].shape == (64, 6)
    # 原权重完整保留在前 32 槽
    old = Model.load(src)
    assert np.array_equal(m.params["w1"][:, :32], old.params["w1"])
    assert np.array_equal(m.params["w2"][:32], old.params["w2"])
    assert np.array_equal(m.params["b2"], old.params["b2"])


def test_new_w2_rows_are_zero(ckpt):
    src, dst = ckpt
    _, Model = widen_checkpoint(PKG_A, src, dst, 64)
    m = Model.load(dst)
    assert np.all(m.params["w2"][32:] == 0)        # 新单元初始不影响输出
    assert np.all(m.params["b1"][32:] == 0)


def test_adam_moments_migrate_with_zero_new_slots(ckpt):
    src, dst = ckpt
    _, Model = widen_checkpoint(PKG_A, src, dst, 64)
    old, new = Model.load(src), Model.load(dst)
    assert np.array_equal(new.first["w1"][:, :32], old.first["w1"])
    assert np.all(new.first["w1"][:, 32:] == 0)
    assert np.all(new.second["w2"][32:] == 0)


def test_normalization_and_counters_inherited(ckpt):
    src, dst = ckpt
    _, Model = widen_checkpoint(PKG_A, src, dst, 64)
    old, new = Model.load(src), Model.load(dst)
    assert np.array_equal(new.mean, old.mean) and np.array_equal(new.scale, old.scale)
    for k in ("seed", "steps", "epochs", "task", "classes", "learning_rate",
              "target_mean", "target_scale"):
        assert new.meta[k] == old.meta[k]


def test_rejects_non_widening(ckpt):
    src, dst = ckpt
    with pytest.raises(ValueError, match="必须大于"):
        widen_checkpoint(PKG_A, src, dst, 32)


def test_rejects_beyond_official_hidden_bound(ckpt):
    src, dst = ckpt
    with pytest.raises(ValueError, match="1 <= hidden <= 128"):
        widen_checkpoint(PKG_A, src, dst, 200)


def test_package_metric_matches_after_migration(ckpt):
    """迁移后主指标与官方基线一致（函数保持的端到端体现）。"""
    src, dst = ckpt
    pkg = load_package(PKG_A)
    Model = _load_official_model_cls(PKG_A)
    sys.path.insert(0, str(PKG_A))
    from baselines.data import read_dataset
    from baselines.train import metrics

    tr = read_dataset(pkg.train_jsonl, pkg.task, list(Model.load(src).meta["classes"]))
    va = read_dataset(pkg.validation_jsonl, pkg.task,
                      list(Model.load(src).meta["classes"]), tr["paths"])
    widen_checkpoint(PKG_A, src, dst, 64)
    assert metrics(Model.load(dst), va)["macro_f1"] == pytest.approx(
        metrics(Model.load(src), va)["macro_f1"], abs=0.0)
