import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.metric import nasa_asym_score, over_estimate_rate
from driver.migrate import _load_official_model_cls
from driver.weighted import class_weights, make_weighted_model, write_launcher

PKG_A = ROOT / "base" / "cpu-a-guide-20261110-v1"
PKG_C = ROOT / "base" / "cpu-c-guide-20261110-v1"
has_a, has_c = PKG_A.is_dir(), PKG_C.is_dir()


# ---------- 类别权重 ----------

def test_class_weights_mean_is_one():
    y = np.array([0, 0, 0, 1, 2, 2])
    w = class_weights(y, 3)
    assert w.mean() == pytest.approx(1.0)
    assert w[1] > w[0]                      # 少数类权重更大


def test_class_weights_absent_class_is_zero():
    w = class_weights(np.array([0, 0, 1, 1]), 3)
    assert w[2] == 0.0
    assert w[:2].mean() == pytest.approx(1.0)


# ---------- NASA 非对称分 ----------

def test_asym_penalizes_over_estimate_more():
    y = np.array([50.0])
    assert nasa_asym_score(y, np.array([60.0])) > nasa_asym_score(y, np.array([40.0])) > 0


def test_over_estimate_rate():
    assert over_estimate_rate(np.array([1.0, 2.0, 3.0]),
                              np.array([2.0, 1.0, 4.0])) == pytest.approx(2 / 3)


# ---------- 可关自证：等价于官方损失 ----------

@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_weighted_ce_with_uniform_classes_matches_official():
    """类别均衡时加权交叉熵的权重全为 1，fit 结果应与官方逐位一致。"""
    Model = _load_official_model_cls(PKG_A)
    W = make_weighted_model(PKG_A, loss="weighted_ce")
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(6), 10)               # 每类 10 个，权重均为 1
    x = rng.normal(size=(y.size, 128))
    assert y.size == 60 and len(np.unique(y)) == 6
    a, b = Model.load(PKG_A / "checkpoint.npz"), W.load(PKG_A / "checkpoint.npz")
    la, lb = a.fit(x, y, 2), b.fit(x, y, 2)
    assert lb == pytest.approx(la, rel=1e-12)
    for k in a.params:
        assert np.allclose(b.params[k], a.params[k], rtol=0, atol=0)


@pytest.mark.skipif(not has_c, reason="需要 base/C 包")
def test_asym_alpha_one_matches_official():
    """alpha=1 时非对称损失退化为官方平方误差，fit 结果逐位一致。"""
    Model = _load_official_model_cls(PKG_C)
    W = make_weighted_model(PKG_C, loss="asym", alpha=1.0)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(48, 24))
    y = rng.uniform(0, 120, size=48)
    a, b = Model.load(PKG_C / "checkpoint.npz"), W.load(PKG_C / "checkpoint.npz")
    la, lb = a.fit(x, y, 2), b.fit(x, y, 2)
    assert lb == pytest.approx(la, rel=1e-12)
    for k in a.params:
        assert np.allclose(b.params[k], a.params[k], rtol=0, atol=0)


# ---------- 非对称损失的方向 ----------

@pytest.mark.skipif(not has_c, reason="需要 base/C 包")
def test_asym_alpha_greater_than_one_changes_fit():
    Model = _load_official_model_cls(PKG_C)
    W = make_weighted_model(PKG_C, loss="asym", alpha=3.0)
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(48, 24)), rng.uniform(0, 120, size=48)
    a, b = Model.load(PKG_C / "checkpoint.npz"), W.load(PKG_C / "checkpoint.npz")
    la, lb = a.fit(x, y, 1), b.fit(x, y, 1)
    assert lb[0] > la[0]                        # 高估被额外惩罚，损失更大
    assert not np.allclose(b.params["w2"], a.params["w2"])


@pytest.mark.skipif(not has_c, reason="需要 base/C 包")
def test_huber_is_bounded_vs_squared_on_outliers():
    Model = _load_official_model_cls(PKG_C)
    W = make_weighted_model(PKG_C, loss="huber", delta=1.0)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 24))
    y = np.full(32, 1000.0)                     # 远离目标分布的离群值
    a, b = Model.load(PKG_C / "checkpoint.npz"), W.load(PKG_C / "checkpoint.npz")
    assert b.fit(x, y, 1)[0] < a.fit(x, y, 1)[0]


# ---------- 产物与官方推理兼容 ----------

@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_checkpoint_loads_with_official_model(tmp_path):
    Model = _load_official_model_cls(PKG_A)
    W = make_weighted_model(PKG_A, loss="weighted_ce")
    rng = np.random.default_rng(4)
    m = W.load(PKG_A / "checkpoint.npz")
    m.fit(rng.normal(size=(64, 128)), rng.integers(0, 6, 64), 1)
    dst = tmp_path / "w.npz"
    m.save(dst)
    official = Model.load(dst)                  # 官方未修改的类能原样加载
    assert official.meta["parameter_count"] == 4326
    assert official.predict(rng.normal(size=(5, 128))).shape == (5, 6)


def test_rejects_unknown_loss_and_bad_alpha():
    with pytest.raises(ValueError):
        make_weighted_model(PKG_A, loss="nope")
    with pytest.raises(ValueError):
        make_weighted_model(PKG_A, loss="asym", alpha=0)


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_write_launcher_contains_injection(tmp_path):
    p = write_launcher(tmp_path / "patched_train.py", PKG_A, ROOT / "src",
                       "asym", 2.0)
    text = p.read_text()
    assert "T.Model = make_weighted_model" in text and "T.main()" in text
    assert "loss='asym'" in text and "alpha=2.0" in text
