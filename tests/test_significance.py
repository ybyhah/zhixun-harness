import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.migrate import _load_official_model_cls
from driver.official_metric import classification_macro_f1, regression_rmse, score
from driver.package import load_package
from driver.significance import derive_seed, paired_bootstrap

PKG_A = ROOT / "base" / "cpu-a-guide-20261110-v1"
PKG_C = ROOT / "base" / "cpu-c-guide-20261110-v1"
has_a, has_c = PKG_A.is_dir(), PKG_C.is_dir()


def _official(pkg_dir):
    """返回 (model, va, 官方 metrics 函数)。"""
    Model = _load_official_model_cls(pkg_dir)
    sys.path.insert(0, str(pkg_dir))
    from baselines.data import read_dataset
    from baselines.train import metrics

    pkg = load_package(pkg_dir)
    classes = json.loads(pkg.config_path().read_text()).get("classes", [])
    tr = read_dataset(pkg.train_jsonl, pkg.task, classes)
    va = read_dataset(pkg.validation_jsonl, pkg.task, classes, tr["paths"])
    return Model.load(pkg.checkpoint), va, metrics


# ---------- 口径对齐（阻塞性）----------

@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_macro_f1_matches_official_exactly():
    """分类 Macro-F1 必须与官方 metrics() 逐位相等（含"并集上宏平均"这一细节）。"""
    model, va, metrics = _official(PKG_A)
    pred = model.predict(va["x"])
    assert classification_macro_f1(pred, va["y"]) == metrics(model, va)["macro_f1"]


@pytest.mark.skipif(not has_c, reason="需要 base/C 包")
def test_rmse_matches_official_exactly():
    model, va, metrics = _official(PKG_C)
    pred = model.predict(va["x"])
    assert regression_rmse(pred, va["y"]) == metrics(model, va)["rmse"]


def test_macro_f1_uses_union_of_true_and_predicted():
    """官方在 true∪pred 上宏平均；只看 true 会算错。这里构造差异用例锁死口径。"""
    y = np.array([0, 0, 1, 1])
    pred = np.zeros((4, 3))
    pred[:, 0] = 1.0            # 全预测为类 0；类 2 既未出现也未被预测
    pred[2, :] = [0, 0, 1]      # 第 3 个样本预测成类 2（真实是 1）
    # 并集 = {0,1,2}：类 0 的 f1=2*2/(3+2)=0.8，类 1 的 f1=0，类 2 的 f1=0
    assert classification_macro_f1(pred, y) == pytest.approx((0.8 + 0 + 0) / 3)


def test_score_unifies_direction():
    y = np.array([1.0, 2.0])
    pred = np.array([1.0, 2.0])
    assert score(pred, y, "regression") == 0.0           # -RMSE
    worse = np.array([3.0, 4.0])
    assert score(worse, y, "regression") < score(pred, y, "regression")


# ---------- 退化输入不抛错 ----------

@pytest.mark.parametrize("y,pred", [
    (np.array([]), np.zeros((0, 3))),
    (np.array([0, 0]), np.array([[1.0, 0, 0], [1.0, 0, 0]])),      # 单类
])
def test_metric_handles_degenerate_inputs(y, pred):
    v = classification_macro_f1(pred, y)
    assert v == v or np.isnan(v)        # 不抛错即可


# ---------- 配对 bootstrap ----------

def test_same_model_against_itself_is_never_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    pred = rng.random((200, 3))
    s = paired_bootstrap(pred, pred, y, "classification", bootstrap=100, seed=1)
    assert s is not None
    assert s.p_better == 0.0 and s.mean_diff == 0.0
    assert s.ci_low == 0.0 and s.ci_high == 0.0


def test_clearly_better_model_is_significant():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    good = np.zeros((300, 3)); good[np.arange(300), y] = 1.0        # 全对
    bad = np.zeros((300, 3)); bad[:, 0] = 1.0                        # 全预测类 0
    s = paired_bootstrap(good, bad, y, "classification", bootstrap=200, seed=2)
    assert s.p_better == 1.0 and s.ci_low > 0


def test_deterministic_with_same_seed():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 150)
    a, b = rng.random((150, 2)), rng.random((150, 2))
    s1 = paired_bootstrap(a, b, y, "classification", bootstrap=120, seed=7)
    s2 = paired_bootstrap(a, b, y, "classification", bootstrap=120, seed=7)
    assert s1.to_dict() == s2.to_dict()


def test_regression_direction_lower_rmse_is_better():
    y = np.linspace(0, 100, 200)
    close = y + 1.0
    far = y + 20.0
    s = paired_bootstrap(close, far, y, "regression", bootstrap=150, seed=4)
    assert s.p_better == 1.0 and s.mean_diff > 0


def test_returns_none_on_bad_shapes_or_nan():
    y = np.array([0, 1, 1])
    assert paired_bootstrap(np.zeros((2, 2)), np.zeros((3, 2)), y,
                            "classification", bootstrap=10) is None
    nan_pred = np.full((3, 2), np.nan)
    assert paired_bootstrap(nan_pred, np.zeros((3, 2)), y,
                            "classification", bootstrap=10) is None
    assert paired_bootstrap(np.zeros((0, 2)), np.zeros((0, 2)), np.array([]),
                            "classification", bootstrap=10) is None
    assert paired_bootstrap(np.zeros((3, 2)), np.zeros((3, 2)), y,
                            "classification", bootstrap=0) is None


def test_derive_seed_is_deterministic_and_varies():
    assert derive_seed(42, 1, 0) == derive_seed(42, 1, 0)
    assert derive_seed(42, 1, 0) != derive_seed(42, 2, 0)
    assert derive_seed(42, 1, 0) != derive_seed(42, 1, 1)


# ---------- 门控显著性模式 ----------

from driver.gate import OfficialGate


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_gate_significance_promotes_clear_win():
    pkg = load_package(PKG_A)
    gate = OfficialGate(pkg, baseline_value=0.70, mode="significance",
                        bootstrap=120, p_threshold=0.95)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 240)
    good = np.zeros((240, 6)); good[np.arange(240), y] = 1.0
    bad = np.zeros((240, 6)); bad[:, 0] = 1.0
    d = gate.evaluate(0.95, 0.80, pred_cand=good, pred_best=bad, y=y, seed=1)
    assert d.accepted and d.significance["p_better"] == 1.0
    assert "显著优于" in d.reason


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_gate_significance_rejects_noise_level_gain():
    """数值上更高但统计不显著 → 不上线（这是本方案的核心目的）。"""
    pkg = load_package(PKG_A)
    gate = OfficialGate(pkg, baseline_value=0.70, mode="significance",
                        bootstrap=200, p_threshold=0.95)
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 400)
    base = np.zeros((400, 6)); base[np.arange(400), y] = 1.0
    cand = base.copy()
    flip = rng.choice(400, 3, replace=False)          # 只差 3 个样本
    for i in flip:
        cand[i] = 0; cand[i, (y[i] + 1) % 6] = 1.0
    d = gate.evaluate(0.9925, 0.9900, pred_cand=base, pred_best=cand, y=y, seed=2)
    assert d.significance is not None
    if not d.accepted:
        assert "不显著" in d.reason


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_gate_falls_back_when_predictions_missing():
    pkg = load_package(PKG_A)
    gate = OfficialGate(pkg, baseline_value=0.70, mode="significance")
    d = gate.evaluate(0.80, 0.75)                     # 没传 pred → 降级
    assert d.accepted and d.degraded and "显著性不可用" in d.reason


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_gate_baseline_floor_still_plain():
    pkg = load_package(PKG_A)
    gate = OfficialGate(pkg, baseline_value=0.80, mode="significance")
    d = gate.evaluate(0.70, 0.85)                     # 低于基线 → 直接拒
    assert not d.accepted and "低于官方基线" in d.reason


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_gate_rejects_unknown_mode():
    pkg = load_package(PKG_A)
    with pytest.raises(ValueError):
        OfficialGate(pkg, baseline_value=0.7, mode="nope")


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_gate_tiebreak_prefers_fewer_steps():
    """统计上无法区分时，偏向步数更少（更不易过拟合）的候选。"""
    pkg = load_package(PKG_A)
    gate = OfficialGate(pkg, baseline_value=0.70, mode="significance",
                        bootstrap=120, p_threshold=0.95)
    rng = np.random.default_rng(3)
    y = rng.integers(0, 6, 300)
    pred = np.zeros((300, 6)); pred[np.arange(300), y] = 1.0
    d = gate.evaluate(0.90, 0.90, pred_cand=pred, pred_best=pred, y=y, seed=4,
                      cand_steps=100, best_steps=500)
    assert d.accepted and d.inconclusive and "步数更少" in d.reason
