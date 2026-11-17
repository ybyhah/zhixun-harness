import numpy as np
import pytest

from harness.datasets.base import TaskAdapter
from harness.datasets.cmapss import CMAPSSAdapter, PROGRESSIVE_ORDER
from harness.monitor.performance import asym_score, evaluate, rmse, score_model
from harness.monitor.distribution import target_shift
from harness.models.industrial import IndustrialModel

RAW = "data/CMAPSSData"


# ---------- 回归指标 ----------

def test_rmse_zero_for_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_value():
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(
        np.sqrt((9 + 16) / 2)
    )


def test_asym_penalizes_late_more():
    """高估 RUL（预测偏晚 d>0）应比同幅低估惩罚更重。"""
    y = np.array([50.0])
    late = asym_score(y, np.array([60.0]))    # d=+10
    early = asym_score(y, np.array([40.0]))   # d=-10
    assert late > early > 0


def test_evaluate_regression_score_is_neg_rmse():
    y = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    r = evaluate(y, pred, None, metric="rmse", task_type="regression")
    assert r.rmse == pytest.approx(rmse(y, pred))
    assert r.score == pytest.approx(-r.rmse)      # 统一方向：越大越好
    assert r.n_samples == 3


def test_regression_score_higher_is_better():
    y = np.array([10.0, 20.0, 30.0])
    good = evaluate(y, y + 1, None, task_type="regression").score
    bad = evaluate(y, y + 10, None, task_type="regression").score
    assert good > bad


def test_target_shift_detects_distribution_change():
    a = np.arange(0.0, 100.0)
    assert target_shift(a, a) == pytest.approx(0.0)
    assert target_shift(a, a + 200) > 0.9


# ---------- 回归模型 ----------

def test_regression_model_predicts_continuous():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = X @ np.arange(6) + rng.normal(size=300)
    m = IndustrialModel(class_set=None, task_type="regression", params={"max_iter": 50})
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (300,)
    assert pred.dtype == np.float64
    assert rmse(y, pred) < y.std()               # 比常数预测强


def test_regression_model_rejects_classification_ops():
    m = IndustrialModel(class_set=None, task_type="regression", params={"max_iter": 20})
    m.fit(np.random.default_rng(0).normal(size=(50, 4)), np.arange(50.0))
    with pytest.raises(RuntimeError):
        m.predict_proba(np.zeros((2, 4)))
    with pytest.raises(RuntimeError):
        m.calibrate(np.zeros((2, 4)), np.zeros(2))


def test_score_model_regression_direction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = X @ np.arange(5)
    m = IndustrialModel(class_set=None, task_type="regression", params={"max_iter": 40})
    m.fit(X, y)
    s = score_model(m, X, y, None, "rmse", "regression")
    assert s <= 0.0                              # -RMSE


# ---------- 数据适配器（需要本地数据）----------

def _has_data():
    from pathlib import Path
    return Path(RAW, "train_FD001.txt").exists()


data_required = pytest.mark.skipif(not _has_data(), reason="需要 data/CMAPSSData")


@data_required
def test_adapter_protocol_and_type():
    ad = CMAPSSAdapter(RAW, mode="single")
    assert isinstance(ad, TaskAdapter)
    assert ad.task_type == "regression"
    assert ad.class_set is None


@data_required
def test_single_mode_batch_count():
    ad = CMAPSSAdapter(RAW, mode="single", single_subset="FD001", n_batches=10)
    batches = ad.load_batches()
    assert len(batches) == 10
    assert all(b.X.shape[0] > 0 for b in batches)


@data_required
def test_rul_is_capped_and_nonneg():
    ad = CMAPSSAdapter(RAW, mode="single", rul_cap=125)
    batches = ad.load_batches()
    ally = np.concatenate([b.y for b in batches])
    assert ally.min() >= 0
    assert ally.max() <= 125


@data_required
def test_constant_sensors_dropped():
    ad = CMAPSSAdapter(RAW, mode="single", drop_constant_sensors=True)
    ad.load_batches()
    assert len(ad.dropped_sensors) > 0           # FD001 已知有常数传感器


@data_required
def test_progressive_mode_covers_all_subsets():
    ad = CMAPSSAdapter(RAW, mode="progressive", batches_per_subset=3)
    batches = ad.load_batches()
    assert len(batches) == 3 * len(PROGRESSIVE_ORDER)
