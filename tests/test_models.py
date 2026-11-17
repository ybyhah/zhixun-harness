import numpy as np
import pytest

from harness.models.industrial import (
    IndustrialModel,
    TrainBudget,
    adaptive_params,
    decay_weights,
    subsample,
)
from harness.models.registry import ModelRegistry

CLASSES = np.arange(1, 7)


def toy(n=200, feats=8, classes=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(classes, size=n)
    X = rng.normal(size=(n, feats)) + y[:, None] * 0.8
    return X, y


def fitted_model(classes=(1, 2, 3), seed=0):
    X, y = toy(classes=classes, seed=seed)
    m = IndustrialModel(class_set=CLASSES, params={"max_iter": 30})
    m.fit(X, y)
    return m, X, y


# ---------- 模型 ----------

def test_proba_aligned_to_global_class_set():
    """只见过 3 个类，概率矩阵仍要有 6 列，未见类别列为 0。"""
    m, X, _ = fitted_model(classes=(1, 2, 3))
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 6)
    assert np.allclose(proba[:, 3:].sum(), 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_returns_global_labels():
    m, X, y = fitted_model()
    pred = m.predict(X)
    assert set(np.unique(pred).tolist()).issubset(set(CLASSES.tolist()))
    assert (pred == y).mean() > 0.6


def test_unfitted_model_raises():
    m = IndustrialModel(class_set=CLASSES)
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((2, 8)))


def test_sample_weight_changes_fit():
    X, y = toy()
    a = IndustrialModel(class_set=CLASSES, params={"max_iter": 30})
    a.fit(X, y)
    b = IndustrialModel(class_set=CLASSES, params={"max_iter": 30})
    w = np.where(y == 1, 10.0, 0.1)
    b.fit(X, y, sample_weight=w)
    assert not np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_class_bias_shifts_decisions():
    m, X, _ = fitted_model()
    before = m.predict(X)
    bias = np.zeros(6)
    bias[1] = 5.0  # 强烈偏向类别 2
    m.set_class_bias(bias)
    assert (m.predict(X) == 2).mean() > (before == 2).mean()


def test_class_bias_length_validated():
    m, _, _ = fitted_model()
    with pytest.raises(ValueError):
        m.set_class_bias(np.zeros(3))


def test_calibrate_keeps_shape():
    m, X, y = fitted_model()
    m.calibrate(X, y)
    assert m.predict_proba(X).shape == (X.shape[0], 6)


def test_save_load_roundtrip(tmp_path):
    m, X, _ = fitted_model()
    p = m.save(tmp_path / "m.pkl")
    loaded = IndustrialModel.load(p)
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


def test_decay_weights_prefer_recent():
    w = decay_weights(np.array([0, 1, 2]), current=2, decay=0.5)
    assert w[2] > w[1] > w[0]
    assert w[2] == pytest.approx(1.0)


# ---------- 训练预算 ----------

BUDGET = TrainBudget(enabled=True, small_n=1000, min_iter=200, floor_iter=40,
                     max_train_samples=500)


def test_adaptive_params_disabled_is_passthrough():
    base = {"max_iter": 200}
    out = adaptive_params(base, 100000, TrainBudget(enabled=False))
    assert out["max_iter"] == 200


def test_adaptive_iter_shrinks_as_samples_grow():
    base = {"max_iter": 200}
    small = adaptive_params(base, 500, BUDGET)["max_iter"]
    large = adaptive_params(base, 10000, BUDGET)["max_iter"]
    assert small == 200
    assert large < small


def test_adaptive_iter_respects_floor():
    out = adaptive_params({"max_iter": 200}, 10_000_000, BUDGET)
    assert out["max_iter"] == BUDGET.floor_iter


def test_adaptive_iter_never_exceeds_base():
    out = adaptive_params({"max_iter": 50}, 100, BUDGET)
    assert out["max_iter"] == 50


def test_subsample_noop_below_limit():
    y = np.array([1, 1, 2, 2])
    assert np.array_equal(subsample(y, limit=10), np.arange(4))


def test_subsample_respects_limit_and_keeps_all_classes():
    rng = np.random.default_rng(0)
    y = rng.choice([1, 2, 3], size=3000, p=[0.9, 0.08, 0.02])
    sel = subsample(y, limit=300)
    assert sel.size <= 320                       # 每类至少 1 条可能略微超出
    assert set(np.unique(y[sel]).tolist()) == {1, 2, 3}


def test_subsample_keeps_rare_class():
    y = np.array([1] * 999 + [2])
    sel = subsample(y, limit=100)
    assert 2 in y[sel]


def test_fit_records_budget_usage():
    rng = np.random.default_rng(0)
    y = rng.choice([1, 2, 3], size=2000)
    X = rng.normal(size=(2000, 6)) + y[:, None]
    m = IndustrialModel(class_set=CLASSES, params={"max_iter": 200}, budget=BUDGET)
    stats = m.fit(X, y)
    assert stats.n_samples_available == 2000
    assert stats.n_samples <= 520                # 已下采样
    assert stats.max_iter_used == 200            # 下采样后样本量低于 small_n


def test_fit_subsamples_sample_weight_consistently():
    rng = np.random.default_rng(1)
    y = rng.choice([1, 2], size=1500)
    X = rng.normal(size=(1500, 4)) + y[:, None]
    w = np.ones(1500)
    m = IndustrialModel(class_set=CLASSES, params={"max_iter": 30}, budget=BUDGET)
    m.fit(X, y, sample_weight=w)                 # 不应因长度不一致报错
    assert m.fitted


# ---------- 版本库 ----------

def test_bootstrap_then_promote_and_history():
    reg = ModelRegistry()
    m0, _, _ = fitted_model(seed=1)
    reg.bootstrap(m0, batch_index=0)
    assert reg.has_online and reg.online_version.version == 1

    m1, _, _ = fitted_model(seed=2)
    reg.stage(m1, batch_index=1, action="WINDOW_RETRAIN")
    reg.promote(score=0.8)
    assert reg.online_version.version == 2
    assert [v.version for v in reg.history] == [1]


def test_discard_keeps_online_and_counts_failures():
    reg = ModelRegistry()
    m0, _, _ = fitted_model(seed=1)
    reg.bootstrap(m0, batch_index=0)
    online_before = reg.online
    reg.stage(fitted_model(seed=2)[0], batch_index=1, action="FULL_RETRAIN")
    reg.discard()
    assert reg.online is online_before
    assert reg.candidate is None
    assert reg.consecutive_failures == 1


def test_rollback_restores_best_scored_version():
    reg = ModelRegistry()
    reg.bootstrap(fitted_model(seed=1)[0], batch_index=0)
    reg.stage(fitted_model(seed=2)[0], batch_index=1, action="A")
    reg.promote(score=0.9)          # v2 好
    reg.stage(fitted_model(seed=3)[0], batch_index=2, action="B")
    reg.promote(score=0.4)          # v3 差，当前在线
    restored = reg.rollback_to_best(batch_index=3)
    assert restored is not None
    assert restored.action == "ROLLBACK_TO_V2"
    assert reg.consecutive_failures == 0
    assert reg.rollbacks == [{"batch_index": 3, "restored_from": 2}]


def test_rollback_without_history_returns_none():
    reg = ModelRegistry()
    reg.bootstrap(fitted_model(seed=1)[0], batch_index=0)
    assert reg.rollback_to_best(batch_index=1) is None


def test_promote_without_candidate_raises():
    reg = ModelRegistry()
    with pytest.raises(RuntimeError):
        reg.promote()
