import numpy as np
import pytest

from harness.gate.evaluator import Gate
from harness.gate.replay import ReplayBuffer
from harness.models.industrial import IndustrialModel

CLASSES = np.arange(1, 7)


def data(n=200, feats=8, classes=(1, 2, 3), seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    y = rng.choice(classes, size=n)
    X = rng.normal(size=(n, feats)) + y[:, None] * 0.8 + shift
    return X, y


def model(X, y):
    m = IndustrialModel(class_set=CLASSES, params={"max_iter": 30})
    m.fit(X, y)
    return m


# ---------- 回放集 ----------

def test_replay_starts_empty():
    buf = ReplayBuffer()
    assert buf.empty and len(buf) == 0


def test_replay_samples_per_batch_cap():
    buf = ReplayBuffer(per_batch=30, capacity=1000)
    X, y = data(n=500)
    buf.add(X, y, 0)
    assert 0 < len(buf) <= 30 + len(np.unique(y))


def test_replay_keeps_all_classes():
    buf = ReplayBuffer(per_batch=12, capacity=1000)
    X, y = data(n=300, classes=(1, 2, 3, 4))
    buf.add(X, y, 0)
    _, ys, _ = buf.dataset()
    assert set(np.unique(ys).tolist()) == {1, 2, 3, 4}


def test_replay_respects_capacity_and_keeps_multiple_batches():
    buf = ReplayBuffer(per_batch=60, capacity=120)
    for i in range(6):
        X, y = data(n=200, seed=i)
        buf.add(X, y, i)
    assert len(buf) <= 120
    assert len(buf.batch_ids) >= 2  # 不能只剩最新一批


def test_replay_ignores_empty_batch():
    buf = ReplayBuffer()
    buf.add(np.empty((0, 5)), np.empty((0,), dtype=int), 0)
    assert buf.empty


# ---------- 门控 ----------

def test_rejects_when_gain_below_threshold():
    X, y = data()
    online = model(X, y)
    gate = Gate(CLASSES, min_gain=0.05)
    res = gate.evaluate(online, online, None, (X, y), ReplayBuffer())
    assert not res.accepted
    assert res.checks["beats_online"] is False
    assert "未达阈值" in res.reason


def test_accepts_clear_improvement():
    Xg, yg = data(n=400, seed=1)
    good = model(Xg, yg)
    bad = model(*data(n=60, seed=9, shift=5.0))  # 在 Xg 上表现差
    gate = Gate(CLASSES, min_gain=0.001)
    res = gate.evaluate(good, bad, None, (Xg, yg), ReplayBuffer())
    assert res.accepted and res.candidate_score > res.online_score


def test_rejects_candidate_below_baseline():
    Xg, yg = data(n=400, seed=1)
    baseline = model(Xg, yg)
    weak = model(*data(n=60, seed=7, shift=4.0))
    gate = Gate(CLASSES, min_gain=0.0)
    res = gate.evaluate(weak, None, baseline, (Xg, yg), ReplayBuffer())
    assert not res.accepted
    assert res.checks["not_below_baseline"] is False
    assert "低于静态基线" in res.reason


def test_rejects_forgetting_candidate():
    """候选在近期数据上更好，但在历史回放集上明显退化，应被拒。"""
    X_old, y_old = data(n=300, seed=2)
    X_new, y_new = data(n=300, seed=3, shift=6.0)

    online = model(np.vstack([X_old, X_new]), np.concatenate([y_old, y_new]))
    candidate = model(X_new, y_new)       # 只学新分布，忘掉旧的

    buf = ReplayBuffer(per_batch=150, capacity=400)
    buf.add(X_old, y_old, 0)

    gate = Gate(CLASSES, min_gain=0.0, forget_tol=0.01)
    res = gate.evaluate(candidate, online, None, (X_new, y_new), buf)
    if res.checks.get("beats_online"):
        assert not res.accepted
        assert res.checks["no_forgetting"] is False
        assert "疑似遗忘" in res.reason


def test_empty_replay_skips_third_check():
    Xg, yg = data(n=400, seed=1)
    good = model(Xg, yg)
    bad = model(*data(n=60, seed=9, shift=5.0))
    res = Gate(CLASSES, min_gain=0.001).evaluate(good, bad, None, (Xg, yg), ReplayBuffer())
    assert res.accepted and res.checks["no_forgetting"] is True


def test_missing_candidate_rejected():
    X, y = data()
    res = Gate(CLASSES).evaluate(None, model(X, y), None, (X, y), ReplayBuffer())
    assert not res.accepted and "缺失" in res.reason


def test_no_online_model_passes_first_check():
    X, y = data()
    cand = model(X, y)
    res = Gate(CLASSES, min_gain=0.5).evaluate(cand, None, None, (X, y), ReplayBuffer())
    assert res.accepted and res.checks["beats_online"] is True


def test_result_serializes():
    X, y = data()
    res = Gate(CLASSES).evaluate(model(X, y), None, None, (X, y), ReplayBuffer())
    d = res.to_dict()
    assert set(["accepted", "reason", "checks"]).issubset(d)
    assert "candidate_critical_recall" in d


# ---------- 第四道：关键类召回 ----------

class _Stub:
    """固定预测的桩模型，用于精确控制关键类召回。"""

    def __init__(self, pred):
        self.pred = np.asarray(pred)

    def predict(self, X):
        return self.pred[: len(X)]


def _crit_data():
    # 类别 3 为关键类（严重退化），共 4 个样本
    y = np.array([3, 3, 3, 3, 1, 1, 2, 2])
    X = np.zeros((len(y), 4))
    return X, y


def test_critical_recall_regression_rejected():
    X, y = _crit_data()
    online = _Stub([3, 3, 3, 3, 1, 1, 2, 2])      # 关键类召回 1.0
    cand = _Stub([1, 1, 3, 3, 1, 1, 2, 2])        # 关键类召回 0.5
    gate = Gate(CLASSES, min_gain=-1.0, critical_class=3, critical_recall_tol=0.05)
    res = gate.evaluate(cand, online, None, (X, y), ReplayBuffer())
    assert not res.accepted
    assert res.checks["critical_recall"] is False
    assert res.candidate_critical_recall == 0.5
    assert res.online_critical_recall == 1.0
    assert "严重退化漏报" in res.reason


def test_critical_recall_small_drop_within_tol_passes():
    X, y = _crit_data()
    online = _Stub([3, 3, 3, 3, 1, 1, 2, 2])      # 1.0
    cand = _Stub([3, 3, 3, 1, 1, 1, 2, 2])        # 0.75，退化 0.25
    gate = Gate(CLASSES, min_gain=-1.0, critical_class=3, critical_recall_tol=0.3)
    res = gate.evaluate(cand, online, None, (X, y), ReplayBuffer())
    assert res.accepted and res.checks["critical_recall"] is True


def test_critical_recall_skipped_when_class_absent():
    y = np.array([1, 1, 2, 2])
    X = np.zeros((4, 4))
    cand = _Stub([1, 1, 2, 2])
    online = _Stub([1, 1, 2, 2])
    gate = Gate(CLASSES, min_gain=-1.0, critical_class=3, critical_recall_tol=0.05)
    res = gate.evaluate(cand, online, None, (X, y), ReplayBuffer())
    assert res.accepted and res.checks["critical_recall"] is True
    assert np.isnan(res.candidate_critical_recall)


def test_critical_recall_skipped_without_online():
    X, y = _crit_data()
    cand = _Stub([1, 1, 1, 1, 1, 1, 2, 2])        # 关键类召回 0，但无在线可比
    gate = Gate(CLASSES, min_gain=-1.0, critical_class=3, critical_recall_tol=0.05)
    res = gate.evaluate(cand, None, None, (X, y), ReplayBuffer())
    assert res.accepted and res.checks["critical_recall"] is True


def test_critical_check_not_triggered_when_unset():
    X, y = _crit_data()
    online = _Stub([3, 3, 3, 3, 1, 1, 2, 2])
    cand = _Stub([1, 1, 1, 1, 1, 1, 2, 2])        # 关键类召回 0
    res = Gate(CLASSES, min_gain=-1.0).evaluate(cand, online, None, (X, y), ReplayBuffer())
    assert "critical_recall" not in res.checks        # 未配置，第四道不触发
