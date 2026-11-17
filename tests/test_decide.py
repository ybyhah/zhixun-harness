from dataclasses import dataclass, field

import numpy as np
import pytest

from harness.decide.actions import (
    ACTION_COST,
    Action,
    ActionExecutor,
    Decider,
    Decision,
    TrainingData,
)
from harness.decide.rules import RuleDecider
from harness.monitor.distribution import DistResult
from harness.monitor.error_stream import ErrorStreamResult
from harness.monitor.performance import PerfResult
from harness.monitor.report import DriftReport

CLASSES = np.arange(1, 7)


def mk_report(drift_type="CONCEPT", n=200, value=0.70, baseline=0.90, top=(1, 2, 3)):
    perf = PerfResult("macro_f1", value, value, n, accuracy=value, macro_f1=value,
                      classes_present=[1, 2, 3], classes_missing=[4, 5, 6])
    dist = DistResult(0.3, 0.9, 0.5, list(top), 0.4, 1.5, True)
    err = ErrorStreamResult("adwin", False, 0.2, 100)
    return DriftReport(
        batch_index=1, perf=perf, dist=dist, error_stream=err,
        perf_baseline=baseline, perf_delta=value - baseline,
        label_shift=0.0, drift_type=drift_type,
    )


@dataclass
class FakeRecord:
    drift_type: str
    gate_result: dict = field(default_factory=dict)


def pool_with(n_batches=4, rows=120, feats=10, seed=0) -> TrainingData:
    rng = np.random.default_rng(seed)
    pool = TrainingData()
    for i in range(n_batches):
        y = rng.choice([1, 2, 3], size=rows)
        X = rng.normal(size=(rows, feats)) + y[:, None] * 0.8
        pool.add(X, y, i)
    return pool


# ---------- 决策器 ----------

def test_rule_decider_satisfies_protocol():
    assert isinstance(RuleDecider(), Decider)


def test_noop_when_stable():
    d = RuleDecider().decide(mk_report("STABLE"), [])
    assert d.action is Action.NOOP


def test_noop_when_covariate_benign():
    """分布变了但性能没掉，不更新才是正确决策。"""
    d = RuleDecider().decide(mk_report("COVARIATE_BENIGN"), [])
    assert d.action is Action.NOOP
    assert "仍适用" in d.reason


def test_small_batch_is_protected():
    d = RuleDecider(min_batch_size=50).decide(mk_report("CONCEPT", n=10), [])
    assert d.action is Action.NOOP
    assert "只累积数据" in d.reason


def test_label_drift_picks_cheapest_action():
    d = RuleDecider().decide(mk_report("LABEL"), [])
    assert d.action is Action.ADJUST_THRESHOLD
    assert ACTION_COST[d.action] == 1


def test_covariate_harmful_picks_window_retrain():
    d = RuleDecider(default_window=3).decide(mk_report("COVARIATE_HARMFUL"), [])
    assert d.action is Action.WINDOW_RETRAIN
    assert d.params["window"] == 3


def test_concept_uses_stronger_decay():
    d = RuleDecider().decide(mk_report("CONCEPT"), [])
    assert d.action is Action.WINDOW_RETRAIN


def test_action_escalates_after_gate_rejection():
    hist = [FakeRecord("LABEL", {"accepted": False})]
    d = RuleDecider().decide(mk_report("LABEL"), hist)
    assert d.action is Action.RECALIBRATE
    assert "动作升级" in d.reason


def test_escalation_stops_at_ladder_end():
    hist = [FakeRecord("CONCEPT", {"accepted": False})] * 5
    d = RuleDecider().decide(mk_report("CONCEPT"), hist)
    assert d.action is Action.FULL_RETRAIN


def test_escalation_resets_after_success():
    hist = [FakeRecord("LABEL", {"accepted": False}), FakeRecord("LABEL", {"accepted": True})]
    d = RuleDecider().decide(mk_report("LABEL"), hist)
    assert d.action is Action.ADJUST_THRESHOLD


def test_offline_only_when_enabled_and_exhausted():
    hist = [FakeRecord("UNTRUSTED", {"accepted": False})] * 3
    assert RuleDecider(allow_offline=False).decide(mk_report("UNTRUSTED"), hist).action \
        is not Action.OFFLINE
    assert RuleDecider(allow_offline=True).decide(mk_report("UNTRUSTED"), hist).action \
        is Action.OFFLINE


def test_feature_subset_drops_worst_psi_features():
    hist = [FakeRecord("COVARIATE_HARMFUL", {"accepted": False})] * 2
    d = RuleDecider().decide(mk_report("COVARIATE_HARMFUL", top=(7, 3, 5)), hist)
    assert d.action is Action.FEATURE_SUBSET
    assert d.params["drop_features"] == [7, 3, 5]


# ---------- 训练数据池 ----------

def test_window_returns_recent_batches_only():
    pool = pool_with(n_batches=4, rows=50)
    X, y, bid = pool.window(2)
    assert X.shape[0] == 100
    assert set(np.unique(bid).tolist()) == {2, 3}


def test_window_clamped_to_available():
    pool = pool_with(n_batches=2, rows=10)
    X, _, _ = pool.window(99)
    assert X.shape[0] == 20


def test_all_returns_everything():
    pool = pool_with(n_batches=3, rows=10)
    X, y, bid = pool.all()
    assert X.shape[0] == 30 and len(np.unique(bid)) == 3


def test_window_for_samples_expands_when_short():
    """每批 50 条，要求 180 条时窗口应从 2 扩展到 4。"""
    pool = pool_with(n_batches=5, rows=50)
    assert pool.window_for_samples(k_pref=2, min_samples=180) == 4


def test_window_for_samples_keeps_preference_when_enough():
    pool = pool_with(n_batches=5, rows=200)
    assert pool.window_for_samples(k_pref=2, min_samples=180) == 2


def test_window_for_samples_capped_by_available():
    pool = pool_with(n_batches=3, rows=10)
    assert pool.window_for_samples(k_pref=2, min_samples=99999) == 3


def test_window_for_samples_disabled_by_zero():
    pool = pool_with(n_batches=5, rows=10)
    assert pool.window_for_samples(k_pref=2, min_samples=0) == 2


# ---------- 执行器 ----------

def executor():
    return ActionExecutor(CLASSES, "hist", {"max_iter": 20}, seed=0)


def test_window_retrain_honors_min_window_samples():
    """窗口偏好 1 批（50 条），但最小样本量 150 应把窗口撑到 3 批。"""
    ex = ActionExecutor(CLASSES, "hist", {"max_iter": 20}, seed=0,
                        min_window_samples=150)
    pool = pool_with(n_batches=4, rows=50)
    m = ex.execute(Decision(Action.WINDOW_RETRAIN, {"window": 1}), pool, None, 3, 0.9)
    assert m is not None and m.stats.n_samples == 150


def test_noop_produces_no_candidate():
    ex = executor()
    assert ex.execute(Decision(Action.NOOP), pool_with(), None, 3, 0.9) is None
    assert ex.train_calls == 0


def test_window_retrain_produces_fitted_model():
    ex = executor()
    pool = pool_with()
    m = ex.execute(Decision(Action.WINDOW_RETRAIN, {"window": 2}), pool, None, 3, 0.9)
    assert m is not None and m.fitted
    assert m.predict_proba(pool.X[-1]).shape[1] == 6
    assert ex.train_calls == 1


def test_full_retrain_uses_all_batches():
    ex = executor()
    pool = pool_with(n_batches=3, rows=60)
    m = ex.execute(Decision(Action.FULL_RETRAIN, {"decay": 0.5}), pool, None, 2, 0.9)
    assert m is not None and m.stats.n_samples == 180


def test_feature_subset_reduces_dimension():
    ex = executor()
    pool = pool_with(feats=10)
    m = ex.execute(
        Decision(Action.FEATURE_SUBSET, {"window": 2, "drop_features": [0, 1]}),
        pool, None, 3, 0.9,
    )
    assert m is not None and m.stats.n_features == 8


def test_threshold_action_requires_online_model():
    ex = executor()
    assert ex.execute(Decision(Action.ADJUST_THRESHOLD), pool_with(), None, 1, 0.9) is None


def test_threshold_action_sets_bias_on_clone():
    ex = executor()
    pool = pool_with()
    X, y, bid = pool.all()
    base = ex.execute(Decision(Action.FULL_RETRAIN), pool, None, 3, 0.9)
    out = ex.execute(Decision(Action.ADJUST_THRESHOLD), pool, base, 3, 0.9)
    assert out is not None and out is not base


def test_single_class_window_cannot_train():
    ex = executor()
    pool = TrainingData()
    pool.add(np.random.default_rng(0).normal(size=(40, 5)), np.ones(40, dtype=int), 0)
    assert ex.execute(Decision(Action.WINDOW_RETRAIN), pool, None, 0, 0.9) is None


# ---------- split_last 门控验证口径 ----------

def test_split_last_in_batch_holds_out_from_last_only():
    pool = pool_with(n_batches=3, rows=100)
    trimmed, vX, vy = pool.split_last(val_frac=0.3, val_mode="in_batch")
    # 验证集约为最近批次的 30%，训练池仍覆盖全部 3 个批次
    assert 20 <= len(vy) <= 40
    assert trimmed.n_batches == 3
    assert trimmed.all()[0].shape[0] == 300 - len(vy)


def test_split_last_holdout_recent_excludes_last_batch():
    pool = pool_with(n_batches=3, rows=100)
    trimmed, vX, vy = pool.split_last(val_mode="holdout_recent")
    # 验证集 = 整个最近批次；训练池不含最近批次
    assert len(vy) == 100
    assert trimmed.n_batches == 2
    assert np.array_equal(vX, pool.X[-1])
    assert trimmed.all()[0].shape[0] == 200


def test_split_last_holdout_recent_falls_back_on_single_batch():
    pool = pool_with(n_batches=1, rows=80)
    trimmed, vX, vy = pool.split_last(val_mode="holdout_recent")
    # 只有 1 批时无更早数据可训练，回退 in_batch：验证集是切片而非整批
    assert len(vy) < 80
    assert trimmed.n_batches == 1


def test_split_last_rejects_unknown_mode():
    pool = pool_with(n_batches=2, rows=50)
    with pytest.raises(ValueError):
        pool.split_last(val_mode="whatever")


def test_split_last_holdout_refit_matches_holdout_recent():
    pool = pool_with(n_batches=3, rows=100)
    t1, x1, y1 = pool.split_last(val_mode="holdout_recent")
    t2, x2, y2 = pool.split_last(val_mode="holdout_refit")
    assert t1.n_batches == t2.n_batches == 2
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert len(y2) == 100
