import numpy as np
import pytest

from harness.monitor.distribution import (
    DistributionMonitor,
    label_shift,
    psi,
)
from harness.monitor.error_stream import ErrorStreamMonitor
from harness.monitor.performance import evaluate, score_model
from harness.monitor.report import classify

CLASSES = np.arange(1, 7)
rng = np.random.default_rng(0)


# ---------- 性能 ----------

def test_macro_f1_ignores_absent_classes():
    """只出现 1/2 两类时，不应因缺失的 3..6 类被拉低。"""
    y = np.array([1, 1, 2, 2])
    r = evaluate(y, y, CLASSES, "macro_f1")
    assert r.macro_f1 == pytest.approx(1.0)
    assert r.classes_missing == [3, 4, 5, 6]
    assert r.classes_present == [1, 2]


def test_evaluate_handles_empty():
    r = evaluate(np.array([]), np.array([]), CLASSES)
    assert r.n_samples == 0 and r.value != r.value  # NaN


def test_score_model_returns_nan_for_none():
    assert score_model(None, np.zeros((2, 3)), np.array([1, 2]), CLASSES, "macro_f1") != \
        score_model(None, np.zeros((2, 3)), np.array([1, 2]), CLASSES, "macro_f1")


# ---------- 分布 ----------

def test_psi_zero_for_same_distribution():
    x = rng.normal(size=500)
    assert psi(x, x) < 1e-9


def test_psi_grows_with_shift():
    a = rng.normal(size=1000)
    small = psi(a, a + 0.2)
    large = psi(a, a + 2.0)
    assert large > small > 0


def test_psi_handles_empty_and_constant():
    assert psi(np.array([]), np.array([1.0])) == 0.0
    assert psi(np.ones(10), np.ones(10)) == 0.0


def test_distribution_monitor_detects_shift():
    mon = DistributionMonitor(psi_threshold=0.2, psi_feature_ratio=0.1)
    ref = rng.normal(size=(400, 10))
    mon.set_reference(ref)
    assert not mon.compare(rng.normal(size=(400, 10))).drifted
    assert mon.compare(rng.normal(size=(400, 10)) + 3.0).drifted


def test_monitor_without_reference_is_quiet():
    mon = DistributionMonitor()
    assert not mon.has_reference
    assert mon.compare(rng.normal(size=(10, 4))).drifted is False


def test_label_shift_l1():
    a = np.array([1, 1, 2, 2])
    b = np.array([1, 1, 1, 1])
    assert label_shift(a, b, CLASSES) == pytest.approx(1.0)
    assert label_shift(a, a, CLASSES) == pytest.approx(0.0)


# ---------- 误差流 ----------

def test_error_stream_reports_rate():
    mon = ErrorStreamMonitor("adwin")
    y = np.array([1, 1, 1, 1])
    res = mon.update(y, np.array([1, 1, 2, 2]))
    assert res.error_rate == pytest.approx(0.5)


def test_error_stream_detects_sudden_change():
    mon = ErrorStreamMonitor("adwin")
    clean = np.ones(500, dtype=int)
    mon.update(clean, clean)                      # 全对
    res = mon.update(clean, np.full(500, 2))      # 突然全错
    assert res.drift_detected


def test_unknown_detector_degrades_gracefully():
    mon = ErrorStreamMonitor("nope")
    assert mon.kind == "none"
    assert mon.update(np.ones(3, dtype=int), np.ones(3, dtype=int)).drift_detected is False


# ---------- 判定表 ----------

def _mk(dist_drift: bool, err_drift: bool = False):
    from harness.monitor.distribution import DistResult
    from harness.monitor.error_stream import ErrorStreamResult
    d = DistResult(0.3, 0.9, 0.5, [1, 2], 0.4, 1.0, dist_drift)
    e = ErrorStreamResult("adwin", err_drift, 0.1, 10)
    return d, e


def _perf(value: float, missing: list[int] | None = None):
    """直接构造指定数值的性能结果，避免依赖具体预测。"""
    from harness.monitor.performance import PerfResult

    return PerfResult(
        metric_name="macro_f1",
        value=value,
        score=value,
        n_samples=100,
        accuracy=value,
        macro_f1=value,
        classes_present=[1, 2],
        classes_missing=missing if missing is not None else [3, 4, 5, 6],
    )


def test_stable_when_nothing_changes():
    d, e = _mk(False)
    r = classify(0, _perf(0.90), d, e, perf_baseline=0.90, label_shift_value=0.0)
    assert r.drift_type == "STABLE" and not r.perf_dropped


def test_covariate_benign_when_dist_moves_but_perf_holds():
    d, e = _mk(True)
    r = classify(1, _perf(0.90), d, e, perf_baseline=0.90, label_shift_value=0.0)
    assert r.drift_type == "COVARIATE_BENIGN" and not r.perf_dropped


def test_covariate_harmful_when_both_change():
    d, e = _mk(True)
    r = classify(2, _perf(0.75), d, e, perf_baseline=0.90, label_shift_value=0.0)
    assert r.drift_type == "COVARIATE_HARMFUL" and r.perf_dropped


def test_concept_when_perf_drops_without_dist_change():
    d, e = _mk(False)
    r = classify(3, _perf(0.75), d, e, perf_baseline=0.90, label_shift_value=0.0)
    assert r.drift_type == "CONCEPT"


def test_label_drift_only_when_inputs_stable():
    """输入分布稳定、仅类别占比变化时才判 LABEL，可用最便宜的动作修复。"""
    d, e = _mk(False)
    r = classify(4, _perf(0.75), d, e, perf_baseline=0.90, label_shift_value=0.5)
    assert r.drift_type == "LABEL"


def test_input_drift_outranks_label_shift():
    """两者同时出现时按分布漂移处理：重训能一并覆盖类别占比变化，调阈值不能。"""
    d, e = _mk(True)
    r = classify(4, _perf(0.75), d, e, perf_baseline=0.90, label_shift_value=0.5)
    assert r.drift_type == "COVARIATE_HARMFUL"
    assert any("伴随类别占比变化" in n for n in r.notes)


def test_untrusted_on_severe_collapse():
    d, e = _mk(True)
    r = classify(5, _perf(0.20), d, e, perf_baseline=0.90, label_shift_value=0.0)
    assert r.drift_type == "UNTRUSTED"


def test_small_drop_within_tolerance_is_stable():
    """下降幅度小于 perf_drop 阈值时不应触发更新。"""
    d, e = _mk(False)
    r = classify(6, _perf(0.88), d, e, perf_baseline=0.90, label_shift_value=0.0)
    assert r.drift_type == "STABLE"


def test_missing_classes_are_noted():
    d, e = _mk(False)
    r = classify(7, _perf(0.90), d, e, perf_baseline=0.90, label_shift_value=0.0)
    assert any("缺少类别" in n for n in r.notes)
