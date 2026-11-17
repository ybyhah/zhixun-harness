import numpy as np
import pytest

from harness.metrics import (
    composite,
    forgetting,
    recovery,
    resource_efficiency,
    summarize,
    worst_stage,
)

NAN = float("nan")


# ---------- 综合性能 ----------

def test_composite_skips_nan():
    assert composite([NAN, 0.4, 0.6]) == pytest.approx(0.5)


def test_composite_empty_is_nan():
    v = composite([])
    assert v != v


def test_composite_all_nan_is_nan():
    v = composite([NAN, NAN])
    assert v != v


# ---------- 最差阶段 ----------

def test_worst_stage_uses_window_mean():
    """单批最小是 0.1，但两批滑窗最小是 (0.1+0.5)/2=0.3。"""
    assert worst_stage([0.9, 0.5, 0.1, 0.5, 0.9], window=2) == pytest.approx(0.3)


def test_worst_stage_window_one_equals_min():
    s = [0.9, 0.2, 0.7]
    assert worst_stage(s, window=1) == pytest.approx(0.2)


def test_worst_stage_window_larger_than_series():
    assert worst_stage([0.4, 0.6], window=10) == pytest.approx(0.5)


def test_worst_stage_single_value():
    assert worst_stage([0.42]) == pytest.approx(0.42)


def test_worst_stage_empty_is_nan():
    v = worst_stage([])
    assert v != v


# ---------- 恢复速度 ----------

def test_recovery_counts_batches_until_back():
    """0.9 后跌到 0.5，两批之后回到 0.9。"""
    events, mean = recovery([0.9, 0.5, 0.6, 0.9], tol=0.05)
    assert events == [2]
    assert mean == pytest.approx(2.0)


def test_recovery_immediate():
    events, _ = recovery([0.9, 0.5, 0.9], tol=0.05)
    assert events == [1]


def test_unrecovered_marked_and_penalized():
    events, mean = recovery([0.9, 0.3, 0.3, 0.3], tol=0.05)
    assert events == [-1]
    assert mean == pytest.approx(4.0)      # 惩罚值 = 批次数


def test_unrecovered_penalty_override():
    _, mean = recovery([0.9, 0.1], tol=0.05, penalty=99)
    assert mean == pytest.approx(99.0)


def test_no_drop_means_no_events():
    events, mean = recovery([0.5, 0.6, 0.7], tol=0.05)
    assert events == [] and mean == 0.0


def test_small_dip_within_tolerance_ignored():
    events, _ = recovery([0.90, 0.87], tol=0.05)
    assert events == []


def test_multiple_recovery_events():
    events, _ = recovery([0.9, 0.5, 0.9, 0.4, 0.95], tol=0.05)
    assert events == [1, 1]


def test_recovery_skips_nan():
    events, _ = recovery([0.9, NAN, 0.5, 0.9], tol=0.05)
    assert events == [1]


# ---------- 抗遗忘 ----------

def test_forgetting_picks_worst_delta():
    assert forgetting([(0.8, 0.79), (0.8, 0.60)]) == pytest.approx(-0.2)


def test_forgetting_positive_when_improving():
    assert forgetting([(0.7, 0.8)]) == pytest.approx(0.1)


def test_forgetting_empty_is_nan():
    v = forgetting([])
    assert v != v


def test_forgetting_ignores_nan_pairs():
    assert forgetting([(NAN, 0.8), (0.8, 0.75)]) == pytest.approx(-0.05)


# ---------- 资源效率 ----------

def test_resource_efficiency_basic():
    t, s = resource_efficiency(gain=0.4, trainings=4, seconds=20.0)
    assert t == pytest.approx(10.0) and s == pytest.approx(50.0)


def test_resource_efficiency_nan_without_gain():
    for gain in (0.0, -0.1, NAN):
        t, s = resource_efficiency(gain, 3, 10.0)
        assert t != t and s != s


# ---------- 汇总 ----------

def test_summarize_fields():
    out = summarize(
        [NAN, 0.5, 0.4, 0.8],
        baseline_composite=0.3,
        trainings=5,
        seconds=60.0,
        replay_pairs=[(0.8, 0.78)],
        promotions=2,
        rollbacks=1,
    )
    assert out.composite == pytest.approx(0.5666666, abs=1e-5)
    assert out.gain_vs_baseline == pytest.approx(0.2666666, abs=1e-5)
    assert out.trainings == 5 and out.promotions == 2 and out.rollbacks == 1
    assert out.forgetting == pytest.approx(-0.02)
    assert out.trainings_per_gain > 0


def test_summarize_recovery_text():
    out = summarize([0.9, 0.3, 0.3])
    assert out.recovery_text() == "未恢复"
    assert summarize([0.5, 0.6]).recovery_text() == "无跌落"


def test_summarize_serializes():
    d = summarize([0.5, 0.6]).to_dict()
    assert "composite" in d and "worst_stage" in d
