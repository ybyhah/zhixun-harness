import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.policy import AdaptivePolicy, RoundPlan, expand_candidates


def h(rnd, value, accepted, lr, epochs=5, parent="p0", trained=True):
    return {"round": rnd, "value": value, "accepted": accepted, "lr": lr,
            "epochs": epochs, "parent": parent, "trained": trained}


# ---------- RoundPlan.skip ----------

def test_roundplan_skip_defaults_false():
    assert RoundPlan(5).skip is False and RoundPlan(5).reason == ""


def test_expand_candidates_does_not_expand_skipped_plan():
    base = RoundPlan(5, 0.003, skip=True, reason="已试过")
    assert expand_candidates(base, 3) == [base]


# ---------- 首轮 ----------

def test_first_round_uses_base_and_never_skips():
    p = AdaptivePolicy(base_epochs=5, base_lr=0.003)
    plan = p.decide([])
    assert plan.epochs == 5 and plan.learning_rate == 0.003 and not plan.skip


def test_first_round_not_skipped_even_if_history_has_untrained():
    p = AdaptivePolicy(base_epochs=5, base_lr=0.003)
    plan = p.decide([h(1, float("nan"), False, 0.003, trained=False)])
    assert not plan.skip


# ---------- 有进展 ----------

def test_keeps_hyperparams_after_promotion():
    p = AdaptivePolicy(base_epochs=5, base_lr=0.004)
    p.decide([])
    plan = p.decide([h(1, 0.80, True, 0.004)])
    assert plan.learning_rate == 0.004 and plan.epochs == 5
    assert "保持超参" in plan.hypothesis


# ---------- 被拒但有变化：减半 ----------

def test_halves_lr_when_rejected_with_change():
    p = AdaptivePolicy(base_epochs=5, base_lr=0.004)
    p.decide([])
    plan = p.decide([h(1, 0.70, False, 0.004)])
    assert plan.learning_rate == pytest.approx(0.002)
    assert "降学习率" in plan.hypothesis


# ---------- 停滞 → 换轴 ----------

def test_stall_switches_axis_to_epochs():
    p = AdaptivePolicy(base_epochs=5, base_lr=0.004)
    p.decide([])
    hist = [h(1, 0.7183, False, 0.002), h(2, 0.7183, False, 0.001)]
    plan = p.decide(hist)
    assert plan.epochs == 10                      # epochs ×2，不再死磕 lr
    assert plan.learning_rate >= 0.001            # lr 回弹，不继续减半
    assert "无效区" in plan.hypothesis and "换轴" in plan.hypothesis


def test_no_stall_when_values_differ():
    p = AdaptivePolicy(base_epochs=5, base_lr=0.004)
    p.decide([])
    plan = p.decide([h(1, 0.70, False, 0.004), h(2, 0.71, False, 0.002)])
    assert plan.epochs == 5                       # 仍有变化 → 不换轴
    assert plan.learning_rate == pytest.approx(0.001)


def test_epochs_axis_capped_at_max():
    p = AdaptivePolicy(base_epochs=40, base_lr=0.004, max_epochs=50)
    p.decide([])
    hist = [h(1, 0.7, False, 0.002, epochs=40), h(2, 0.7, False, 0.001, epochs=40)]
    plan = p.decide(hist)
    assert plan.epochs == 50                      # 触顶后裁剪


# ---------- 去重 → NOOP ----------

def test_repeated_config_becomes_noop():
    """去重是安全网：若策略提出的配置在同一父检查点下已试过，判 NOOP 省算力。"""
    p = AdaptivePolicy(base_epochs=5, base_lr=0.004)
    p.next_parent = "p0"
    p.decide([])                                   # 记下 (p0, 5, 0.004)
    # 上一轮被拒且有变化 → 策略会提 lr 减半到 0.002；先把它预置成"已试过"
    p.tried.add(p._key("p0", 5, 0.002))
    plan = p.decide([h(1, 0.70, False, 0.004, parent="p0")])
    assert plan.skip and "已试过" in plan.reason
    assert plan.learning_rate == pytest.approx(0.002)


def test_new_parent_is_not_deduped():
    p = AdaptivePolicy(base_epochs=5, base_lr=0.004)
    p.next_parent = "p0"
    p.decide([])
    p.tried.add(p._key("p0", 5, 0.002))
    p.next_parent = "p1"                           # 父检查点变了 → 不算重复
    plan = p.decide([h(1, 0.70, False, 0.004, parent="p0")])
    assert not plan.skip


# ---------- 合法性 ----------

def test_returned_hyperparams_always_legal():
    p = AdaptivePolicy(base_epochs=5, base_lr=1e-6, min_lr=1e-9)
    p.decide([])
    hist = [h(1, 0.5, False, 1e-6)]
    for _ in range(12):
        plan = p.decide(hist)
        assert 1 <= plan.epochs <= 50
        assert plan.learning_rate is None or 1e-6 <= plan.learning_rate <= 0.1
        hist.append(h(len(hist) + 1, 0.5, False, plan.learning_rate,
                      epochs=plan.epochs))


def test_lr_none_is_tolerated():
    p = AdaptivePolicy(base_epochs=5, base_lr=None)
    p.decide([])
    plan = p.decide([h(1, 0.5, False, None)])
    assert plan.learning_rate is None and 1 <= plan.epochs <= 50
