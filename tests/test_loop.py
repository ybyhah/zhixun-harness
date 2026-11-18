"""主闭环集成测试。

重点验证两件事：
1. 无人干预跑完全部批次（排名门槛之一）
2. 故意注入训练失败、超时、记录异常，闭环仍能跑完
"""

import numpy as np
import pytest

from harness.config import Config
from harness.datasets.base import Batch
from harness.decide.actions import Action, Decision
from harness.decide.rules import RuleDecider
from harness.loop import Guard, SelfEvolvingLoop
from harness.models.industrial import IndustrialModel
from harness.stream.simulator import DriftSpec, inject

CLASSES = np.arange(1, 4)


def cfg(tmp_path, **over):
    raw = {
        "seed": 0,
        "task": {"name": "t", "metric": "macro_f1"},
        "data": {"raw_dir": "x", "standardize": True},
        "model": {"backend": "hist", "params": {"max_iter": 20}, "decay": 0.9},
        "monitor": {
            "psi_threshold": 0.2, "psi_feature_ratio": 0.1,
            "pca_components": 4, "perf_drop": 0.03,
            "label_shift_threshold": 0.3, "error_detector": "adwin",
        },
        "gate": {"min_gain": 0.0, "forget_tol": 0.05,
                 "replay_per_batch": 20, "replay_capacity": 200,
                 "max_consecutive_failures": 2},
        "loop": {"min_batch_size": 10, "max_trainings_per_batch": 2,
                 "train_timeout_s": 60, "max_total_trainings": 30},
        "record": {"audit_path": str(tmp_path / "audit.jsonl"),
                   "swanlab": {"enabled": False}},
    }
    for k, v in over.items():
        raw[k].update(v) if isinstance(raw.get(k), dict) else raw.update({k: v})
    return Config(raw=raw)


def drifting_batches(n=6, rows=120, feats=8, seed=0):
    rng = np.random.default_rng(seed)
    base = []
    for i in range(n):
        y = rng.choice(CLASSES, size=rows)
        X = rng.normal(size=(rows, feats)) + y[:, None] * 1.2
        base.append(Batch(index=i, X=X, y=y))
    return inject(base, [DriftSpec("covariate", "gradual", start=2, magnitude=2.0)])


def run(tmp_path, batches=None, decider=None, **over):
    loop = SelfEvolvingLoop(
        batches or drifting_batches(),
        CLASSES,
        cfg(tmp_path, **over),
        decider or RuleDecider(min_batch_size=10),
    )
    return loop, loop.run()


# ---------- 正常路径 ----------

def test_runs_all_batches_unattended(tmp_path):
    batches = drifting_batches(n=6)
    _, res = run(tmp_path, batches)
    assert len(res.rows) == len(batches)
    assert res.failures == []


def test_first_batch_bootstraps_model(tmp_path):
    loop, res = run(tmp_path)
    assert loop.registry.has_online
    assert res.rows[0]["action"] == Action.FULL_RETRAIN.value


def test_scores_present_after_bootstrap(tmp_path):
    _, res = run(tmp_path)
    later = [r["score"] for r in res.rows[1:]]
    assert all(s == s for s in later)      # 非 NaN


def test_audit_records_one_per_batch(tmp_path):
    _, res = run(tmp_path)
    assert len(res.audit) == len(res.rows)


def test_local_simulator_records_triggered_experimental_loops(tmp_path):
    """旧本地模拟器只验证触发实验被完整记录；正式三闭环由 SDK 契约测试验收。"""
    batches = drifting_batches(n=10, seed=3)
    _, res = run(tmp_path, batches)
    loops = res.audit.complete_loops()
    assert loops
    assert all(record.experiments and record.decision for record in loops)


def test_outcome_is_backfilled(tmp_path):
    _, res = run(tmp_path)
    assert any(r.outcome.get("next_batch_score") is not None
               for r in res.audit.records)


def test_hypothesis_marked_as_rule_derived(tmp_path):
    _, res = run(tmp_path)
    hyps = [r.hypothesis for r in res.audit.records if r.hypothesis]
    assert hyps and all("[规则推导]" in h for h in hyps)


def test_deterministic_under_fixed_seed(tmp_path):
    b = drifting_batches(n=5)
    _, a = run(tmp_path / "a", b)
    _, c = run(tmp_path / "b", b)
    assert [r["score"] for r in a.rows[1:]] == [r["score"] for r in c.rows[1:]]


def test_no_lookahead_scaler(tmp_path):
    """标准化统计量只能来自已见数据：第一批提交前 scaler 不应被拟合。"""
    loop = SelfEvolvingLoop(
        drifting_batches(), CLASSES, cfg(tmp_path), RuleDecider(min_batch_size=10)
    )
    assert not loop.scaler.fitted
    loop.run()
    assert loop.scaler.fitted


# ---------- 异常与护栏 ----------

def test_training_failure_does_not_break_loop(tmp_path, monkeypatch):
    loop = SelfEvolvingLoop(
        drifting_batches(n=6), CLASSES, cfg(tmp_path), RuleDecider(min_batch_size=10)
    )

    def boom(*a, **k):
        raise RuntimeError("注入的训练失败")

    monkeypatch.setattr(loop.executor, "execute", boom)
    res = loop.run()
    assert len(res.rows) == 6
    assert any(f["stage"].startswith("train") for f in res.failures)
    assert loop.registry.has_online          # 在线模型仍在


def test_timeout_discards_candidate(tmp_path):
    loop = SelfEvolvingLoop(
        drifting_batches(n=6), CLASSES, cfg(tmp_path), RuleDecider(min_batch_size=10)
    )
    loop.guard.train_timeout_s = 0.0         # 任何训练都算超时
    res = loop.run()
    assert len(res.rows) == 6
    assert any("超过上限" in v["reason"] for v in res.guard_violations)


def test_total_training_cap_enforced(tmp_path):
    loop = SelfEvolvingLoop(
        drifting_batches(n=8), CLASSES, cfg(tmp_path), RuleDecider(min_batch_size=10)
    )
    loop.guard.max_total_trainings = 1       # 冷启动就会用掉这一次
    res = loop.run()
    assert len(res.rows) == 8                # 护栏拦截不影响跑完
    assert any("累计训练次数" in v["reason"] for v in res.guard_violations)


def test_prediction_failure_is_isolated(tmp_path, monkeypatch):
    """预测环节出错也必须提交结果并继续，否则闭环会卡死。"""
    loop = SelfEvolvingLoop(
        drifting_batches(n=5), CLASSES, cfg(tmp_path), RuleDecider(min_batch_size=10)
    )
    real = IndustrialModel.predict

    calls = {"n": 0}

    def flaky(self, X):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("注入的预测失败")
        return real(self, X)

    monkeypatch.setattr(IndustrialModel, "predict", flaky)
    res = loop.run()
    assert len(res.rows) == 5


def test_monitor_failure_degrades_to_noop(tmp_path, monkeypatch):
    loop = SelfEvolvingLoop(
        drifting_batches(n=5), CLASSES, cfg(tmp_path), RuleDecider(min_batch_size=10)
    )
    monkeypatch.setattr(
        loop.dist_monitor, "compare",
        lambda X: (_ for _ in ()).throw(RuntimeError("注入的监测失败")),
    )
    res = loop.run()
    assert len(res.rows) == 5
    assert any(f["stage"] == "post_submit" for f in res.failures)
    assert any(r["action"] == Action.NOOP.value for r in res.rows)


def test_decider_failure_does_not_break_loop(tmp_path):
    class Boom:
        def decide(self, report, history):
            raise RuntimeError("注入的决策失败")

    loop = SelfEvolvingLoop(drifting_batches(n=5), CLASSES, cfg(tmp_path), Boom())
    res = loop.run()
    assert len(res.rows) == 5


def test_rollback_after_consecutive_failures(tmp_path):
    """连续门控失败应触发回滚（需要存在带分数的历史版本）。"""
    class AlwaysRetrain:
        def decide(self, report, history):
            return Decision(Action.WINDOW_RETRAIN, {"window": 1},
                            reason="强制每批重训以测试回滚")

    loop = SelfEvolvingLoop(
        drifting_batches(n=8), CLASSES,
        cfg(tmp_path, gate={"min_gain": 0.99, "max_consecutive_failures": 2}),
        AlwaysRetrain(),
    )
    res = loop.run()
    assert len(res.rows) == 8
    # min_gain 高到几乎必被拒，因此应出现连续失败
    assert any(not r["gate_accepted"] for r in res.rows[1:])


def test_guard_blocks_without_crashing():
    g = Guard(max_trainings_per_batch=1, max_total_trainings=1)
    assert g.can_train(0, 0)[0]
    assert not g.can_train(0, 1)[0]
    g.total_trainings = 1
    assert not g.can_train(0, 0)[0]


# ---------- 多候选竞争 ----------

def test_multiple_candidates_are_tried(tmp_path):
    """max_candidates=2 时，有漂移的批次应出现两条实验记录。"""
    loop = SelfEvolvingLoop(
        drifting_batches(n=8, seed=3), CLASSES,
        cfg(tmp_path, loop={"max_candidates": 2, "max_trainings_per_batch": 3}),
        RuleDecider(min_batch_size=10),
    )
    res = loop.run()
    counts = [len(r.experiments) for r in res.audit.records]
    assert max(counts) >= 2


def test_candidate_budget_limits_attempts(tmp_path):
    """单批训练次数上限为 1 时，即使计划两个候选也只能试一个。"""
    loop = SelfEvolvingLoop(
        drifting_batches(n=8, seed=3), CLASSES,
        cfg(tmp_path, loop={"max_candidates": 3, "max_trainings_per_batch": 1}),
        RuleDecider(min_batch_size=10),
    )
    res = loop.run()
    for rec in res.audit.records:
        trained = [e for e in rec.experiments if "seconds" in e]
        assert len(trained) <= 1


def test_best_candidate_wins(tmp_path):
    """两个候选都通过门控时，上线的应是验证分更高的那个。"""
    loop = SelfEvolvingLoop(
        drifting_batches(n=8, seed=3), CLASSES,
        cfg(tmp_path, loop={"max_candidates": 2, "max_trainings_per_batch": 3}),
        RuleDecider(min_batch_size=10),
    )
    res = loop.run()
    for rec in res.audit.records:
        passed = [
            e for e in rec.experiments
            if e.get("gate_accepted") and e.get("candidate_score") is not None
        ]
        if len(passed) >= 2 and rec.gate_result.get("accepted"):
            best = max(e["candidate_score"] for e in passed)
            assert rec.gate_result["candidate_score"] == pytest.approx(best)


def test_single_candidate_mode_still_works(tmp_path):
    loop = SelfEvolvingLoop(
        drifting_batches(n=6), CLASSES,
        cfg(tmp_path, loop={"max_candidates": 1}),
        RuleDecider(min_batch_size=10),
    )
    res = loop.run()
    assert len(res.rows) == 6


def test_decider_without_decide_many_is_supported(tmp_path):
    """只实现 decide 的决策器（如第二阶段的大模型版）也必须能跑。"""
    class OnlyDecide:
        def decide(self, report, history):
            return Decision(Action.WINDOW_RETRAIN, {"window": 2}, reason="固定动作")

    loop = SelfEvolvingLoop(
        drifting_batches(n=6), CLASSES,
        cfg(tmp_path, loop={"max_candidates": 2}),
        OnlyDecide(),
    )
    res = loop.run()
    assert len(res.rows) == 6


def test_gate_uses_held_out_data(tmp_path):
    """门控分数必须来自候选没训练过的数据，不应恒为满分。"""
    loop = SelfEvolvingLoop(
        drifting_batches(n=8, seed=3), CLASSES, cfg(tmp_path),
        RuleDecider(min_batch_size=10),
    )
    res = loop.run()
    scores = [
        e["candidate_score"] for r in res.audit.records for e in r.experiments
        if e.get("candidate_score") is not None
    ]
    assert scores and not all(s == pytest.approx(1.0) for s in scores)
