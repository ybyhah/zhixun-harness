import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver import trainer
from driver.gate import OfficialGate
from driver.metric import round_value, to_score
from driver.package import load_package
from driver.policy import RulePolicy

PKG_A = ROOT / "base" / "cpu-a-guide-20261110-v1"
PKG_B = ROOT / "base" / "cpu-b-rul-20261110-v1"
has_a = PKG_A.is_dir()
has_b = PKG_B.is_dir()


# ---------- 包定位 ----------

@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_load_package_a():
    pkg = load_package(PKG_A)
    assert pkg.profile == "A" and pkg.task == "classification"
    assert pkg.higher_better and pkg.metric_name == "macro_f1"
    assert pkg.parameter_count == 4326
    assert pkg.train_jsonl.is_file() and pkg.validation_jsonl.is_file()


@pytest.mark.skipif(not has_b, reason="需要 base/B 包")
def test_load_package_b_is_regression():
    pkg = load_package(PKG_B)
    assert pkg.profile == "B" and pkg.task == "regression"
    assert not pkg.higher_better and pkg.metric_name == "rmse"


# ---------- 指标方向 ----------

@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_metric_direction():
    a = load_package(PKG_A)
    assert to_score(a, 0.7) == 0.7                      # 分类越大越好
    assert round_value(a, {"macro_f1": 0.73, "accuracy": 0.9}) == 0.73


@pytest.mark.skipif(not has_b, reason="需要 base/B 包")
def test_metric_direction_regression():
    b = load_package(PKG_B)
    assert to_score(b, 681.0) == -681.0                 # 回归 -RMSE
    assert round_value(b, {"rmse": 700.0, "mae": 560.0}) == 700.0


# ---------- 门控 ----------

@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_gate_classification():
    pkg = load_package(PKG_A)
    gate = OfficialGate(pkg, baseline_value=0.714, min_gain=0.0)
    assert gate.evaluate(0.73, 0.72).accepted            # 超基线且超最优
    assert not gate.evaluate(0.70, 0.72).accepted        # 低于基线被拒
    assert not gate.evaluate(0.721, 0.73).accepted       # 不及最优被拒


@pytest.mark.skipif(not has_b, reason="需要 base/B 包")
def test_gate_regression_lower_is_better():
    pkg = load_package(PKG_B)
    gate = OfficialGate(pkg, baseline_value=681.0, min_gain=0.0)
    assert gate.evaluate(666.0, 670.0).accepted          # RMSE 更低更好
    assert not gate.evaluate(690.0, 670.0).accepted      # RMSE 变高被拒


# ---------- 策略 ----------

def test_policy_first_round_uses_base():
    plan = RulePolicy(base_epochs=5, base_lr=0.003).decide([])
    assert plan.epochs == 5 and plan.learning_rate == 0.003


def test_policy_halves_lr_after_reject():
    p = RulePolicy(base_epochs=5, base_lr=0.004)
    plan = p.decide([{"round": 1, "value": 0.7, "accepted": False, "lr": 0.004}])
    assert plan.learning_rate == pytest.approx(0.002)


def test_policy_keeps_lr_after_accept():
    p = RulePolicy(base_epochs=5, base_lr=0.004)
    plan = p.decide([{"round": 1, "value": 0.8, "accepted": True, "lr": 0.004}])
    assert plan.learning_rate == 0.004


# ---------- trainer 命令拼装（mock 子进程） ----------

@pytest.mark.skipif(not has_b, reason="需要 base/B 包")
def test_trainer_builds_resume_and_provenance(monkeypatch, tmp_path):
    pkg = load_package(PKG_B)
    captured = {}

    class FakeProc:
        returncode = 0
        stdout = json.dumps({"validation_after": {"rmse": 670.0, "mae": 560.0}})
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["cwd"] = kw.get("cwd")
        out = Path([cmd[i + 1] for i, a in enumerate(cmd) if a == "--output"][0])
        out.mkdir(parents=True)
        (out / "training.json").write_text(FakeProc.stdout)
        return FakeProc()

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)
    prev = tmp_path / "prev.npz"
    prev.write_bytes(b"x")
    rr = trainer.run_round(pkg, tmp_path / "r2", resume=prev, epochs=5, stage_id="round-2")
    assert rr.ok and rr.validation_after["rmse"] == 670.0
    assert "--resume" in captured["cmd"]
    assert "--label-provenance" in captured["cmd"]   # B 必带
    assert captured["cwd"] == str(pkg.root)


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_trainer_first_round_uses_from_baseline(monkeypatch, tmp_path):
    pkg = load_package(PKG_A)
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        out = Path([cmd[i + 1] for i, a in enumerate(cmd) if a == "--output"][0])
        out.mkdir(parents=True)
        (out / "training.json").write_text(json.dumps({"validation_after": {"macro_f1": 0.72}}))

        class P:
            returncode = 0
            stdout = "{}"
            stderr = ""
        return P()

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)
    rr = trainer.run_round(pkg, tmp_path / "r1", resume=None, epochs=5, stage_id="round-1")
    assert rr.ok and "--from-baseline" in captured["cmd"]
    assert "--label-provenance" not in captured["cmd"]   # A 不需要


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_trainer_reports_failure_on_nonzero(monkeypatch, tmp_path):
    pkg = load_package(PKG_A)

    def fake_run(cmd, **kw):
        class P:
            returncode = 1
            stdout = ""
            stderr = "boom\nlast line"
        return P()

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)
    rr = trainer.run_round(pkg, tmp_path / "r", resume=None, epochs=5, stage_id="s")
    assert not rr.ok and "last line" in rr.error


# ---------- 多候选扩展 ----------

from driver.policy import RoundPlan, expand_candidates


def test_expand_k1_returns_base():
    base = RoundPlan(5, 0.003, hypothesis="h")
    out = expand_candidates(base, 1)
    assert out == [base]


def test_expand_respects_limit_and_dedups():
    base = RoundPlan(5, 0.004)
    out = expand_candidates(base, 3)
    assert len(out) == 3
    assert len({(p.epochs, p.learning_rate) for p in out}) == 3
    assert out[0].epochs == 5 and out[0].learning_rate == 0.004     # base 居首
    assert any(abs(p.learning_rate - 0.002) < 1e-12 for p in out)  # 含 lr×0.5


def test_expand_clamps_to_official_bounds():
    out = expand_candidates(RoundPlan(1, 0.09), 5)
    assert all(1 <= p.epochs <= 50 for p in out)
    assert all(p.learning_rate <= 0.1 for p in out)


def test_expand_lr_none_only_perturbs_epochs():
    out = expand_candidates(RoundPlan(5, None), 4)
    assert all(p.learning_rate is None for p in out)
    assert {p.epochs for p in out} >= {5, 3, 7}


# ---------- 多候选择优（mock 训练） ----------

@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_driver_multicandidate_picks_best_and_counts_trainings(monkeypatch, tmp_path):
    from driver import loop as loop_mod
    from driver.loop import OfficialDriver
    from driver.trainer import RoundResult

    pkg = load_package(PKG_A)
    # 按学习率给定验证分：lr 较小者更好；用来验证"择优"
    def fake_run_round(pkg_, out, *, resume, epochs, stage_id, learning_rate=None,
                       batch_size=None, timeout_s=0, cpu_threads=1, **kw):
        out.mkdir(parents=True, exist_ok=True)
        ck = out / "checkpoint.npz"
        ck.write_bytes(b"x")
        lr = learning_rate if learning_rate is not None else 0.003
        macro = 0.80 - lr * 10          # lr 越小分越高
        return RoundResult(stage_id, out, ck, {"validation_after": {"macro_f1": macro}}, ok=True)

    monkeypatch.setattr(loop_mod, "run_round", fake_run_round)
    drv = OfficialDriver(pkg, tmp_path / "r", rounds=2, candidates=3,
                         policy=RulePolicy(base_epochs=5, base_lr=0.004),
                         audit_path=tmp_path / "a.jsonl")
    res = drv.run()
    assert res.trainings == 6                       # 2 轮 × 3 候选
    first = res.rows[0]
    assert first.accepted                           # 选出的最优过了基线门控
    # 最优候选应是最小 lr（0.002），分 = 0.80-0.02
    assert first.value == pytest.approx(0.78)
    audit = drv.audit.records[0]
    assert len(audit.experiments) == 3
    assert sum(1 for e in audit.experiments if e["selected"]) == 1


@pytest.mark.skipif(not has_a, reason="需要 base/A 包")
def test_driver_multicandidate_survives_partial_failure(monkeypatch, tmp_path):
    from driver import loop as loop_mod
    from driver.loop import OfficialDriver
    from driver.trainer import RoundResult

    pkg = load_package(PKG_A)
    calls = {"n": 0}

    def fake_run_round(pkg_, out, *, resume, epochs, stage_id, learning_rate=None,
                       batch_size=None, timeout_s=0, cpu_threads=1, **kw):
        calls["n"] += 1
        out.mkdir(parents=True, exist_ok=True)
        ck = out / "checkpoint.npz"
        ck.write_bytes(b"x")
        if calls["n"] % 3 == 1:                     # 每轮首个候选失败
            return RoundResult(stage_id, out, ck, {}, ok=False, error="boom")
        return RoundResult(stage_id, out, ck, {"validation_after": {"macro_f1": 0.78}}, ok=True)

    monkeypatch.setattr(loop_mod, "run_round", fake_run_round)
    drv = OfficialDriver(pkg, tmp_path / "r", rounds=2, candidates=3,
                         policy=RulePolicy(base_epochs=5, base_lr=0.004),
                         audit_path=tmp_path / "a.jsonl")
    res = drv.run()
    assert len(res.failures) == 2                   # 每轮 1 个候选失败，其余照常
    assert res.rows[0].accepted
