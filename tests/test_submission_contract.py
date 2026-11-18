import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
sys.path.insert(0, str(SUBMISSION))

import contest_submission as cs  # noqa: E402


def test_first_round_accepts_valid_platform_candidate_without_parent_metric():
    selected, detail = cs._select_from_platform(
        task="classification",
        metric="macro_f1",
        direction="higher",
        candidate={"macro_f1": 0.8},
        parent={},
        first_round=True,
    )
    assert selected is True
    assert detail["source"] == "platform"
    assert detail["candidate"] == pytest.approx(0.8)


def test_later_round_rejects_candidate_not_better_than_parent():
    selected, detail = cs._select_from_platform(
        task="regression",
        metric="rmse",
        direction="lower",
        candidate={"rmse": 10.1},
        parent={"rmse": 10.0},
        first_round=False,
    )
    assert selected is False
    assert detail["reason"] == "candidate_not_better"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "0.8"])
def test_missing_or_invalid_candidate_metric_is_conservative(value):
    candidate = {} if value is None else {"macro_f1": value}
    selected, detail = cs._select_from_platform(
        task="classification",
        metric="macro_f1",
        direction="higher",
        candidate=candidate,
        parent={"macro_f1": 0.7},
        first_round=False,
    )
    assert selected is False
    assert detail["reason"] == "candidate_metric_missing"


def test_later_round_without_parent_platform_metric_is_conservative():
    selected, detail = cs._select_from_platform(
        task="classification",
        metric="macro_f1",
        direction="higher",
        candidate={"macro_f1": 0.8},
        parent={},
        first_round=False,
    )
    assert selected is False
    assert detail["reason"] == "parent_metric_missing"


def test_c_candidate_with_worse_nasa_is_rejected_even_when_rmse_improves():
    selected, detail = cs._select_from_platform(
        task="regression",
        metric="rmse",
        direction="lower",
        candidate={"rmse": 9.0, "nasa_asymmetric_score": 120.0},
        parent={"rmse": 10.0, "nasa_asymmetric_score": 100.0},
        first_round=False,
        dataset_key="C",
    )
    assert selected is False
    assert detail["reason"] == "nasa_stability_veto"


def test_c_candidate_allows_small_nasa_tradeoff_when_rmse_improves():
    selected, detail = cs._select_from_platform(
        task="regression", metric="rmse", direction="lower",
        candidate={"rmse": 9.0, "nasa_asymmetric_score": 104.9},
        parent={"rmse": 10.0, "nasa_asymmetric_score": 100.0},
        first_round=False, dataset_key="C",
    )
    assert selected is True
    assert detail["reason"] == "candidate_better"


def test_a_candidate_requires_meaningful_f1_gain_and_accuracy_floor():
    parent = {"macro_f1": 0.800, "accuracy": 0.900}
    small_gain, detail = cs._select_from_platform(
        task="classification", metric="macro_f1", direction="higher",
        candidate={"macro_f1": 0.801, "accuracy": 0.901}, parent=parent,
        first_round=False, dataset_key="A")
    assert small_gain is False
    assert detail["reason"] == "gain_below_minimum"

    accuracy_drop, detail = cs._select_from_platform(
        task="classification", metric="macro_f1", direction="higher",
        candidate={"macro_f1": 0.803, "accuracy": 0.897}, parent=parent,
        first_round=False, dataset_key="A")
    assert accuracy_drop is False
    assert detail["reason"] == "accuracy_floor"

    accepted, detail = cs._select_from_platform(
        task="classification", metric="macro_f1", direction="higher",
        candidate={"macro_f1": 0.803, "accuracy": 0.899}, parent=parent,
        first_round=False, dataset_key="A")
    assert accepted is True
    assert detail["reason"] == "candidate_better"


@pytest.mark.parametrize(
    "metric,direction,candidate,parent,expected_gain",
    [
        ("macro_f1", "higher", {"macro_f1": 0.75}, {"macro_f1": 0.80}, -0.0625),
        ("rmse", "lower", {"rmse": 110.0}, {"rmse": 100.0}, -0.10),
    ],
)
def test_platform_metric_drift_detects_relative_degradation(
        metric, direction, candidate, parent, expected_gain):
    drift = cs._platform_metric_drift(metric, direction, candidate, parent)
    assert drift["detector"] == "platform_metric_delta"
    assert drift["enabled"] is True
    assert drift["status"] == "HARMFUL"
    assert drift["drift_detected"] is True
    assert drift["reason"] == "main_metric_degraded"
    assert drift["relative_gain"] == pytest.approx(expected_gain)


def test_platform_metric_drift_handles_stable_and_missing_metrics():
    stable = cs._platform_metric_drift(
        "macro_f1", "higher", {"macro_f1": 0.80}, {"macro_f1": 0.80})
    assert stable["status"] == "STABLE"
    assert stable["drift_detected"] is False
    unknown = cs._platform_metric_drift("rmse", "lower", {}, {"rmse": 100.0})
    assert unknown["status"] == "UNKNOWN"
    assert unknown["reason"] == "metric_missing"
    assert unknown["drift_detected"] is False


def test_platform_metric_drift_detects_c_nasa_worsening():
    drift = cs._platform_metric_drift(
        "rmse", "lower",
        {"rmse": 9.0, "nasa_asymmetric_score": 110.0},
        {"rmse": 10.0, "nasa_asymmetric_score": 100.0},
        dataset_key="C")
    assert drift["status"] == "HARMFUL"
    assert drift["reason"] == "nasa_worsened"
    assert drift["drift_detected"] is True


def test_platform_metric_drift_tolerates_small_c_nasa_tradeoff():
    drift = cs._platform_metric_drift(
        "rmse", "lower",
        {"rmse": 9.0, "nasa_asymmetric_score": 104.9},
        {"rmse": 10.0, "nasa_asymmetric_score": 100.0},
        dataset_key="C")
    assert drift["status"] == "IMPROVED"
    assert drift["drift_detected"] is False


def test_history_metrics_reads_platform_summary_and_configuration_fallback():
    assert cs._history_metrics({"summary": {"metrics": {"rmse": 9.5}}}) == {"rmse": 9.5}
    assert cs._history_metrics({"metrics": {"rmse": 9.6}}) == {"rmse": 9.6}
    assert cs._history_metrics(
        {"configuration": {"platform_metrics": {"rmse": 9.7}}}
    ) == {"rmse": 9.7}


def test_history_prompt_contains_platform_metric_and_configuration():
    history = [{
        "id": "r1",
        "round_index": 1,
        "selected_artifact_id": "parent-1",
        "summary": {"metrics": {"rmse": 9.5}},
        "configuration": {"epochs": 5, "learning_rate": 0.001},
    }]
    prompt = cs._history_prompt(history)
    assert '"rmse": 9.5' in prompt
    assert "epochs=5" in prompt
    assert "lr=0.001" in prompt


def test_history_prompt_contains_formal_platform_drift():
    history = [{
        "id": "r1", "round_index": 1,
        "summary": {"metrics": {"rmse": 10.0}},
        "configuration": {
            "platform_drift": {
                "status": "HARMFUL", "reason": "main_metric_degraded",
                "relative_gain": -0.05,
            }
        },
    }]
    prompt = cs._history_prompt(history)
    assert "drift=HARMFUL" in prompt
    assert "main_metric_degraded" in prompt
    assert "-0.05" in prompt


def test_first_round_loop_evidence_is_complete_with_empty_history():
    evidence = cs._loop_evidence(
        [], ["call-1"], "首轮假设", "candidate_measured", "platform_summary")
    assert evidence["history_read"] is True
    assert evidence["history_ids"] == []
    assert evidence["complete"] is True


def test_complete_sdk_loop_count_uses_structured_formal_evidence():
    history = [
        {"configuration": {"loop_evidence": {"complete": True}}},
        {"configuration": {"loop_evidence": {"complete": True}}},
        {"configuration": {"loop_evidence": {"complete": True}}},
        {"configuration": {"loop_evidence": {"complete": False}}},
    ]
    assert cs._complete_sdk_loop_count(history) == 3


def test_parent_metrics_prefers_matching_parent_artifact():
    history = [
        {"id": "r1", "selected_artifact_id": "old", "summary": {"metrics": {"rmse": 11.0}}},
        {"id": "r2", "selected_artifact_id": "parent-1", "summary": {"metrics": {"rmse": 9.5}}},
    ]
    rnd = type("R", (), {"parent": type("P", (), {
        "artifact_id": "parent-1", "receipt": {"artifact_id": "parent-1"}
    })()})()
    assert cs._parent_metrics(rnd, history) == {"rmse": 9.5}


def test_parent_metrics_uses_task_contract_before_unmatched_history():
    history = [
        {"id": "r1", "selected_artifact_id": "other",
         "summary": {"metrics": {"rmse": 11.0}}},
    ]
    rnd = type("R", (), {
        "parent": type("P", (), {
            "artifact_id": "parent-1", "receipt": {"artifact_id": "parent-1"}
        })(),
        "task": {"parent_metrics": {"rmse": 9.5}},
    })()
    assert cs._parent_metrics(rnd, history) == {"rmse": 9.5}


def test_avoid_repeat_changes_duplicate_configuration():
    history = [{
        "configuration": {
            "parent_artifact_id": "parent-0",
            "epochs": 5,
            "learning_rate": 0.001,
        }
    }]
    assert cs._avoid_repeat("parent-0", 5, 0.001, history) != (5, 0.001)


@pytest.mark.parametrize("round_index", [1, 2])
def test_a_early_round_strategy_uses_strong_known_start(round_index):
    epochs, lr, reason = cs._apply_profile_strategy(
        "A", round_index, 3, 0.0005, [], "parent-a")
    assert (epochs, lr) == (10, 0.01)
    assert "A_early_round" in reason


def test_a_third_round_keeps_qwen_proposal():
    assert cs._apply_profile_strategy(
        "A", 3, 7, 0.002, [], "parent-a") == (7, 0.002, "")


def test_b_stagnation_switches_to_first_untried_axis():
    history = [{
        "configuration": {
            "parent_artifact_id": "parent-b", "epochs": 5,
            "learning_rate": 0.001,
            "platform_decision": {"reason": "candidate_not_better"},
        }
    }]
    epochs, lr, reason = cs._apply_profile_strategy(
        "B", 4, 10, 0.002, history, "parent-b")
    assert (epochs, lr) == (3, 0.0005)
    assert "B_stagnation_switch" in reason


def test_b_improving_previous_round_keeps_qwen_proposal():
    history = [{
        "configuration": {
            "parent_artifact_id": "old-parent", "epochs": 5,
            "learning_rate": 0.001,
            "platform_decision": {"reason": "candidate_better"},
        }
    }]
    assert cs._apply_profile_strategy(
        "B", 4, 7, 0.002, history, "parent-b") == (7, 0.002, "")


def test_best_history_artifact_requires_platform_allowlist():
    history = [{
        "id": "r1", "selected_artifact_id": "history-best",
        "summary": {"metrics": {"rmse": 8.0}},
    }]
    choice = cs._best_allowed_history(
        history, [{"artifact_id": "history-best"}], "rmse", "lower",
        {"rmse": 10.0}, dataset_key="B")
    assert choice == {
        "artifact_id": "history-best", "history_id": "r1",
        "metrics": {"rmse": 8.0},
    }
    assert cs._best_allowed_history(
        history, [{"artifact_id": "other"}], "rmse", "lower",
        {"rmse": 10.0}, dataset_key="B") is None


def test_c_history_artifact_with_worse_nasa_is_not_restored():
    history = [{
        "id": "r1", "selected_artifact_id": "unsafe-history",
        "summary": {"metrics": {
            "rmse": 8.0, "nasa_asymmetric_score": 110.0,
        }},
    }]
    assert cs._best_allowed_history(
        history, [{"artifact_id": "unsafe-history"}], "rmse", "lower",
        {"rmse": 10.0, "nasa_asymmetric_score": 100.0},
        dataset_key="C") is None


class _Parent:
    artifact_id = "parent-1"
    receipt = {"artifact_id": "parent-1"}

    @staticmethod
    def path(relative):
        assert relative == "checkpoint.npz"
        return Path("parent-checkpoint.npz")


class _Data:
    train_path = "train.jsonl"
    validation_path = "validation.jsonl"


class _Round:
    def __init__(self, history, summary, round_index=2, artifacts=None):
        self.dataset_key = "A"
        self.round_index = round_index
        self.max_output_tokens = 512
        self.task = {"task": "classification", "metric": "macro_f1",
                     "metric_direction": "higher"}
        self.parent = _Parent()
        self.data = _Data()
        self.call_ids = []
        self._history = history
        self._summary = summary
        self.artifacts = artifacts or []
        self.finished = None

    def history(self):
        return self._history

    def model(self, messages, **options):
        assert options["response_format"] == {"type": "json_object"}
        assert options["max_tokens"] <= 256
        self.call_ids.append("call-1")
        return type("Receipt", (), {
            "call_id": "call-1",
            "text": '{"epochs": 2, "learning_rate": 0.003, "hypothesis": "test"}',
        })()

    def submit_candidate(self, directory, **kwargs):
        return {"artifact_id": "candidate-1", "summary": {"metrics": self._summary}}

    def finish(self, **kwargs):
        self.finished = kwargs


def _fake_training(monkeypatch, tmp_path, local_promote):
    def run(*args, **kwargs):
        cand = tmp_path / "cand"
        cand.mkdir()
        (cand / "checkpoint.npz").write_bytes(b"checkpoint")
        (cand / "training.json").write_text(json.dumps({
            "seed": 42,
            "validation_before": {"macro_f1": 0.70},
            "validation_after": {"macro_f1": 0.99},
        }), encoding="utf-8")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    monkeypatch.setattr(cs.subprocess, "run", run)
    monkeypatch.setattr(cs, "_decide_promote", lambda *a, **k: (local_promote, {"p": 1.0}))


def test_train_rejects_platform_regression_even_if_local_gate_promotes(monkeypatch, tmp_path):
    history = [{
        "id": "r1", "selected_artifact_id": "parent-1",
        "summary": {"metrics": {"macro_f1": 0.80}},
        "configuration": {"epochs": 5, "learning_rate": 0.001},
    }]
    rnd = _Round(history, {"macro_f1": 0.79})
    _fake_training(monkeypatch, tmp_path, local_promote=True)
    cs.train(rnd)
    assert rnd.finished["selected_artifact_id"] == "parent-1"
    assert rnd.finished["configuration"]["platform_metrics"] == {"macro_f1": 0.79}
    assert "平台" in rnd.finished["reason"]


def test_train_promotes_platform_improvement_even_if_local_gate_rejects(monkeypatch, tmp_path):
    history = [{
        "id": "r1", "selected_artifact_id": "parent-1",
        "summary": {"metrics": {"macro_f1": 0.80}},
        "configuration": {"epochs": 5, "learning_rate": 0.001},
    }]
    rnd = _Round(history, {"macro_f1": 0.81})
    _fake_training(monkeypatch, tmp_path, local_promote=False)
    cs.train(rnd)
    assert rnd.finished["selected_artifact_id"] == "candidate-1"
    assert rnd.finished["configuration"]["local_diagnostic"] == {"p": 1.0}
    assert rnd.finished["configuration"]["loop_evidence"]["complete"] is True
    assert (rnd.finished["configuration"]["loop_evidence"]["experiment_status"]
            == "candidate_measured")
    assert rnd.finished["configuration"]["platform_drift"]["enabled"] is True


def test_train_failure_records_complete_formal_loop(monkeypatch, tmp_path):
    history = [{
        "id": "r1", "selected_artifact_id": "parent-1",
        "summary": {"metrics": {"macro_f1": 0.80}},
        "configuration": {"epochs": 5, "learning_rate": 0.001},
    }]
    rnd = _Round(history, {"macro_f1": 0.81})
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    monkeypatch.setattr(
        cs.subprocess, "run",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 1, "stderr": "training failed"})(),
    )
    cs.train(rnd)
    evidence = rnd.finished["configuration"]["loop_evidence"]
    assert evidence["complete"] is True
    assert evidence["experiment_status"] == "train_failed"
    assert evidence["decision_source"] == "failure_fallback"
    assert rnd.finished["configuration"]["platform_drift"]["status"] == "UNKNOWN"


def test_train_can_restore_allowed_platform_history_best(monkeypatch, tmp_path):
    history = [
        {
            "id": "r1", "selected_artifact_id": "history-best",
            "summary": {"metrics": {"macro_f1": 0.85}},
            "configuration": {"epochs": 5, "learning_rate": 0.001},
        },
        {
            "id": "r2", "selected_artifact_id": "parent-1",
            "summary": {"metrics": {"macro_f1": 0.80}},
            "configuration": {"epochs": 5, "learning_rate": 0.001},
        },
    ]
    rnd = _Round(
        history, {"macro_f1": 0.81}, round_index=3,
        artifacts=[{"artifact_id": "history-best"}, {"artifact_id": "parent-1"}],
    )
    _fake_training(monkeypatch, tmp_path, local_promote=True)
    cs.train(rnd)
    assert rnd.finished["selected_artifact_id"] == "history-best"
    assert rnd.finished["configuration"]["selected_source"] == "history"
    assert rnd.finished["configuration"]["selected_history_id"] == "r1"


def test_formal_dockerfile_and_runbook_use_sdk_submission_image():
    dockerfile = (ROOT / "submission" / "Dockerfile").read_text(encoding="utf-8")
    runbook = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "COPY submission/decision.py /app/decision.py" in dockerfile
    assert 'ENTRYPOINT ["python3", "/app/harness.py"]' in dockerfile
    assert "-f submission/Dockerfile" in runbook
