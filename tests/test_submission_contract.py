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
    def __init__(self, history, summary, round_index=2):
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


def test_formal_dockerfile_and_runbook_use_sdk_submission_image():
    dockerfile = (ROOT / "submission" / "Dockerfile").read_text(encoding="utf-8")
    runbook = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "COPY submission/decision.py /app/decision.py" in dockerfile
    assert 'ENTRYPOINT ["python3", "/app/harness.py"]' in dockerfile
    assert "-f submission/Dockerfile" in runbook
