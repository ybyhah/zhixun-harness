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


def test_history_prompt_defaults_to_latest_three_but_can_include_all_rounds():
    history = [
        {"id": f"r{i}", "round_index": i,
         "summary": {"metrics": {"rmse": float(i)}}, "configuration": {}}
        for i in range(1, 6)
    ]
    default_prompt = cs._history_prompt(history)
    assert "轮1:" not in default_prompt and "轮2:" not in default_prompt
    assert all(f"轮{i}:" in default_prompt for i in (3, 4, 5))

    full_prompt = cs._history_prompt(history, include_all=True)
    assert all(f"轮{i}:" in full_prompt for i in range(1, 6))


def test_parse_model_reply_recovers_numbers_from_truncated_json():
    data, truncated = cs._parse_model_reply(
        '{"epochs": 12, "learning_rate": 0.0008, "hypothesis": "学习率减半以')
    assert truncated is True
    assert data == {"epochs": 12.0, "learning_rate": 0.0008}
    data, truncated = cs._parse_model_reply('{"learning_rate": 1e-4}')
    assert truncated is False and data["learning_rate"] == 1e-4
    with pytest.raises(ValueError):
        cs._parse_model_reply('{"hypothesis": "没有数字')


def test_ask_model_uses_truncated_reply_numbers_and_compact_c_prompt():
    history = [{
        "id": f"r{i}", "round_index": i, "selected_artifact_id": "parent-1",
        "configuration": {
            "parent_artifact_id": "parent-1", "learning_rate": 0.0001 * i,
            "selected_source": "parent",
            "platform_metrics": {"rmse": 50.0 + i, "mae": 40.0,
                                 "nasa_asymmetric_score": 9.0},
        },
    } for i in range(1, 6)]
    rnd = _Round(history, {}, round_index=6)
    rnd.dataset_key = "C"
    rnd.task = {"task": "regression", "metric": "rmse", "metric_direction": "lower"}
    rnd.reply = '{"learning_rate": 0.00033, "hypothesis": "截'
    epochs, lr, _, hypothesis, _ = cs._ask_model(rnd, "regression", "rmse", history)
    assert (epochs, lr) == (1, 0.00033)
    assert "截断" in hypothesis
    prompt = rnd.prompts[0]
    assert "轮1:" not in prompt and "轮5:" in prompt
    assert '"mae"' not in prompt and "nasa_asymmetric_score" in prompt
    assert "已试过的 lr" in prompt and "0.0001" in prompt


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


def test_resource_skip_requires_three_loops_two_rejections_and_no_harmful_drift():
    history = []
    for index in range(1, 4):
        history.append({
            "id": f"r{index}", "selected_artifact_id": "parent-1",
            "configuration": {
                "parent_artifact_id": "parent-1",
                "loop_evidence": {"complete": True},
                "platform_decision": {"reason": "candidate_not_better"},
                "platform_drift": {"status": "STABLE"},
            },
        })
    assert cs._should_skip_training(history, "parent-1") is True

    history[-1]["configuration"]["resource_action"] = "training_skipped"
    assert cs._should_skip_training(history, "parent-1") is False
    history[-1]["configuration"].pop("resource_action")
    history[-1]["configuration"]["platform_drift"]["status"] = "HARMFUL"
    assert cs._should_skip_training(history, "parent-1") is False


def test_c_resource_policy_skips_after_two_stable_rejections():
    history = [{
        "id": f"r{i}", "selected_artifact_id": "parent-c",
        "configuration": {
            "parent_artifact_id": "parent-c",
            "loop_evidence": {"complete": True},
            "platform_decision": {"reason": "candidate_not_better"},
            "platform_drift": {"status": "STABLE"},
        },
    } for i in range(1, 4)]
    assert cs._should_skip_training(
        history, "parent-c", dataset_key="C") is True


def test_c_skips_when_harmful_round_is_followed_by_another_rejection():
    history = [{
        "id": f"r{i}", "selected_artifact_id": "parent-c",
        "configuration": {
            "parent_artifact_id": "parent-c",
            "loop_evidence": {"complete": True},
            "platform_decision": {"reason": "nasa_stability_veto"},
            "platform_drift": {"status": "STABLE"},
        },
    } for i in range(1, 4)]
    history[-2]["configuration"]["platform_drift"] = {"status": "HARMFUL"}
    assert cs._should_skip_training(history, "parent-c", dataset_key="C") is True
    assert cs._should_skip_training(history, "parent-c", dataset_key="B") is False
    history[-1]["configuration"]["platform_drift"] = {"status": "HARMFUL"}
    assert cs._should_skip_training(history, "parent-c", dataset_key="C") is False


def test_parent_metrics_prefers_matching_parent_artifact():
    history = [
        {"id": "r1", "selected_artifact_id": "old", "summary": {"metrics": {"rmse": 11.0}}},
        {"id": "r2", "selected_artifact_id": "parent-1", "summary": {"metrics": {"rmse": 9.5}}},
    ]
    rnd = type("R", (), {"parent": type("P", (), {
        "artifact_id": "parent-1", "receipt": {"artifact_id": "parent-1"}
    })()})()
    assert cs._parent_metrics(rnd, history) == {"rmse": 9.5}


def test_parent_metrics_ignores_rejected_candidate_of_kept_parent_round():
    history = [
        {"id": "r1", "selected_artifact_id": "parent-1",
         "configuration": {"parent_artifact_id": "base",
                           "selected_source": "candidate",
                           "platform_metrics": {"rmse": 51.816}}},
        {"id": "r2", "selected_artifact_id": "parent-1",
         "configuration": {"parent_artifact_id": "parent-1",
                           "selected_source": "parent",
                           "platform_metrics": {"rmse": 51.868},
                           "parent_platform_metrics": {"rmse": 51.816},
                           "selected_platform_metrics": {"rmse": 51.816}}},
    ]
    rnd = type("R", (), {"parent": type("P", (), {
        "artifact_id": "parent-1", "receipt": {"artifact_id": "parent-1"}
    })()})()
    assert cs._parent_metrics(rnd, history) == {"rmse": 51.816}


def test_selected_history_metrics_without_selected_field_uses_parent_metrics():
    record = {"selected_artifact_id": "parent-1",
              "configuration": {"parent_artifact_id": "parent-1",
                                "platform_metrics": {"rmse": 9.0},
                                "parent_platform_metrics": {"rmse": 10.0}}}
    assert cs._selected_history_metrics(record) == {"rmse": 10.0}
    old_format = {"selected_artifact_id": "x",
                  "configuration": {"platform_metrics": {"rmse": 9.7}}}
    assert cs._selected_history_metrics(old_format) == {"rmse": 9.7}


def test_best_history_uses_selected_metrics_not_rejected_candidate():
    history = [{
        "id": "r1", "selected_artifact_id": "parent-1",
        "configuration": {"parent_artifact_id": "parent-1",
                          "selected_source": "parent",
                          "platform_metrics": {"rmse": 5.0},
                          "selected_platform_metrics": {"rmse": 10.0}},
    }]
    assert cs._best_allowed_history(
        history, [{"artifact_id": "parent-1"}], "rmse", "lower",
        {"rmse": 9.0}, dataset_key="B") is None


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


def test_c_avoid_repeat_compares_learning_rate_after_epoch_is_fixed():
    history = [{
        "configuration": {
            "parent_artifact_id": "parent-c", "epochs": 9,
            "learning_rate": 0.001,
        }
    }]
    epochs, learning_rate = cs._avoid_repeat(
        "parent-c", 1, 0.001, history, dataset_key="C")
    assert epochs == 1
    assert learning_rate == pytest.approx(0.0005)


@pytest.mark.parametrize("round_index", [1, 2])
def test_a_early_round_strategy_uses_strong_known_start(round_index):
    epochs, lr, reset_optimizer, reason = cs._apply_profile_strategy(
        "A", round_index, 3, 0.0005, [], "parent-a")
    assert (epochs, lr) == (10, 0.01)
    assert reset_optimizer is False
    assert "A_early_round" in reason


def test_a_third_round_keeps_qwen_proposal():
    assert cs._apply_profile_strategy(
        "A", 3, 7, 0.002, [], "parent-a") == (7, 0.002, False, "")


def test_a_late_round_caps_epochs_to_limit_overfitting():
    epochs, lr, reset, reason = cs._apply_profile_strategy(
        "A", 7, 40, 0.0008, [], "parent-a")
    assert (epochs, lr, reset) == (cs.A_MAX_EPOCHS, 0.0008, False)
    assert reason.startswith("A_epoch_cap")


def test_ask_model_clamps_a_epochs_before_repeat_check():
    rnd = _Round([], {}, round_index=5)
    rnd.reply = '{"epochs": 40, "learning_rate": 0.001, "hypothesis": "加长训练"}'
    epochs, lr, _, _, _ = cs._ask_model(rnd, "classification", "macro_f1", [])
    assert epochs == cs.A_MAX_EPOCHS and lr == 0.001
    assert f"整数1-{cs.A_MAX_EPOCHS}" in rnd.prompts[0]


def test_a_skips_after_three_loops_and_two_stable_rejections_then_resumes():
    record = {
        "configuration": {
            "parent_artifact_id": "parent-a",
            "platform_decision": {"reason": "candidate_not_better"},
            "platform_drift": {"status": "STABLE"},
            "loop_evidence": {"complete": True},
        }
    }
    assert cs._should_skip_training([record] * 2, "parent-a", dataset_key="A") is False
    assert cs._should_skip_training([record] * 3, "parent-a", dataset_key="A") is True
    skipped = {"configuration": {**record["configuration"],
                                 "resource_action": "training_skipped"}}
    assert cs._should_skip_training(
        [record] * 3 + [skipped], "parent-a", dataset_key="A") is False
    assert cs._should_skip_training(
        [record] * 3 + [skipped, record], "parent-a", dataset_key="A") is False


def test_c_first_round_restores_first_submission_safe_start():
    result = cs._apply_profile_strategy(
        "C", 1, 20, 0.01, [], "baseline-c")
    assert result == (
        5, 0.001, False,
        "C_first_submission_start:恢复首次提交的RMSE_MAE安全起点")


def test_c_second_round_restores_first_submission_followup():
    history = [{
        "selected_artifact_id": "round-1-c",
        "configuration": {
            "parent_artifact_id": "baseline-c", "epochs": 5,
            "learning_rate": 0.001,
            "platform_decision": {"reason": "first_round_valid_candidate"},
        }
    }]
    result = cs._apply_profile_strategy(
        "C", 2, 20, 0.01, history, "round-1-c")
    assert result == (
        3, 0.0005, False,
        "C_first_submission_followup:恢复首次提交的RMSE_MAE安全续训")


def test_c_second_round_retries_safe_start_when_first_round_was_not_selected():
    history = [{
        "selected_artifact_id": "baseline-c",
        "configuration": {
            "parent_artifact_id": "baseline-c", "epochs": 5,
            "learning_rate": 0.001,
            "platform_decision": {"reason": "candidate_not_better"},
        }
    }]
    result = cs._apply_profile_strategy(
        "C", 2, 20, 0.01, history, "baseline-c")
    assert result == (
        5, 0.001, False,
        "C_first_submission_start:首轮未上线，重试RMSE_MAE安全起点")


def test_c_later_strategy_forces_one_epoch_and_keeps_qwen_learning_rate():
    result = cs._apply_profile_strategy(
        "C", 4, 20, 0.0005, [], "parent-c")
    assert result == (
        1, 0.0005, False, "C_platform_search:单步续训便于平台反馈归因")


def test_b_first_round_restores_first_submission_start():
    epochs, lr, reset_optimizer, reason = cs._apply_profile_strategy(
        "B", 1, 3, 0.01, [], "baseline-b")
    assert (epochs, lr) == (10, 0.001)
    assert reset_optimizer is False
    assert "B_first_submission_start" in reason


def test_b_second_round_restores_first_submission_followup():
    history = [{
        "selected_artifact_id": "round-1-b",
        "configuration": {
            "parent_artifact_id": "baseline-b", "epochs": 10,
            "learning_rate": 0.001,
            "platform_decision": {"reason": "first_round_valid_candidate"},
        }
    }]
    epochs, lr, reset_optimizer, reason = cs._apply_profile_strategy(
        "B", 2, 8, 0.01, history, "round-1-b")
    assert (epochs, lr) == (5, 0.001)
    assert reset_optimizer is False
    assert "B_first_submission_followup" in reason


def test_b_second_round_retries_safe_start_when_first_round_was_not_selected():
    history = [{
        "selected_artifact_id": "baseline-b",
        "configuration": {
            "parent_artifact_id": "baseline-b", "epochs": 10,
            "learning_rate": 0.001,
            "platform_decision": {"reason": "candidate_not_better"},
        }
    }]
    epochs, lr, reset_optimizer, reason = cs._apply_profile_strategy(
        "B", 2, 8, 0.01, history, "baseline-b")
    assert (epochs, lr) == (10, 0.001)
    assert reset_optimizer is False
    assert "B_first_submission_start" in reason


def test_b_stagnation_switches_to_first_untried_axis():
    history = [{
        "configuration": {
            "parent_artifact_id": "parent-b", "epochs": 5,
            "learning_rate": 0.001,
            "platform_decision": {"reason": "candidate_not_better"},
        }
    }]
    epochs, lr, reset_optimizer, reason = cs._apply_profile_strategy(
        "B", 4, 10, 0.002, history, "parent-b")
    assert (epochs, lr) == (3, 0.0005)
    assert reset_optimizer is True
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
        "B", 4, 7, 0.002, history, "parent-b") == (7, 0.002, False, "")


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
        self.prompts = []
        self.reply = '{"epochs": 2, "learning_rate": 0.003, "hypothesis": "test"}'
        self._history = history
        self._summary = summary
        self.artifacts = artifacts or []
        self.submitted_kwargs = None
        self.finished = None

    def history(self):
        return self._history

    def model(self, messages, **options):
        assert options["response_format"] == {"type": "json_object"}
        assert options["max_tokens"] <= cs.MODEL_MAX_TOKENS
        self.prompts.append(messages[0]["content"])
        assert options["enable_thinking"] is False
        self.call_ids.append("call-1")
        return type("Receipt", (), {
            "call_id": "call-1",
            "text": self.reply,
        })()

    def submit_candidate(self, directory, **kwargs):
        self.submitted_kwargs = kwargs
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


def _fake_training_report(monkeypatch, tmp_path, before, after):
    def run(*args, **kwargs):
        cand = tmp_path / "cand"
        cand.mkdir()
        (cand / "checkpoint.npz").write_bytes(b"checkpoint")
        (cand / "training.json").write_text(json.dumps({
            "seed": 42, "validation_before": before, "validation_after": after,
        }), encoding="utf-8")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    monkeypatch.setattr(cs.subprocess, "run", run)


def test_first_round_keeps_baseline_when_candidate_worse_than_local_proxy(
        monkeypatch, tmp_path):
    rnd = _Round([], {"macro_f1": 0.7264, "accuracy": 0.885}, round_index=1)
    _fake_training_report(monkeypatch, tmp_path,
                          {"macro_f1": 0.80, "accuracy": 0.90},
                          {"macro_f1": 0.7264, "accuracy": 0.885})
    cs.train(rnd)
    config = rnd.finished["configuration"]
    assert rnd.finished["selected_artifact_id"] == "parent-1"
    assert config["parent_metric_source"] == "local_validation_before"
    assert config["platform_decision"]["reason"] == "candidate_not_better"


def test_first_round_proxy_rejected_when_local_and_platform_metrics_disagree(
        monkeypatch, tmp_path):
    rnd = _Round([], {"macro_f1": 0.70}, round_index=1)
    _fake_training_report(monkeypatch, tmp_path,
                          {"macro_f1": 0.80}, {"macro_f1": 0.90})
    cs.train(rnd)
    config = rnd.finished["configuration"]
    assert rnd.finished["selected_artifact_id"] == "candidate-1"
    assert config["parent_metric_source"] == "proxy_rejected:metric_mismatch"
    assert config["platform_decision"]["reason"] == "first_round_valid_candidate"


def _c_round(tmp_path, monkeypatch, *, platform_candidate, report_extra,
             history=None, artifacts=None, round_index=4):
    history = history if history is not None else [{
        "id": "r1", "round_index": 1, "selected_artifact_id": "parent-1",
        "configuration": {"parent_artifact_id": "base", "selected_source": "candidate",
                          "selected_platform_metrics": {
                              "rmse": 52.0, "nasa_asymmetric_score": 100.0}},
    }]
    rnd = _Round(history, platform_candidate, round_index=round_index,
                 artifacts=artifacts)
    rnd.dataset_key = "C"
    rnd.task = {"task": "regression", "metric": "rmse", "metric_direction": "lower"}
    rnd.reply = '{"learning_rate": 0.0004, "hypothesis": "小步"}'

    def run(cmd, **kwargs):
        cand = tmp_path / "cand"
        cand.mkdir()
        (cand / "checkpoint.npz").write_bytes(b"checkpoint")
        (cand / "training.json").write_text(json.dumps({
            "seed": 42, "validation_before": {"rmse": 52.0},
            "validation_after": platform_candidate, **report_extra,
        }), encoding="utf-8")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    monkeypatch.setattr(cs.subprocess, "run", run)
    cs.train(rnd)
    return rnd


_HIDDEN_BETTER = {
    "hidden_like_max": 200, "hidden_like_rows": 100,
    "validation_before_hidden_like": {"rmse": 38.0, "nasa_asymmetric_score": 1e8},
    "validation_after_hidden_like": {"rmse": 34.0, "nasa_asymmetric_score": 1e7},
}


def test_c_adopts_hidden_like_gain_despite_small_public_degradation(monkeypatch, tmp_path):
    rnd = _c_round(tmp_path, monkeypatch,
                   platform_candidate={"rmse": 53.456, "nasa_asymmetric_score": 900.0},
                   report_extra=_HIDDEN_BETTER)
    config = rnd.finished["configuration"]
    assert rnd.finished["selected_artifact_id"] == "candidate-1"
    assert config["platform_decision"]["source"] == "local_hidden_like"
    assert config["platform_decision"]["public_guard"]["status"] == "passed"
    assert config["local_hidden_like"]["decision_source"] == "local_hidden_like"
    assert config["platform_drift"]["basis"] == "local_hidden_like"
    assert config["platform_drift"]["status"] == "IMPROVED"
    assert rnd.finished["reason"].startswith("类隐藏集")


def test_c_public_guard_blocks_large_public_degradation(monkeypatch, tmp_path):
    rnd = _c_round(tmp_path, monkeypatch,
                   platform_candidate={"rmse": 55.0, "nasa_asymmetric_score": 90.0},
                   report_extra=_HIDDEN_BETTER)
    decision = rnd.finished["configuration"]["platform_decision"]
    assert rnd.finished["selected_artifact_id"] == "parent-1"
    assert decision["reason"] == "public_guard"
    assert decision["public_guard"]["status"] == "blocked"


def test_c_rejects_hidden_like_nasa_worsening(monkeypatch, tmp_path):
    extra = {**_HIDDEN_BETTER, "validation_after_hidden_like": {
        "rmse": 34.0, "nasa_asymmetric_score": 2e8}}
    rnd = _c_round(tmp_path, monkeypatch,
                   platform_candidate={"rmse": 51.0, "nasa_asymmetric_score": 90.0},
                   report_extra=extra)
    config = rnd.finished["configuration"]
    assert rnd.finished["selected_artifact_id"] == "parent-1"
    assert config["platform_decision"]["reason"] == "nasa_stability_veto"
    assert config["platform_drift"]["status"] == "HARMFUL"


def test_c_hidden_like_path_never_restores_history(monkeypatch, tmp_path):
    history = [
        {"id": "r1", "round_index": 1, "selected_artifact_id": "old-best",
         "configuration": {"parent_artifact_id": "base", "selected_source": "candidate",
                           "selected_platform_metrics": {
                               "rmse": 40.0, "nasa_asymmetric_score": 1.0}}},
        {"id": "r2", "round_index": 2, "selected_artifact_id": "parent-1",
         "configuration": {"parent_artifact_id": "old-best", "selected_source": "candidate",
                           "selected_platform_metrics": {
                               "rmse": 52.0, "nasa_asymmetric_score": 100.0}}},
    ]
    worse = {**_HIDDEN_BETTER, "validation_after_hidden_like": {
        "rmse": 39.0, "nasa_asymmetric_score": 1e8}}
    rnd = _c_round(tmp_path, monkeypatch,
                   platform_candidate={"rmse": 52.5, "nasa_asymmetric_score": 100.0},
                   report_extra=worse, history=history,
                   artifacts=[{"artifact_id": "old-best"}, {"artifact_id": "parent-1"}])
    assert rnd.finished["selected_artifact_id"] == "parent-1"
    assert rnd.finished["configuration"]["selected_source"] == "parent"


def test_c_history_prompt_prefers_hidden_like_metrics():
    history = [{"round_index": 1, "configuration": {
        "platform_metrics": {"rmse": 52.0},
        "local_hidden_like": {"candidate": {"rmse": 34.0, "nasa_asymmetric_score": 9.0}},
    }}]
    prompt = cs._history_prompt(history, metric="rmse")
    assert 'hidden_like_metrics={"rmse": 34.0' in prompt
    assert "52.0" not in prompt


def test_a_ignores_hidden_like_fields_and_uses_platform(monkeypatch, tmp_path):
    history = [{
        "id": "r1", "selected_artifact_id": "parent-1",
        "summary": {"metrics": {"macro_f1": 0.80}},
        "configuration": {"epochs": 5, "learning_rate": 0.001},
    }]
    rnd = _Round(history, {"macro_f1": 0.79})
    _fake_training(monkeypatch, tmp_path, local_promote=True)
    cs.train(rnd)
    assert rnd.finished["selected_artifact_id"] == "parent-1"
    assert "local_hidden_like" not in rnd.finished["configuration"]


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
    assert rnd.finished["configuration"]["local_diagnostic"] == {
        "source": "training_report",
        "validation_before": {"macro_f1": 0.70},
        "validation_after": {"macro_f1": 0.99},
        "early_stopped": None, "training_seconds": None,
    }
    assert rnd.finished["configuration"]["loop_evidence"]["complete"] is True
    assert (rnd.finished["configuration"]["loop_evidence"]["experiment_status"]
            == "candidate_measured")
    assert rnd.finished["configuration"]["platform_drift"]["enabled"] is True
    candidate_evidence = rnd.submitted_kwargs["configuration"]["report_evidence"]
    assert rnd.submitted_kwargs["hypothesis"]
    assert rnd.submitted_kwargs["change_summary"]
    assert candidate_evidence["hypothesis"] == "test；A_early_round:前两轮采用已验证的强起点"
    assert candidate_evidence["change_summary"].startswith("从父检查点续训")
    assert candidate_evidence["model_call_ids"] == ["call-1"]
    assert candidate_evidence["candidate_count"] == 1
    finish_evidence = rnd.finished["configuration"]["report_evidence"]
    assert finish_evidence["hypothesis"] == rnd.finished["hypothesis"]
    assert finish_evidence["change_summary"] == rnd.finished["change_summary"]
    assert finish_evidence["reason"] == rnd.finished["reason"]
    assert finish_evidence["model_call_ids"] == ["call-1"]
    assert finish_evidence["history_ids"] == ["r1"]
    assert finish_evidence["candidate_count"] == 1
    assert finish_evidence["candidate_artifact_ids"] == ["candidate-1"]
    assert rnd.finished["model_call_ids"] == ["call-1"]


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
    report = rnd.finished["configuration"]["report_evidence"]
    assert report["hypothesis"] == rnd.finished["hypothesis"]
    assert report["reason"] == rnd.finished["reason"]
    assert report["model_call_ids"] == ["call-1"]
    assert report["candidate_count"] == 0


def test_train_can_skip_one_low_value_round_after_three_complete_loops(
        monkeypatch, tmp_path):
    history = []
    for index in range(1, 4):
        history.append({
            "id": f"r{index}", "round_index": index,
            "selected_artifact_id": "parent-1",
            "summary": {"metrics": {"macro_f1": 0.80, "accuracy": 0.90}},
            "configuration": {
                "parent_artifact_id": "parent-1",
                "epochs": 5, "learning_rate": 0.001,
                "loop_evidence": {"complete": True},
                "platform_decision": {"reason": "candidate_not_better"},
                "platform_drift": {"status": "STABLE"},
            },
        })
    rnd = _Round(history, {"macro_f1": 0.81}, round_index=4)
    rnd.dataset_key = "B"
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    monkeypatch.setattr(
        cs.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must be skipped")))

    cs.train(rnd)

    assert rnd.finished["selected_artifact_id"] == "parent-1"
    assert rnd.finished["configuration"]["resource_action"] == "training_skipped"
    assert rnd.finished["configuration"]["complete_loops_total"] == 3
    assert rnd.finished["configuration"]["loop_evidence"]["complete"] is False
    report = rnd.finished["configuration"]["report_evidence"]
    assert report["model_call_ids"] == ["call-1"]
    assert report["history_ids"] == ["r1", "r2", "r3"]
    assert report["candidate_count"] == 0


def test_train_c_uses_one_epoch_and_trains_after_recent_improvement(monkeypatch, tmp_path):
    history = [{
        "id": f"r{i}", "round_index": i,
        "selected_artifact_id": "parent-1",
        "summary": {"metrics": {
            "rmse": 52.0, "nasa_asymmetric_score": 100.0}},
        "configuration": {
            "parent_artifact_id": "parent-1", "epochs": 5,
            "learning_rate": 0.001,
            "loop_evidence": {"complete": True},
            "platform_decision": {
                "reason": "candidate_better" if i == 3 else "candidate_not_better"},
            "platform_drift": {"status": "STABLE"},
        },
    } for i in range(1, 4)]
    rnd = _Round(
        history, {"rmse": 51.9, "nasa_asymmetric_score": 99.0},
        round_index=4)
    rnd.dataset_key = "C"
    rnd.task = {
        "task": "regression", "metric": "rmse", "metric_direction": "lower"}
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        cand = tmp_path / "cand"
        cand.mkdir()
        (cand / "checkpoint.npz").write_bytes(b"checkpoint")
        (cand / "training.json").write_text(json.dumps({
            "seed": 42,
            "validation_before": {"rmse": 52.0},
            "validation_after": {"rmse": 51.9},
        }), encoding="utf-8")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    monkeypatch.setattr(cs.subprocess, "run", run)
    monkeypatch.setattr(
        cs, "_decide_promote",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bootstrap removed")))
    cs.train(rnd)

    index = seen["cmd"].index("--epochs")
    assert seen["cmd"][index + 1] == "1"
    assert rnd.finished["configuration"].get("resource_action") is None


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
