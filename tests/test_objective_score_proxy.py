import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import objective_score_proxy as proxy  # noqa: E402


def test_load_static_baselines_reads_official_training_reports(tmp_path):
    values = {"A": ("classification", 0.714), "B": ("regression", 681.16),
              "C": ("regression", 52.87)}
    for profile, (task, value) in values.items():
        package = tmp_path / f"cpu-{profile.lower()}-guide"
        package.mkdir()
        key = "macro_f1" if task == "classification" else "rmse"
        (package / "training.json").write_text(json.dumps({
            "profile": profile, "task": task,
            "validation_after": {key: value},
        }), encoding="utf-8")

    baselines = proxy.load_static_baselines(tmp_path)

    assert baselines == {
        "A": {"metric": "macro_f1", "direction": "higher", "value": 0.714},
        "B": {"metric": "rmse", "direction": "lower", "value": 681.16},
        "C": {"metric": "rmse", "direction": "lower", "value": 52.87},
    }


def test_summarize_rounds_handles_direction_recovery_and_resources():
    rows = [
        {"round": "1", "value": "0.70", "best_value": "0.70",
         "accepted": "0", "seconds": "1.0"},
        {"round": "2", "value": "0.60", "best_value": "0.70",
         "accepted": "0", "seconds": "2.0"},
        {"round": "3", "value": "0.80", "best_value": "0.80",
         "accepted": "1", "seconds": "3.0"},
    ]
    result = proxy.summarize_rounds(
        "A", rows, {"metric": "macro_f1", "direction": "higher", "value": 0.70})

    assert result["average_relative_gain"] == pytest.approx(0.0)
    assert result["worst_relative_gain"] == pytest.approx(-1 / 7)
    assert result["recovery_rounds"] == 1
    assert result["max_drawdown"] == pytest.approx(1 / 7)
    assert result["total_seconds"] == 6.0
    assert result["promotions"] == 1
    assert result["rejections"] == 2
    assert result["final_best"] == 0.8


def test_write_reports_labels_outputs_as_proxy(tmp_path):
    rows = [{
        "profile": "A", "metric": "macro_f1", "baseline": 0.714,
        "average_relative_gain": 0.01, "worst_relative_gain": -0.02,
        "recovery_rounds": 1, "max_drawdown": 0.03, "total_seconds": 10.0,
        "promotions": 2, "rejections": 8, "adoption_rate": 0.2,
        "final_best": 0.72, "resource_efficiency": 0.001,
    }]

    csv_path, md_path = proxy.write_reports(rows, tmp_path)

    assert csv_path.is_file() and md_path.is_file()
    assert "客观评分代理" in md_path.read_text(encoding="utf-8")
    assert "不等同于平台正式分数" in md_path.read_text(encoding="utf-8")
