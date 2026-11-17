import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
sys.path.insert(0, str(SUBMISSION))

from baselines import train as baseline_train  # noqa: E402


def test_train_run_writes_best_intermediate_epoch(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self, task, paths, classes, x, y, *, hidden, seed,
                     learning_rate, batch_size):
            del x, y
            self.meta = {
                "task": task, "paths": paths, "classes": classes,
                "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
                "learning_rate": learning_rate, "batch_size": batch_size,
                "parameter_count": 1,
            }

        @classmethod
        def load(cls, path):
            raise AssertionError(f"unexpected resume: {path}")

        def validate(self):
            return None

        def fit(self, x, y, epochs):
            del x, y
            assert epochs == 1
            self.meta["epochs"] += 1
            self.meta["steps"] += 1
            return [1.0 / self.meta["epochs"]]

        def save(self, path):
            Path(path).write_text(f"epoch={self.meta['epochs']}", encoding="utf-8")

    def fake_dataset(path, task, classes, expected_paths=None):
        del task, classes, expected_paths
        is_validation = "validation" in str(path)
        return {
            "ids": ["v1"] if is_validation else ["t1"],
            "x": np.array([[1.0]]), "y": np.array([0]),
            "paths": ['[["key","x"]]'], "sha256": "sha256:data",
        }

    scores = {0: 0.50, 1: 0.80, 2: 0.90, 3: 0.70}
    monkeypatch.setattr(baseline_train, "Model", FakeModel)
    monkeypatch.setattr(baseline_train, "read_dataset", fake_dataset)
    monkeypatch.setattr(
        baseline_train, "metrics",
        lambda model, data: {"accuracy": scores[model.meta["epochs"]],
                             "macro_f1": scores[model.meta["epochs"]]},
    )
    monkeypatch.setattr(
        baseline_train, "checkpoint_digest",
        lambda path: "sha256:" + Path(path).read_text(encoding="utf-8"),
    )

    output = tmp_path / "candidate"
    report = baseline_train.train_run(
        "A", "train.jsonl", "validation.jsonl", output, epochs=3)

    assert (output / "checkpoint.npz").read_text(encoding="utf-8") == "epoch=2"
    assert report["selected_epoch"] == 2
    assert report["trained_final_epochs"] == 3
    assert report["best_validation"] == {"accuracy": 0.9, "macro_f1": 0.9}
    assert report["validation_after"] == report["best_validation"]
    history = [json.loads(line) for line in
               (output / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["epoch"] for item in history] == [1, 2, 3]

