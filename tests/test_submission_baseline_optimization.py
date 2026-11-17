import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
sys.path.insert(0, str(SUBMISSION))

from baselines import train as baseline_train  # noqa: E402
from baselines import model as baseline_model  # noqa: E402

Model = baseline_model.Model


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


def test_c_regression_weights_penalize_overestimation_only():
    error = np.array([-1.0, 1.0])
    assert np.array_equal(
        baseline_model._regression_weights(error, 1.0), np.array([1.0, 1.0]))
    assert np.array_equal(
        baseline_model._regression_weights(error, 1.5), np.array([1.0, 1.5]))


def test_c_weight_changes_training_loss_without_affecting_b_default():
    x = np.array([[0.0], [0.0]])
    y = np.array([0.0, 0.0])

    def model(weight):
        current = Model("regression", ["x"], [], x, y, hidden=1,
                        seed=42, learning_rate=0.001, batch_size=2)
        for value in current.params.values():
            value.fill(0.0)
        current.params["b2"].fill(1.0)
        current.meta["overestimate_weight"] = weight
        return current

    b_loss = model(1.0).fit(x, y, 1)[0]
    c_loss = model(1.5).fit(x, y, 1)[0]
    assert c_loss == 1.5 * b_loss


def test_nasa_score_penalizes_equal_overestimate_more_than_underestimate():
    over = baseline_train._nasa_score(np.array([11.0]), np.array([1.0]))
    under = baseline_train._nasa_score(np.array([0.0]), np.array([10.0]))
    assert over > under > 0


def test_c_train_run_persists_safety_weight_and_nasa_metric(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_rows = [
        {"id": f"t{i}", "inputs": {"x": float(i)}, "answer": float(12 - i)}
        for i in range(4)
    ]
    validation_rows = [
        {"id": f"v{i}", "inputs": {"x": float(i) + 0.5}, "answer": float(10 - i)}
        for i in range(2)
    ]
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
    validation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in validation_rows), encoding="utf-8")

    output = tmp_path / "c-candidate"
    report = baseline_train.train_run(
        "C", train_path, validation_path, output, epochs=1)
    saved = Model.load(output / "checkpoint.npz")

    assert saved.meta["overestimate_weight"] == 1.5
    assert report["configuration"]["overestimate_weight"] == 1.5
    assert report["best_validation"]["nasa_asymmetric_score"] >= 0

