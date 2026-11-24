import json
import sys
from pathlib import Path

import numpy as np
import pytest


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


def test_reset_optimizer_moments_preserves_steps_and_parameters():
    x = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    model = Model("regression", ["x"], [], x, y, hidden=1, seed=42)
    model.meta["steps"] = 9
    before = {name: value.copy() for name, value in model.params.items()}
    for value in (*model.first.values(), *model.second.values()):
        value.fill(2.0)

    baseline_train._reset_optimizer_moments(model)

    assert model.meta["steps"] == 9
    assert all(np.array_equal(model.params[name], before[name]) for name in before)
    assert all(not np.any(value) for value in model.first.values())
    assert all(not np.any(value) for value in model.second.values())


def test_nasa_score_penalizes_equal_overestimate_more_than_underestimate():
    over = baseline_train._nasa_score(np.array([11.0]), np.array([1.0]))
    under = baseline_train._nasa_score(np.array([0.0]), np.array([10.0]))
    assert over > under > 0


def test_c_epoch_selection_balances_rmse_and_nasa():
    best = {"rmse": 10.0, "nasa_asymmetric_score": 100.0}
    assert baseline_train._better_validation(
        "C", "regression", {"rmse": 9.9, "nasa_asymmetric_score": 104.0}, best)
    assert not baseline_train._better_validation(
        "C", "regression", {"rmse": 9.8, "nasa_asymmetric_score": 106.0}, best)
    assert baseline_train._better_validation(
        "C", "regression", {"rmse": 10.1, "nasa_asymmetric_score": 94.0}, best)


def test_c_stops_after_three_epochs_without_multiobjective_improvement(
        monkeypatch, tmp_path):
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
            self.meta["epochs"] += epochs
            self.meta["steps"] += epochs
            return [1.0]

        def save(self, path):
            Path(path).write_text(f"epoch={self.meta['epochs']}", encoding="utf-8")

    def fake_dataset(path, task, classes, expected_paths=None):
        del task, classes, expected_paths
        return {
            "ids": ["v1"] if "validation" in str(path) else ["t1"],
            "x": np.array([[1.0]]), "y": np.array([1.0]),
            "paths": ['[["key","x"]]'], "sha256": "sha256:data",
        }

    values = {
        0: {"mae": 11.0, "rmse": 11.0, "nasa_asymmetric_score": 110.0},
        1: {"mae": 9.0, "rmse": 9.0, "nasa_asymmetric_score": 90.0},
        2: {"mae": 9.1, "rmse": 9.1, "nasa_asymmetric_score": 91.0},
        3: {"mae": 9.2, "rmse": 9.2, "nasa_asymmetric_score": 92.0},
        4: {"mae": 9.3, "rmse": 9.3, "nasa_asymmetric_score": 93.0},
    }
    monkeypatch.setattr(baseline_train, "Model", FakeModel)
    monkeypatch.setattr(baseline_train, "read_dataset", fake_dataset)
    monkeypatch.setattr(
        baseline_train, "metrics",
        lambda model, data: values.get(model.meta["epochs"], values[4]))
    monkeypatch.setattr(
        baseline_train, "checkpoint_digest",
        lambda path: "sha256:" + Path(path).read_text(encoding="utf-8"))

    output = tmp_path / "c-early-stop"
    report = baseline_train.train_run(
        "C", "train.jsonl", "validation.jsonl", output, epochs=10)

    assert report["trained_final_epochs"] == 4
    assert report["selected_epoch"] == 1
    assert report["early_stopped"] is True


@pytest.mark.parametrize("profile", ["A", "B"])
def test_a_and_b_stop_after_three_epochs_without_improvement(
        monkeypatch, tmp_path, profile):
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

        def validate(self):
            return None

        def fit(self, x, y, epochs):
            del x, y
            self.meta["epochs"] += epochs
            self.meta["steps"] += epochs
            return [1.0]

        def save(self, path):
            Path(path).write_text(f"epoch={self.meta['epochs']}", encoding="utf-8")

    def fake_dataset(path, task, classes, expected_paths=None):
        del task, classes, expected_paths
        return {
            "ids": ["v1"] if "validation" in str(path) else ["t1"],
            "x": np.array([[1.0]]), "y": np.array([0]),
            "paths": ['[["key","x"]]'], "sha256": "sha256:data",
        }

    def fake_metrics(model, data):
        epoch = model.meta["epochs"]
        score = 0.9 if epoch == 1 else 0.5
        if model.meta["task"] == "classification":
            return {"accuracy": score, "macro_f1": score}
        return {"mae": 1.0 / score, "rmse": 1.0 / score}

    monkeypatch.setattr(baseline_train, "Model", FakeModel)
    monkeypatch.setattr(baseline_train, "read_dataset", fake_dataset)
    monkeypatch.setattr(baseline_train, "metrics", fake_metrics)
    monkeypatch.setattr(
        baseline_train, "checkpoint_digest",
        lambda path: "sha256:" + Path(path).read_text(encoding="utf-8"))

    report = baseline_train.train_run(
        profile, "train.jsonl", "validation.jsonl", tmp_path / "out", epochs=20)

    assert report["trained_final_epochs"] == 1 + baseline_train.EARLY_STOP_PATIENCE
    assert report["selected_epoch"] == 1
    assert report["early_stopped"] is True


def test_c_best_epoch_uses_hidden_like_metrics(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self, task, paths, classes, x, y, *, hidden, seed,
                     learning_rate, batch_size):
            self.meta = {"task": task, "paths": paths, "classes": classes,
                         "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
                         "learning_rate": learning_rate, "batch_size": batch_size,
                         "parameter_count": 1}

        def validate(self):
            return None

        def fit(self, x, y, epochs):
            self.meta["epochs"] += epochs
            return [1.0]

        def save(self, path):
            Path(path).write_text(f"epoch={self.meta['epochs']}", encoding="utf-8")

    def fake_dataset(path, task, classes, expected_paths=None):
        y = np.array([100.0, 300.0]) if "validation" in str(path) else np.array([1.0])
        return {"ids": [f"{path}{i}" for i in range(len(y))], "x": np.ones((len(y), 1)),
                "y": y, "paths": ['[["key","x"]]'], "sha256": "sha256:data"}

    # 第 1 轮完整集更好、类隐藏集更差；第 2 轮相反
    full = {0: 50.0, 1: 40.0, 2: 60.0, 3: 70.0}
    sub = {0: 50.0, 1: 45.0, 2: 30.0, 3: 35.0}

    def fake_metrics(model, data):
        value = (sub if len(data["y"]) == 1 else full)[model.meta["epochs"]]
        return {"mae": value, "rmse": value, "nasa_asymmetric_score": 10.0}

    monkeypatch.setattr(baseline_train, "Model", FakeModel)
    monkeypatch.setattr(baseline_train, "read_dataset", fake_dataset)
    monkeypatch.setattr(baseline_train, "metrics", fake_metrics)
    monkeypatch.setattr(baseline_train, "checkpoint_digest",
                        lambda path: "sha256:" + Path(path).read_text(encoding="utf-8"))
    report = baseline_train.train_run(
        "C", "train.jsonl", "validation.jsonl", tmp_path / "out", epochs=3)
    assert report["selected_epoch"] == 2
    assert report["validation_after_hidden_like"]["rmse"] == 30.0
    assert report["validation_after"]["rmse"] == 60.0
    assert report["validation_before_hidden_like"]["rmse"] == 50.0


def test_c_train_run_uses_official_weight_and_nasa_metric(tmp_path):
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

    assert saved.meta["overestimate_weight"] == 1.0
    assert report["configuration"]["overestimate_weight"] == 1.0
    assert report["best_validation"]["nasa_asymmetric_score"] >= 0


def _regression_model(cap_meta=None):
    x = np.array([[0.0], [1.0]])
    model = Model("regression", ["x"], [], x, np.array([0.0, 1.0]), hidden=1,
                  seed=42, learning_rate=0.001, batch_size=2)
    for value in model.params.values():
        value.fill(0.0)
    model.meta["target_mean"], model.meta["target_scale"] = 300.0, 1.0
    model.meta.update(cap_meta or {})
    model.validate()
    return model


def test_prediction_cap_limits_regression_output_and_old_checkpoints_unchanged():
    x = np.array([[0.0]])
    assert _regression_model().predict(x)[0] == 300.0
    assert _regression_model({"prediction_cap": 200}).predict(x)[0] == 200.0


def test_target_cap_changes_regression_training_target():
    x, y = np.array([[0.0], [0.0]]), np.array([400.0, 400.0])
    capped = _regression_model({"target_cap": 200})
    plain = _regression_model()
    capped.fit(x, y, 1)
    plain.fit(x, y, 1)
    # 目标 400 截断为 200（低于当前输出 300），梯度方向相反
    assert capped.params["b2"][0] < 0 < plain.params["b2"][0]


@pytest.mark.parametrize("value", [0, -1, float("inf"), True, "200"])
def test_invalid_regression_cap_is_rejected(value):
    with pytest.raises(ValueError):
        _regression_model({"prediction_cap": value})


def test_cap_rejected_for_classification():
    x = np.array([[0.0], [1.0]])
    model = Model("classification", ["x"], ["a", "b"], x, np.array([0, 1]),
                  hidden=1, seed=42, learning_rate=0.001, batch_size=2)
    model.meta["target_cap"] = 200
    with pytest.raises(ValueError):
        model.validate()


def test_c_train_run_writes_caps_and_hidden_like_metrics(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("".join(
        json.dumps({"id": f"t{i}", "inputs": {"x": float(i)}, "answer": float(i * 60)}) + "\n"
        for i in range(6)), encoding="utf-8")
    validation_path.write_text("".join(
        json.dumps({"id": f"v{i}", "inputs": {"x": float(i)}, "answer": float(i * 100)}) + "\n"
        for i in range(4)), encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline_train.train_run("C", train_path, validation_path, baseline, epochs=1)
    parent = Model.load(baseline / "checkpoint.npz")
    assert parent.meta["prediction_cap"] == 200 and parent.meta["target_cap"] == 200

    report = baseline_train.train_run(
        "C", train_path, validation_path, tmp_path / "cand",
        resume=baseline / "checkpoint.npz", epochs=2)
    assert report["hidden_like_max"] == 200
    assert report["hidden_like_rows"] == 3          # 0、100、200 三个样本
    for key in ("validation_before_hidden_like", "validation_after_hidden_like"):
        assert set(report[key]) == {"mae", "rmse", "nasa_asymmetric_score"}
    assert report["configuration"]["prediction_cap"] == 200


def test_a_train_run_has_no_caps_or_hidden_like_metrics(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    rows = [{"id": f"t{i}", "inputs": {"x": float(i)}, "answer": "a" if i % 2 else "b"}
            for i in range(4)]
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    validation_path.write_text("".join(
        json.dumps({**r, "id": "v" + r["id"]}) + "\n" for r in rows), encoding="utf-8")
    config = tmp_path / "A.json"
    config.write_text(json.dumps({"profile": "A", "task": "classification",
                                  "classes": ["a", "b"], "hidden": 4, "seed": 42,
                                  "learning_rate": 0.01, "batch_size": 4, "epochs": 1}),
                      encoding="utf-8")
    report = baseline_train.train_run(config, train_path, validation_path,
                                      tmp_path / "out", epochs=1)
    saved = Model.load(tmp_path / "out" / "checkpoint.npz")
    assert "prediction_cap" not in saved.meta and "target_cap" not in saved.meta
    assert report["validation_after_hidden_like"] is None

