import numpy as np

from bench_external import _error_stream, to_dicts
from harness import metrics


def test_to_dicts_roundtrip():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    ds = to_dicts(X)
    assert ds == [{0: 1.0, 1: 2.0}, {0: 3.0, 1: 4.0}]


def test_error_stream_classification_error_rate():
    preds = [
        (1, np.array([0, 1, 2, 2]), np.array([0, 1, 2, 0])),   # 1/4 错
        (2, np.array([0, 0]), np.array([1, 1])),               # 全错
    ]
    out = _error_stream([], preds, "classification")
    assert out[1] == 0.25
    assert out[2] == 1.0


def test_error_stream_regression_mae():
    preds = [(1, np.array([10.0, 20.0]), np.array([12.0, 18.0]))]
    out = _error_stream([], preds, "regression")
    assert out[1] == 2.0


def test_composite_matches_metrics():
    scores = {1: 0.5, 2: 0.7, 3: float("nan"), 4: 0.9}
    vals = list(scores.values())
    assert metrics.composite(vals) == np.mean([0.5, 0.7, 0.9])
