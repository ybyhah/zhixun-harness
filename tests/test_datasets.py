import numpy as np
import pytest

from harness.datasets.base import Batch, TaskAdapter
from harness.datasets.gas_sensor import CLASS_SET, N_FEATURES, GasSensorAdapter
from harness.datasets.preprocess import IncrementalScaler, fill_missing

RAW = "data/raw"

# 公开资料记载的各批次样本数
EXPECTED_ROWS = [445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600]


@pytest.fixture(scope="module")
def batches() -> list[Batch]:
    return GasSensorAdapter(RAW).load_batches()


def test_adapter_satisfies_protocol():
    assert isinstance(GasSensorAdapter(RAW), TaskAdapter)


def test_batch_count_and_shapes(batches):
    assert len(batches) == 10
    for b, n in zip(batches, EXPECTED_ROWS):
        assert b.X.shape == (n, N_FEATURES)
        assert b.y.shape == (n,)
        assert len(b) == n


def test_labels_within_global_class_set(batches):
    for b in batches:
        assert set(np.unique(b.y)).issubset(set(CLASS_SET.tolist()))


def test_batches_345_have_no_toluene(batches):
    """批次 3/4/5 无 Toluene(6)，这是数据集已知特性，也是缺类处理的依据。"""
    for idx in (2, 3, 4):
        assert 6 not in set(np.unique(batches[idx].y).tolist())


def test_batch10_is_balanced(batches):
    counts = np.bincount(batches[9].y, minlength=7)[1:7]
    assert counts.tolist() == [600] * 6


def test_features_are_finite(batches):
    for b in batches:
        assert np.isfinite(b.X).all()


def test_incremental_scaler_only_uses_seen_data():
    scaler = IncrementalScaler(2)
    first = np.array([[0.0, 0.0], [2.0, 2.0]])
    scaler.partial_fit(first)
    before = scaler.transform(first).copy()
    # 喂入新数据后统计量变化，旧数据的变换结果也会随之改变
    scaler.partial_fit(np.array([[100.0, 100.0]]))
    after = scaler.transform(first)
    assert not np.allclose(before, after)


def test_scaler_passthrough_when_unfitted():
    scaler = IncrementalScaler(3)
    X = np.ones((2, 3))
    assert np.allclose(scaler.transform(X), X)


def test_fill_missing_replaces_non_finite():
    X = np.array([[1.0, np.nan], [np.inf, 2.0]])
    out = fill_missing(X)
    assert np.isfinite(out).all()
    assert out[0, 1] == 0.0 and out[1, 0] == 0.0
