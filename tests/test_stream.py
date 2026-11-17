import numpy as np
import pytest

from harness.datasets.base import Batch
from harness.stream.batch_stream import (
    OFFLINE,
    BatchStream,
    StreamProtocolError,
)
from harness.stream.simulator import DriftSpec, inject


def make_batches(n=4, rows=50, feats=6, seed=0) -> list[Batch]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        X = rng.normal(size=(rows, feats))
        y = rng.integers(1, 4, size=rows)
        out.append(Batch(index=i, X=X, y=y))
    return out


# ---------- 串行卡点 ----------

def test_next_batch_hides_labels():
    s = BatchStream(make_batches())
    view = s.next_batch()
    assert not hasattr(view, "y")
    assert view.index == 0


def test_cannot_skip_submit():
    s = BatchStream(make_batches())
    s.next_batch()
    with pytest.raises(StreamProtocolError):
        s.next_batch()


def test_cannot_submit_twice():
    s = BatchStream(make_batches())
    view = s.next_batch()
    s.submit(np.zeros(len(view)))
    with pytest.raises(StreamProtocolError):
        s.submit(np.zeros(len(view)))


def test_labels_returned_only_after_submit():
    batches = make_batches()
    s = BatchStream(batches)
    view = s.next_batch()
    res = s.submit(np.zeros(len(view)))
    assert np.array_equal(res.y_true, batches[0].y)
    assert res.kind == "prediction"


def test_prediction_length_is_validated():
    s = BatchStream(make_batches())
    s.next_batch()
    with pytest.raises(ValueError):
        s.submit(np.zeros(3))


def test_offline_submission_is_recorded():
    s = BatchStream(make_batches())
    s.next_batch()
    res = s.submit(OFFLINE)
    assert res.kind == "offline"
    assert s.offline_batches == [0]


def test_stream_exhausts():
    batches = make_batches(n=3)
    s = BatchStream(batches)
    seen = 0
    while (view := s.next_batch()) is not None:
        s.submit(np.zeros(len(view)))
        seen += 1
    assert seen == 3 and s.exhausted


# ---------- 漂移注入 ----------

def _psi_like(a: np.ndarray, b: np.ndarray) -> float:
    """用均值的标准化差异粗略衡量分布变化。"""
    return float(np.abs(a.mean(axis=0) - b.mean(axis=0)).mean())


def test_covariate_injection_shifts_distribution():
    base = make_batches(n=5)
    out = inject(base, [DriftSpec(kind="covariate", speed="gradual", magnitude=2.0)])
    assert _psi_like(out[-1].X, base[-1].X) > _psi_like(out[0].X, base[0].X)


def test_covariate_leaves_labels_untouched():
    base = make_batches(n=3)
    out = inject(base, [DriftSpec(kind="covariate", magnitude=1.0)])
    for a, b in zip(base, out):
        assert np.array_equal(a.y, b.y)


def test_label_injection_changes_class_mix():
    base = make_batches(n=3, rows=300)
    out = inject(base, [DriftSpec(kind="label", speed="sudden", start=1, magnitude=1.0)])
    def mix(y):
        c = np.bincount(y, minlength=5)[1:4]
        return c / c.sum()
    assert np.abs(mix(out[-1].y) - mix(base[-1].y)).sum() > 0.05


def test_concept_injection_changes_mapping_not_features():
    base = make_batches(n=3, rows=300)
    out = inject(base, [DriftSpec(kind="concept", speed="sudden", start=1, magnitude=1.0)])
    assert np.allclose(out[-1].X, base[-1].X)
    assert not np.array_equal(out[-1].y, base[-1].y)


def test_inject_does_not_mutate_input():
    base = make_batches(n=3)
    snapshot = [b.X.copy() for b in base]
    inject(base, [DriftSpec(kind="covariate", magnitude=3.0)])
    for b, snap in zip(base, snapshot):
        assert np.allclose(b.X, snap)
