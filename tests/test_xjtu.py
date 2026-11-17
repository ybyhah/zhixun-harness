import numpy as np
import pytest

from harness.datasets.base import TaskAdapter
from harness.datasets.vibration import FEATURE_NAMES
from harness.datasets.xjtu import CLASS_SET, XJTUAdapter, find_fpt, label_bearing
from harness.monitor.performance import evaluate

LIVES = {"1_1": 120, "1_2": 60, "2_1": 42, "2_2": 80, "3_1": 100}
COND = {"1": 0, "2": 1, "3": 2}


def _rms_curve(n, knee, rng):
    """平稳段 + knee 之后指数上升。"""
    t = np.arange(n)
    r = 1.0 + rng.normal(scale=0.02, size=n)
    r[knee:] += np.exp((t[knee:] - knee) / (n - knee) * 3) - 0.9
    return r


@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(0)
    rows, bearing, cond, minute, life = [], [], [], [], []
    rms_col = FEATURE_NAMES.index("h_rms")
    for b, n in LIVES.items():
        rms = _rms_curve(n, int(n * 0.5), rng)
        order = rng.permutation(n)                       # 打乱，验证适配器会按分钟排序
        for i in order:
            x = rng.normal(size=len(FEATURE_NAMES)) + COND[b[0]] * 3
            x[rms_col] = rms[i]
            rows.append(x)
            bearing.append(b)
            cond.append(COND[b[0]])
            minute.append(i + 1)
            life.append(n)
    p = tmp_path / "f.npz"
    np.savez(p, X=np.asarray(rows, dtype=np.float32), feature_names=np.asarray(FEATURE_NAMES),
             bearing=np.asarray(bearing), condition=np.asarray(cond),
             minute=np.asarray(minute), life=np.asarray(life),
             missing=np.asarray([], dtype=str))
    return p


def test_find_fpt_detects_knee():
    rng = np.random.default_rng(1)
    i = find_fpt(_rms_curve(200, 120, rng))
    assert i is not None and 115 <= i <= 140


def test_find_fpt_none_on_flat():
    assert find_fpt(np.ones(100)) is None


def test_labels_monotone_and_severe_present():
    rng = np.random.default_rng(2)
    n = 150
    frac = np.arange(1, n + 1) / n
    y, fpt, src = label_bearing(_rms_curve(n, 80, rng), frac)
    assert src == "changepoint"
    assert np.all(np.diff(y) >= 0)
    assert set(np.unique(y)) == {0, 1, 2}
    assert np.all(y[frac >= 0.9] == 2)


def test_fallback_when_no_changepoint():
    n = 50
    frac = np.arange(1, n + 1) / n
    y, fpt, src = label_bearing(np.ones(n), frac, fallback_fpt=0.6)
    assert src == "fallback" and fpt == 0.6
    assert np.all(y[frac < 0.6] == 0)


def test_fpt_clipped_in_pure_life_mode():
    n = 100
    frac = np.arange(1, n + 1) / n
    _, fpt, _ = label_bearing(np.ones(n), frac, severe_frac=0.9, fallback_fpt=0.95,
                              severe_after_fpt=False)
    assert fpt == 0.9


def test_severe_starts_after_late_fpt():
    """突发失效：FPT 晚于 severe_frac 时，FPT 之前不应标 Severe。"""
    n = 200
    rng = np.random.default_rng(4)
    frac = np.arange(1, n + 1) / n
    rms = _rms_curve(n, 190, rng)                   # 拐点在 95%
    y, fpt, _ = label_bearing(rms, frac, severe_frac=0.9)
    assert fpt > 0.9
    assert np.all(y[frac < fpt] == 0)
    assert np.all(y[frac >= fpt] == 2)
    y_old, _, _ = label_bearing(rms, frac, severe_frac=0.9, severe_after_fpt=False)
    assert (y_old == 2).sum() > (y == 2).sum()


def test_short_life_does_not_fail():
    n = 42
    frac = np.arange(1, n + 1) / n
    y, _, _ = label_bearing(_rms_curve(n, 20, np.random.default_rng(3)), frac)
    assert (y == 2).sum() > 0


def test_condition_mode(cache):
    a = XJTUAdapter(cache, stream_mode="condition")
    assert isinstance(a, TaskAdapter)
    batches = a.load_batches()
    assert len(batches) == 5
    assert a.batch_bearings == ["1_1", "1_2", "2_1", "2_2", "3_1"]
    assert [b.index for b in batches] == list(range(5))
    assert sum(len(b) for b in batches) == sum(LIVES.values())
    for b in batches:
        assert np.all(np.diff(b.y) >= 0)                 # 按分钟排序后标签单调
        assert set(np.unique(b.y)) <= {0, 1, 2}
    # 跨工况批次的特征分布确实改变
    assert batches[4].X.mean() - batches[0].X.mean() > 3
    assert a.label_info["1_1"]["n"] == 120


def test_within_mode(cache):
    batches = XJTUAdapter(cache, stream_mode="within", single_condition=1).load_batches()
    assert len(batches) == 2


def test_missing_cache_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_xjtu_features"):
        XJTUAdapter(tmp_path / "none.npz").load_batches()


def test_bad_stream_mode(cache):
    with pytest.raises(ValueError):
        XJTUAdapter(cache, stream_mode="x")


# ---------- 严重退化漏报（关键类召回） ----------

def test_critical_recall_drops_when_severe_missed():
    y = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    good = evaluate(y, y.copy(), CLASS_SET, critical_class=2)
    miss = y.copy()
    miss[4:7] = 1                                      # 4 个 Severe 漏判 3 个
    bad = evaluate(y, miss, CLASS_SET, critical_class=2)
    assert good.critical_recall == pytest.approx(1.0)
    assert bad.critical_recall == pytest.approx(0.25)
    assert bad.to_dict()["critical_recall"] == pytest.approx(0.25)


def test_critical_recall_nan_when_absent_or_unset():
    y = np.array([0, 1, 1])
    assert np.isnan(evaluate(y, y, CLASS_SET, critical_class=2).critical_recall)
    r = evaluate(np.array([0, 2]), np.array([0, 2]), CLASS_SET)
    assert np.isnan(r.critical_recall)
    assert r.macro_f1 == pytest.approx(1.0)           # 未配置时原有字段不受影响
