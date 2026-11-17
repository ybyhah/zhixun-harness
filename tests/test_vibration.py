import numpy as np
import pytest

from harness.datasets.vibration import (
    CHANNEL_NAMES, FEATURE_NAMES, FS, channel_features, extract_features,
)

N = 32768


def _idx(name: str) -> int:
    return CHANNEL_NAMES.index(name)


def _sine(freq=1000.0, amp=2.0):
    t = np.arange(N) / FS
    return amp * np.sin(2 * np.pi * freq * t)


def test_dimension_fixed():
    assert len(FEATURE_NAMES) == 40
    sig = np.column_stack([_sine(), _sine(2000.0, 1.0)])
    assert extract_features(sig).shape == (40,)


def test_sine_rms_and_peak_frequency():
    f = channel_features(_sine(1000.0, 2.0))
    assert f[_idx("rms")] == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)
    assert f[_idx("peak_hz")] == pytest.approx(1000.0, abs=1.0)
    assert f[_idx("crest_factor")] == pytest.approx(np.sqrt(2), rel=1e-2)
    # 1000Hz 落在第 0 段（0~2133Hz）
    assert f[_idx("band0_ratio")] > 0.99


def test_band_ratios_sum_to_one():
    rng = np.random.default_rng(0)
    f = channel_features(rng.normal(size=N))
    ratios = [f[_idx(f"band{i}_ratio")] for i in range(6)]
    assert sum(ratios) == pytest.approx(1.0, abs=1e-6)


def test_impulses_raise_kurtosis_and_envelope_peak():
    rng = np.random.default_rng(1)
    base = rng.normal(scale=0.1, size=N)
    imp = base.copy()
    imp[:: FS // 100] += 5.0                      # 100Hz 冲击序列，模拟故障
    f0, f1 = channel_features(base), channel_features(imp)
    assert f1[_idx("kurt")] > f0[_idx("kurt")] * 5
    assert f1[_idx("env_peak_hz")] == pytest.approx(100.0, abs=2.0)


def test_constant_signal_is_finite():
    f = channel_features(np.full(N, 0.5))
    assert np.all(np.isfinite(f))


@pytest.mark.parametrize("bad", [np.array([]), np.ones(5), np.full(100, np.nan)])
def test_bad_input_returns_nan_row(bad):
    f = channel_features(bad)
    assert f.shape == (20,) and np.all(np.isnan(f))


def test_nan_values_are_dropped():
    x = _sine()
    x[::100] = np.nan
    assert np.all(np.isfinite(channel_features(x)))


def test_wrong_shape_returns_nan():
    assert np.all(np.isnan(extract_features(np.ones(100))))
