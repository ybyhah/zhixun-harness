"""振动信号特征提取：一段原始波形压成一行特征向量。

用于 A2 XJTU-SY 轴承数据（25.6kHz，每分钟一段 32768 点，水平/垂直两通道）。
只依赖 numpy + scipy，CPU 可算。每通道 20 维：时域 10、频域 8、包络谱 2。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

FS = 25600
N_BANDS = 6
ENV_MAX_HZ = 1000.0
EPS = 1e-12

TIME_NAMES = ["abs_mean", "rms", "peak", "p2p", "std", "skew", "kurt",
              "shape_factor", "crest_factor", "impulse_factor"]
FREQ_NAMES = [f"band{i}_ratio" for i in range(N_BANDS)] + ["centroid_hz", "peak_hz"]
ENV_NAMES = ["env_peak_amp", "env_peak_hz"]
CHANNEL_NAMES = TIME_NAMES + FREQ_NAMES + ENV_NAMES
N_PER_CHANNEL = len(CHANNEL_NAMES)
FEATURE_NAMES = [f"{ch}_{n}" for ch in ("h", "v") for n in CHANNEL_NAMES]


def _time_features(x: np.ndarray) -> list[float]:
    abs_mean = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x))
    if std < EPS:                       # 常数段：高阶矩无定义，记 0
        sk, ku = 0.0, 0.0
    else:
        sk, ku = float(skew(x)), float(kurtosis(x, fisher=False))
    return [
        abs_mean, rms, peak, float(np.ptp(x)), std, sk, ku,
        rms / (abs_mean + EPS), peak / (rms + EPS), peak / (abs_mean + EPS),
    ]


def _freq_features(x: np.ndarray, fs: int) -> list[float]:
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    total = spec.sum() + EPS
    edges = np.linspace(0.0, fs / 2.0, N_BANDS + 1)
    ratios = []
    for i in range(N_BANDS):
        hi = freqs <= edges[i + 1] if i == N_BANDS - 1 else freqs < edges[i + 1]
        ratios.append(float(spec[(freqs >= edges[i]) & hi].sum() / total))
    centroid = float((freqs * spec).sum() / total)
    peak_hz = float(freqs[int(np.argmax(spec))]) if spec.sum() > EPS else 0.0
    return ratios + [centroid, peak_hz]


def _envelope_features(x: np.ndarray, fs: int) -> list[float]:
    env = np.abs(hilbert(x - x.mean()))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env)) / env.size
    freqs = np.fft.rfftfreq(env.size, 1.0 / fs)
    band = (freqs > 0) & (freqs <= ENV_MAX_HZ)
    if not band.any():
        return [0.0, 0.0]
    k = int(np.argmax(spec[band]))
    return [float(spec[band][k]), float(freqs[band][k])]


def channel_features(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """单通道 20 维特征。空、过短或全为非有限值时返回 NaN 行。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 16:
        return np.full(N_PER_CHANNEL, np.nan)
    feats = _time_features(x) + _freq_features(x, fs) + _envelope_features(x, fs)
    return np.asarray(feats, dtype=np.float64)


def extract_features(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """sig 形状 (n, 2)：第 0 列水平、第 1 列垂直。返回 40 维向量。"""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] < 2:
        return np.full(len(FEATURE_NAMES), np.nan)
    return np.concatenate([channel_features(sig[:, 0], fs), channel_features(sig[:, 1], fs)])
