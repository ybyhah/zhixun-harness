# A2 XJTU-SY 轴承退化识别任务适配

> 目标：把原始振动波形提成特征，接入现有 `SelfEvolvingLoop`，与 A1 分类、A3 回归共用同一 Harness，只换配置。
> 任务类型：三分类健康状态识别，指标 Macro-F1 + 严重退化漏报（赛题 A2 口径）。

## 1. 数据现状（2026-11-10 实测）

```
data/raw_xjtu/Data/XJTU-SY_Bearing_Datasets.part01..06.rar   RAR5 分卷，共 4.4GB
  35Hz12kN/    Bearing1_1..1_5   寿命 123/161/158/122/52 分钟
  37.5Hz11kN/  Bearing2_1..2_5   寿命 491/161/533/42/339
  40Hz10kN/    Bearing3_1..3_5   寿命 2538/2496/371/1515/114
  每个 CSV：表头 + 32768 行 × 2 列（水平、垂直振动），25.6kHz，约 1.2MB
```

- 共 9216 个 CSV，全量解压约 11GB；磁盘剩余 7.6GB，**不能全量解压**。
- `data/raw_xjtu/extracted/` 已有 1556 个 CSV（8 个轴承，Bearing2_3 只到 288/533），不完整，本方案不依赖它，也不删除它。
- 本机无 unrar/7z，只有 `/usr/bin/bsdtar`（libarchive）。实测 `bsdtar -tf` 能单独列出每一卷的文件；跨卷边界的少数文件单卷无法解出，按"缺失样本"跳过并记录。
- 未安装 pyarrow，缓存用 `numpy.savez_compressed`（npz）。

## 2. 方案总览

```
RAR 分卷 ──(scripts/extract_xjtu_features.py, 一次性)──▶ data/raw_xjtu/features.npz
              逐卷、逐轴承解到临时目录 → 提特征 → 删除临时 CSV
features.npz ──(XJTUAdapter.load_batches)──▶ 打标签 + 按工况切批 ──▶ 现有 BatchStream / SelfEvolvingLoop
```

闭环主干不改。新增内容：特征提取、标签方案、工况漂移批次流，外加一个通用的"关键类召回率"指标。

## 3. 特征提取 `src/harness/datasets/vibration.py`（新增）

每个 CSV（一分钟快照）压成一行。水平、垂直两通道各 20 维，共 40 维，只用 numpy + scipy。

| 组 | 特征 | 维数 |
|---|---|---|
| 时域 | 均值绝对值、RMS、峰值、峰峰值、标准差、偏度、峭度、波形因子、峰值因子、脉冲因子 | 10 |
| 频域 | 0–12.8kHz 均分 6 段的能量占比、重心频率、谱峰频率 | 8 |
| 包络谱 | Hilbert 包络（去均值）FFT 后 0–1kHz 内主峰幅值、主峰频率 | 2 |

```python
FS = 25600
TIME_NAMES = ["abs_mean", "rms", "peak", "p2p", "std", "skew", "kurt",
              "shape_factor", "crest_factor", "impulse_factor"]

def channel_features(x: np.ndarray, fs: int = FS, n_bands: int = 6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 16:                       # 空或过短：整行 NaN，交给 fill_missing 兜底
        return np.full(N_PER_CHANNEL, np.nan)
    eps = 1e-12
    abs_mean, rms, peak = np.mean(np.abs(x)), np.sqrt(np.mean(x**2)), np.max(np.abs(x))
    ...
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(x.size, 1 / fs)
    band_ratio = [spec[m].sum() / (spec.sum() + eps) for m in band_masks]
    env = np.abs(scipy.signal.hilbert(x)); env -= env.mean()
    ...

def extract_features(sig: np.ndarray) -> np.ndarray:   # sig: (n, 2)
    return np.concatenate([channel_features(sig[:, 0]), channel_features(sig[:, 1])])

FEATURE_NAMES = [f"{ch}_{n}" for ch in ("h", "v") for n in CHANNEL_NAMES]
```

边界：常数段时偏度、峭度、各因子由 `eps` 保护，不出 inf；长度不足 32768 按实际长度算，不补零。

## 4. 特征缓存 `scripts/extract_xjtu_features.py`（新增）

1. 对每一卷执行 `bsdtar -tf` 拿文件清单，按轴承分组。
2. 每卷每个轴承执行 `bsdtar -xf <卷> -C <tmp> <该轴承的文件>`，单个文件失败不影响其他文件（失败名单写入缓存元数据）。
3. 用 `numpy.loadtxt(skiprows=1, delimiter=",")` 读取并提特征，**处理完立即删除该临时目录**。峰值占用约等于单卷内最大轴承的体积（不到 2GB）。
4. 全部完成后写 `data/raw_xjtu/features.npz`：

| 键 | 内容 |
|---|---|
| `X` | (N, 40) float32 |
| `feature_names` | 40 个名字 |
| `bearing` | 如 `"2_3"` |
| `condition` | `0/1/2`，对应 35Hz/37.5Hz/40Hz |
| `minute` | CSV 文件名中的分钟序号 |
| `life` | 该轴承的寿命（官方表 + 清单中最大分钟取大） |
| `missing` | 解压或读取失败的文件名 |

参数：`--rar-dir`（默认 `data/raw_xjtu/Data`）、`--out`、`--bearings 1_1,2_4`（只跑部分轴承，用于调试）、`--force`。缓存已存在且未加 `--force` 时直接退出。脚本只在本地开发时运行，评测时闭环只读 npz。

## 5. 健康状态标签（`xjtu.py` 内）

标签 `0=Health / 1=Degrading / 2=Severe`，全局类别集合固定为 `[0, 1, 2]`。

```python
def label_bearing(rms: np.ndarray, life_frac: np.ndarray, severe_frac=0.9,
                  base_frac=0.1, k_sigma=3.0, consecutive=3, fallback_fpt=0.6):
    n = len(rms); n_base = max(10, int(n * base_frac))
    mu, sd = rms[:n_base].mean(), rms[:n_base].std()
    over = rms > mu + k_sigma * sd
    fpt = first index i >= n_base where over[i:i+consecutive] all True
    fpt_frac = life_frac[fpt] if found else fallback_fpt
    fpt_frac = min(fpt_frac, severe_frac)           # FPT 不能晚于严重阈值
    y = np.where(life_frac >= severe_frac, 2, np.where(life_frac >= fpt_frac, 1, 0))
```

- RMS 取水平通道 `h_rms`。寿命分数 `life_frac = minute / life`。
- 标签用到整条寿命，这是离线构造真值，等价于评测流"每批自带标签"，不属于预测时偷看未来。
- 每个轴承的 FPT 来源（变点或回退）写入 `adapter.label_info`，供结论引用。

## 6. 工况漂移批次流

| `stream_mode` | 批次构造 | 批次数 | 用途 |
|---|---|---|---|
| `condition`（默认） | 35Hz → 37.5Hz → 40Hz，每工况内一个轴承一批，按编号排序 | 15 | 主实验，工况切换（转速 + 载荷都变）是最强协变量漂移 |
| `within` | 只取 `single_condition` 一个工况，一个轴承一批 | 5 | 弱漂移对照 |

批 0 = Bearing1_1（123 条，含三类），用于冷启动和静态基线。批内按分钟排序。复用现有 `BatchStream`，提交后才给标签的约束不变。

## 7. 严重退化漏报指标（通用改动）

`monitor/performance.py`：
- `evaluate(..., critical_class=None)` 新增可选参数。分类时，若该类在本批出现，计算 `critical_recall`（该类召回率）；未出现则为 NaN。
- `PerfResult` 增加 `critical_recall` 字段，`to_dict` 一并输出。
- 严重退化漏报率 = `1 - critical_recall`。

`loop.py`：`self.critical_class = cfg.task.get("critical_class")`，两处 `evaluate` 调用传入，`_row` 增加 `critical_recall` 列。`scripts/run_loop.py` 的 `compute_e0` 同样传入，并在逐批表里打印 E0/E2 的 Severe 召回。A1/A3 不配置该项，默认 None，行为不变（只多一列 NaN）。

本期**只报告，不卡门控**。是否纳入门控等官方评分表发布后再定。

## 8. 适配器、工厂与配置

`src/harness/datasets/xjtu.py`（新增）`XJTUAdapter`，实现 `TaskAdapter` 协议：
- `task_type="classification"`，`metric_name="macro_f1"`，`class_set=np.array([0, 1, 2])`
- `__init__(cache_path, stream_mode="condition", single_condition=0, severe_frac=0.9, fpt_kwargs...)`
- `load_batches()`：读取 npz，缓存不存在时抛 `FileNotFoundError`，提示先运行 `python scripts/extract_xjtu_features.py`

`datasets/factory.py` 增加 `xjtu` 分支。

`configs/a2_xjtu.yaml`（新增）：以 `a1_gas.yaml` 为模板，改动如下：
- `task`：`name: a2_xjtu`、`adapter: xjtu`、`n_classes: 3`、`critical_class: 2`
- `data`：`cache: data/raw_xjtu/features.npz`、`stream_mode: condition`、`severe_frac: 0.9`
- `loop`：批次最小样本 42，`min_batch_size` 设为 30；`min_window_samples` 设为 300
- `record.audit_path`：`outputs/audit_a2.jsonl`

其余阈值先沿用 A1，跑通后再按结果调整，调参过程记入结论。

`scripts/run_loop.py` 已按 `--config` 驱动，输出 `e2_a2_xjtu.csv`、`summary_a2_xjtu.csv`、`curve_a2_xjtu.png`，不需要改结构。

## 9. 影响文件

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/harness/datasets/vibration.py` | 新增 | `channel_features`、`extract_features`、`FEATURE_NAMES` |
| `src/harness/datasets/xjtu.py` | 新增 | `label_bearing`、`XJTUAdapter` |
| `src/harness/datasets/factory.py` | 修改 | `build_adapter` 增加 `xjtu` 分支 |
| `src/harness/monitor/performance.py` | 修改 | `evaluate`、`PerfResult` 增加 `critical_recall` |
| `src/harness/loop.py` | 修改 | `__init__` 读 `critical_class`；`_after_submit`、`_row` 传参并输出 |
| `scripts/extract_xjtu_features.py` | 新增 | 逐卷解压、提特征、写缓存 |
| `scripts/run_loop.py` | 修改 | `compute_e0` 传 `critical_class`，打印 Severe 召回 |
| `configs/a2_xjtu.yaml` | 新增 | A2 配置 |
| `tests/test_vibration.py`、`tests/test_xjtu.py` | 新增 | 用合成信号和小 npz 测试，不依赖真实数据 |
| `outputs/CONCLUSIONS.md` | 修改 | 追加 A2 结论 |

## 10. 边界条件

| 情况 | 处理 |
|---|---|
| 跨卷边界文件单卷解不出 | 跳过并写入 `missing`；寿命按官方表，不受影响 |
| 磁盘不足 | 逐轴承解压、处理后立即删除临时目录；开始前检查剩余空间 ≥ 3GB，否则报错退出 |
| CSV 行数不足、含 NaN | 按实际长度计算；特征 NaN 由现有 `fill_missing` 处理 |
| 短寿命轴承（2_4 仅 42 分钟） | 基线段至少 10 点；找不到变点时回退 `fallback_fpt` |
| 某批缺类 | Macro-F1 已按本批出现的类别计算；Severe 未出现时召回为 NaN，汇总时跳过 |
| 门控验证集与训练集相邻分钟高度相关 | 门控分偏乐观，和 A1 是同一已知问题，本期不改，写入结论 |

## 11. 预期结果与验收

1. `python3 scripts/extract_xjtu_features.py` 产出 `features.npz`，打印每个轴承的样本数、缺失数与耗时。
2. E0 静态基线在进入 37.5Hz、40Hz 工况后 Macro-F1 明显下降。
3. E2 规则闭环的综合 Macro-F1 高于 E0，Severe 召回不低于 E0。
4. A1、A3 配置跑出的结果与改动前一致，`python3 -m pytest` 全部通过。

## 12. 待确认（不阻塞，先按默认做）

- 官方若提供 A2 预提取特征或批次化版本，适配器改读官方文件，本地提特征只保留作调试用。
- 类别数与阈值（3 类、`severe_frac=0.9`）以官方评测脚本为准，已做成配置项。
- 评测镜像里不需要 bsdtar：提特征只在本地运行，镜像只带 npz 或官方数据。
