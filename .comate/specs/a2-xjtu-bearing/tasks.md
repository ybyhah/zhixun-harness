# A2 XJTU-SY 轴承退化识别适配任务计划

- [x] Task 1: 振动信号特征提取 `datasets/vibration.py`
    - 1.1: 实现单通道时域 10 维特征（均值绝对值、RMS、峰值、峰峰值、标准差、偏度、峭度、波形/峰值/脉冲因子），常数段用 eps 保护
    - 1.2: 实现频域 8 维特征（6 段能量占比、重心频率、谱峰频率）
    - 1.3: 实现包络谱 2 维特征（Hilbert 包络去均值后 0–1kHz 主峰幅值与频率）
    - 1.4: 实现 `extract_features(sig)` 拼接双通道 40 维，导出 `FEATURE_NAMES`
    - 1.5: 空、过短、含 NaN 输入返回 NaN 行而不报错
    - 1.6: 新增 `tests/test_vibration.py`：正弦信号 RMS≈A/√2、谱峰频率正确、维度固定为 40、异常输入不崩

- [x] Task 2: 特征缓存脚本 `scripts/extract_xjtu_features.py`
    - 2.1: 用 `bsdtar -tf` 列出每卷文件并按轴承分组，记录官方寿命表
    - 2.2: 逐卷逐轴承解压到临时目录，读取、提特征后立即删除临时目录；单个文件失败记入 `missing`
    - 2.3: 开始前检查磁盘剩余空间 ≥ 3GB；支持 `--rar-dir`、`--out`、`--bearings`、`--force`
    - 2.4: 写出 `features.npz`（X、feature_names、bearing、condition、minute、life、missing），打印每轴承样本数、缺失数与耗时
    - 2.5: 先用 `--bearings 1_5,2_4` 小规模试跑验证

- [x] Task 3: 标签与适配器 `datasets/xjtu.py`
    - 3.1: 实现 `label_bearing`：RMS 3σ 连续超限确定 FPT，失败回退 `fallback_fpt`，FPT 不晚于 `severe_frac`
    - 3.2: 实现 `XJTUAdapter`：读 npz、打标签、`class_set=[0,1,2]`，缓存缺失时提示先运行提取脚本
    - 3.3: 实现 `condition`（15 批）与 `within`（5 批）两种批次流，批内按分钟排序
    - 3.4: 记录每个轴承的 FPT 来源到 `label_info`
    - 3.5: 新增 `tests/test_xjtu.py`：用合成小 npz 测试标签单调推进、短寿命轴承不报错、Severe 非空、两种模式批次数与顺序正确、满足 `TaskAdapter` 协议

- [x] Task 4: 严重退化漏报指标
    - 4.1: `monitor/performance.py` 的 `evaluate` 增加 `critical_class` 参数，`PerfResult` 增加 `critical_recall`
    - 4.2: `loop.py` 读 `task.critical_class`，两处 `evaluate` 传入，`_row` 增加 `critical_recall` 列
    - 4.3: `scripts/run_loop.py` 的 `compute_e0` 传入该参数，逐批表打印 E0/E2 的 Severe 召回
    - 4.4: 补充测试：漏判 Severe 时召回下降、该类未出现时为 NaN、未配置时 A1/A3 行为不变

- [x] Task 5: 工厂与配置接入
    - 5.1: `datasets/factory.py` 增加 `xjtu` 分支
    - 5.2: 新增 `configs/a2_xjtu.yaml`（3 类、`critical_class: 2`、`stream_mode: condition`、`min_window_samples: 300`、`audit_a2.jsonl`）
    - 5.3: 运行完整测试套件，确认全部通过

- [x] Task 6: 全量特征提取与 A2 实验
    - 6.1: 全量运行提取脚本生成 `features.npz`，核对 15 个轴承样本数与缺失数
    - 6.2: 运行 `run_loop.py --config configs/a2_xjtu.yaml`，得到 E0/E2 对照、Severe 召回、训练与上线次数
    - 6.3: 若结果明显异常（如 E2 不优于 E0），排查标签与阈值并记录调整依据
    - 6.4: 重复运行一次确认固定种子可复现
    - 6.5: 重跑 A1、A3，确认结果与改动前一致

- [x] Task 7: 更新结论文档
    - 7.1: 在 `outputs/CONCLUSIONS.md` 追加 A2 小节：数据处理、标签方案、E0/E2 指标、Severe 漏报、已知问题
    - 7.2: 更新三任务状态表
