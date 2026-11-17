# A2 XJTU-SY 轴承适配完成总结

## 结果

| 指标 | E0 静态基线 | E2 规则闭环 |
|---|---|---|
| 综合 Macro-F1 | 0.5203 | 0.5861 |
| 最差阶段 | 0.3530 | 0.3786 |
| Severe 平均召回 | 0.742 | 0.699 |
| 训练 / 上线 | 1 / 0 | 13 / 5 |
| 总耗时 / 单批最大 | — | 46.6s / 9.3s |

- 完整闭环记录 6 条，0 降级事件，固定种子可复现。
- 测试 203 个全部通过（新增 23 个）。A1、A3 重跑，逐批分数与改动前一致。

## 新增与修改

| 文件 | 内容 |
|---|---|
| `src/harness/datasets/vibration.py` | 新增，双通道 40 维振动特征 |
| `src/harness/datasets/xjtu.py` | 新增，FPT 变点、三分类标签、`condition` / `within` 两种批次流 |
| `scripts/extract_xjtu_features.py` | 新增，逐卷逐轴承解压 → 提特征 → 删除临时文件，9216 个 CSV 0 缺失，约 7 分钟 |
| `configs/a2_xjtu.yaml` | 新增 |
| `src/harness/datasets/factory.py` | 增加 `xjtu` 分支 |
| `src/harness/monitor/performance.py`、`loop.py`、`scripts/run_loop.py` | 增加通用的 `critical_recall`（关键类召回），只报告不卡门控 |
| `tests/test_vibration.py`、`tests/test_xjtu.py` | 新增，全部用合成数据 |
| `outputs/CONCLUSIONS.md` | 追加 A2 小节 |
| `data/raw_xjtu/features.npz` | 特征缓存，1.3MB |

## 与 doc.md 的偏差

- **标签口径**：增加 `severe_after_fpt`（默认开启），Severe 起点取 max(寿命 90%, FPT)。起因是 4 个突发失效轴承在寿命 90% 处 RMS 仍为健康水平，按纯寿命分数标 Severe 属于标签噪声。修正后 E2 综合性能从 0.5716 升到 0.5861。
- **未达成的预期**：
  - 预期 2 未成立：E0 在工况切换处没有明显下跌，轴承之间的差异比工况漂移大。
  - 预期 3 部分未成立：E2 的 Severe 召回比 E0 低 0.043。
  - 两项原因都已写入结论。

## 遗留问题

1. 门控没有纳入 Severe 召回。建议下一步加一道"Severe 召回不低于线上模型"的检查，再做对照实验。
2. `run_loop.py --set data.*` 不会生效：适配器在覆盖配置之前就由 `base_cfg` 构造了。这是原有行为，本期未改。做标签口径对照需要直接改配置文件。
3. 门控验证集从当前轴承随机切分，相邻分钟样本高度相关，门控分偏乐观（和 A1 是同一问题）。
4. `data/raw_xjtu/extracted/`（1.9GB 不完整的旧解压结果）已不再需要，未删除，可自行清理。
