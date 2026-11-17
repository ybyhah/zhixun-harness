# 外部开源基线对比评测完成总结

## 做了什么

清理了 `data/raw_xjtu/extracted/`（1.9GB 不完整旧解压），释放约 2.4GB。
引入 river / frouros / nannyml 与两个参照仓库，在 A1/A2/A3 同口径下与现有 Harness 对比，产出 `outputs/COMPARISON.md`。**未改 harness 主干代码**（用户指定 bench-only）。

## 三条核心结论

1. **规则闭环三个任务全面优于标准在线学习器**（综合性能）：
   | 任务 | E2 | River HAT | River ARF |
   |---|---|---|---|
   | A1 | 0.6079 | 0.5088 | 0.1690 |
   | A2 | 0.5861 | 0.4028 | 0.3763 |
   | A3 | -22.07 | -42.73 | -24.67 |
2. **成熟漂移检测库几乎每批告警**，误报远多于我们；我们的 monitor 是"是否需要动作"的决策器，更克制。
3. **门控偏乐观被量化**：A1 平均高估 +0.38，A2 +0.52（门控分 vs 下一批真实分）。

## 新增/修改文件

| 文件 | 内容 |
|---|---|
| `scripts/bench_external.py` | 新增，三维度对比（online/drift/estimate），复用 harness 适配器与指标 |
| `outputs/COMPARISON.md` | 新增，对比报告 + 下一轮优化清单 |
| `tests/test_bench_external.py` | 新增 4 个纯函数测试 |
| `requirements.txt` | 增加"仅本地对比"注释段（river/nannyml，注释掉不默认装） |
| `pytest.ini` | `pythonpath` 增加 `scripts` |
| `.gitignore` | 新增，忽略 `third_party/` |
| `third_party/aideml`、`third_party/Rotating-machine-fault-data-set` | clone 的只读参照，不进镜像 |

测试 216 个全部通过。harness `src/` 一行未改。

## 与 doc/tasks 的偏差

- **Task 4 未用 NannyML CBPE 的完整 API**：改用"当前批内分层留出集打分 vs 下一批真实分"作为门控偏乐观的简化估计。原因是 CBPE 的 reference/analysis chunk 接口与我们"逐批、提交后给标签"的流式口径不直接吻合，强行套用会引入与主闭环不一致的新口径。简化估计直接复用 harness 的 `IndustrialModel` 和 `evaluate`，口径一致、结论可信，符合 doc 第 6 节的 fallback 约定。nannyml 已安装备用。
- **参照仓库 5.2**：选的 `hustcxl/Rotating-machine-fault-data-set` 是数据集索引目录，无方法代码，没有可直接对比的特征/标签方案。但从中拿到物理事实（UER204 轴承、三工况转速），据此指出我们 A2 频域特征没有锁定故障特征频率（BPFO/BPFI/BSF/FTF）这一缺口。aideml 收获充分，归纳了 7 条第二阶段大模型慢通路的设计借鉴。

## 下一轮优化清单（本轮未实施，按预计收益排序）

1. 门控改用"上一批数据"或无标签性能估计，缓解 +0.38/+0.52 高估。
2. monitor 接入 ADWIN 快速触发信号（frouros 现成），做降漏报对照。
3. 门控加入 Severe 召回检查（承接 A2 结论）。
4. A2 频域特征锁定 BPFO/BPFI/BSF/FTF，试峭度触发 FPT。
5. 第二阶段大模型慢通路按 AIDE 的实验树 + 双角色评审 + 原子假设设计。

## 注意事项

- river/nannyml 安装时把 numpy 升到 2.2.6，frouros 声明需要 <2.2，但实测 frouros 仍可导入、全部测试通过。若要严格对齐 frouros 的版本声明，可在干净环境单独管理对比依赖。
- `third_party/` 和对比库都不得进入提交镜像（C4/C5）。报告已声明许可证：river/frouros BSD-3、nannyml Apache-2.0、aideml MIT。
