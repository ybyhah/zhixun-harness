# 外部开源基线对比评测

> 目标：引入相似开源项目作为**外部基线与参照**，在 A1/A2/A3 三个任务上与现有 Harness 对比，量化差距，产出对比报告。
> 本轮**不改 harness 主干代码**（用户指定 bench-only）。只新增对比脚本与报告，发现的问题写成下一轮优化清单。

## 0. 约束与前提

- 评测环境离线（C4），但本地开发可联网。安装 river/nannyml、clone 参考仓库需要联网，执行时请求升级权限。
- 这些外部库**只用于本地对比**，不进入提交镜像，避免违反"离线 + 镜像 ≤2GB"。报告里声明清楚（C14 合规）。
- 已装：frouros 0.9.0、sklearn 1.9.0、scipy 1.14.1。缺：river、nannyml。

## 1. 四个对比维度

### 1.1 漂移检测对比（River + Frouros vs 自写 monitor）

把三个任务逐批的 0/1 误差流（分类）和特征流喂给外部检测器，对比它们告警的批次与我们 `monitor` 判定 drift 的批次是否一致。

| 检测器 | 库 | 输入 |
|---|---|---|
| ADWIN | river / frouros | 误差流 |
| KSWIN | river | 误差流 |
| DDM / HDDM | frouros | 误差流 |
| Page-Hinkley | river | 误差流 |

回答：我们的 PSI+KS+PCA+ADWIN 组合，漏报/误报相比成熟库如何。

### 1.2 增量在线学习基线（River vs E0/E2）

用 River 的 `HoeffdingAdaptiveTreeClassifier`、`ARFClassifier`（自适应随机森林）作为"持续在线学习"基线，逐批 `predict_many`→评分→`learn_many`。与 E0 静态、E2 规则闭环在同一批次流、同一指标上比。

回答：标准在线学习器（无门控、无回滚）能否打过我们"有依据才更新"的闭环。这正是赛题强调的点，是很强的答辩对照。回归任务用 River 回归器同法。

### 1.3 无标签性能估计（NannyML vs 门控分数）

A1/A2 用 NannyML 的 CBPE（基于置信度的性能估计）在"下一批"上估 Macro-F1/ROC-AUC，三方对比：
- 我们门控给候选打的分（当前批内验证，已知偏乐观）
- NannyML 的估计
- 下一批真实分数

回答：门控偏乐观有多严重，NannyML 式估计能否作为更可信的门控信号（为下一轮 gate 改造提供依据）。

### 1.4 参照仓库（只读，不执行）

clone 后阅读对比，不跑：
- `WecoAI/aideml`（MIT）：LLM 实验 Agent 的树搜索与历史利用，参照第二阶段大模型慢通路设计。
- 一个 XJTU-SY 轴承特征提取/诊断仓库：对比它的特征集合与标签方案 vs 我们的 40 维 + 三分类。

## 2. 影响文件（全部新增，不改 harness）

```
新增 scripts/bench_external.py        四个维度的对比，按 --dim 选择运行
新增 outputs/COMPARISON.md            对比结论报告
新增 tests/test_bench_external.py     对脚本里纯函数（对齐、汇总）做轻量测试
修改 requirements.txt                 增加 river、nannyml 到"仅本地对比"注释段
新增 third_party/                     clone 的参照仓库（加入忽略，不进镜像）
```

harness 的 `src/` 一行不改。对比脚本复用现有 `build_adapter` / `IndustrialModel` / `metrics` 读取同一批次流，保证口径一致。

## 3. 脚本设计 `scripts/bench_external.py`

```
--dim drift      1.1 漂移检测对齐
--dim online     1.2 River 在线学习基线
--dim estimate   1.3 NannyML 性能估计
--dim all        全部
--config <cfg>   复用 configs/a1_gas.yaml 等
```

- 用现有 `build_adapter(cfg, ROOT).load_batches()` 拿批次，严格按批次顺序，提交后才评分，不偷看未来，与主闭环一致。
- 外部库缺失时该维度打印"未安装，跳过"，不让整个脚本崩。
- 每个维度输出一张对比表到 stdout，并把数值写进 `COMPARISON.md` 对应小节（人工整理结论）。

## 4. 边界与风险

| 情况 | 处理 |
|---|---|
| river/nannyml 装不上（网络/版本） | 该维度跳过，报告里注明缺库 |
| nannyml 需要概率输出 | `IndustrialModel` 分类可 `predict_proba`；回归任务 NannyML 用 DLE，若不便则只做 A1/A2 |
| River API 版本差异 | 固定 `river>=0.21`，脚本里做最小封装 |
| 外部库引入随机性 | 固定 seed；River 在线学习本身对顺序敏感，报告注明 |
| 参照仓库体积大 | `git clone --depth 1`；只读，不进 harness 镜像 |

## 5. 预期产出

1. `COMPARISON.md`：四维对比，含"我们相对成熟库的漏报/误报""在线学习基线是否更优""门控偏乐观的量化"三条结论。
2. 一份下一轮优化清单（基于对比发现，但本轮不实施）。
3. `python3 -m pytest` 仍全绿；harness 行为不变。

## 6. 待确认

- 参照仓库具体选哪个轴承仓库，执行时据搜索结果定（README 质量、是否含特征提取代码）。
- 若 nannyml 依赖过重装不动，1.3 退化为"用上一批数据做验证集"的简化估计，仍能量化门控偏乐观。
