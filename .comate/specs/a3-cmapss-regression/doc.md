# A3 C-MAPSS 回归任务适配

> 赛题第三个公开数据集：NASA C-MAPSS 发动机剩余寿命（RUL）预测。
> 目标：把回归任务接入现有的流式自进化闭环，与 A1（分类）共用同一套 Harness。
> 数据位置：`data/CMAPSSData/`（FD001~FD004，各含 train/test/RUL）。

## 1. 为什么需要这个 spec

现有闭环（`src/harness/`）是按**分类**写死的：模型是分类器、指标是 Macro-F1、
漂移判定表基于"性能下降 + 类别占比变化"。C-MAPSS 是**回归**任务，三处都不适用：

| 维度 | A1 分类 | A3 回归 |
|---|---|---|
| 预测目标 | 6 类气体 | 连续值 RUL（剩余循环数） |
| 指标 | Accuracy / Macro-F1（越高越好） | RMSE / 非对称评分（**越低越好**） |
| 标签漂移 | 类别占比变化 | 不适用，改为**目标值分布漂移** |
| 模型 | HistGradientBoostingClassifier | HistGradientBoostingRegressor |

"越低越好"会让门控、参照线、恢复速度的**符号全部反向**，是本次适配的最大风险点。

## 2. 核心设计：C-MAPSS 怎么构造成漂移流

C-MAPSS 不像 A1 自带时间批次，需要人为构造。方案如下（**待用户确认**）：

### 2.1 批次划分方案（推荐方案 A）

**方案 A：四子集渐进漂移（推荐）**

把 FD001→FD003→FD002→FD004 按"复杂度递增"排成一条流，每个子集内再按发动机
分组切成若干批次：

| 阶段 | 子集 | 工况数 | 故障模式 | 相对前一阶段的漂移 |
|---|---|---|---|---|
| 1 | FD001 | 1 | 1（HPC） | 起点 |
| 2 | FD003 | 1 | 2（HPC+Fan） | 故障模式漂移 |
| 3 | FD002 | 6 | 1（HPC） | 工况漂移（最剧烈） |
| 4 | FD004 | 6 | 2 | 工况+故障双重漂移 |

这条顺序天然覆盖赛题点名的"工况切换"漂移，适合做对照实验。

**方案 B：单子集内按发动机编号切批**

只用 FD001，按 unit 号把 100 台发动机切成 10 批，模拟"新设备陆续上线"。
更接近 A1 的形态，但漂移较弱（单一工况）。

**方案 C：按工况切批（仅 FD002/FD004）**

利用 6 种工况，每种工况一个批次，直接构造工况漂移。

推荐先做方案 A，它最贴赛题；方案 B 作为简单对照。**最终以官方 B 包的下发方式为准**，
本地构造只为开发和对照实验。

### 2.2 RUL 标签处理

- 训练集：每台发动机运行到失效，RUL = 最大循环数 − 当前循环数
- 常规做法是对 RUL 做**分段线性截断**（RUL 超过阈值 R_early 时截断为 R_early，
  默认 125），因为早期退化不明显，预测远期 RUL 无意义
- 测试集：末尾循环对应 `RUL_FDxxx.txt` 里的真值

### 2.3 特征

26 列：unit、cycle、3 个工况设置、21 个传感器。
- 固定无效传感器（部分传感器全程常数，需剔除）
- 可加滑动窗口统计量（均值、斜率），但第一版先用单循环原始传感器值 + 工况设置，保持简单

## 3. 实现方式：扩展而非改写

现有分类代码必须保持可用（A1 已跑通）。通过**任务类型分支**扩展：

- `TaskAdapter.task_type` 已有 `"regression"` 枚举，补齐回归分支
- 新增 `RegressionModel`（或给 `IndustrialModel` 加 `task_type` 参数）
- 指标层新增回归指标，`score_model` 统一为"越大越好"的方向（RMSE 取负）
- 漂移判定表增加回归变体

### 3.1 指标方向统一（关键）

门控、参照线、恢复速度全部假设"分数越大越好"。回归 RMSE 越小越好，为避免
符号散落各处，**统一在指标层把回归分数转成"越大越好"**：

```
score = -RMSE          # 对外暴露的统一分数，越大越好
```

这样门控、参照线、`metrics.py` 的全部逻辑无需改符号。对外报告时再转回正的 RMSE。

### 3.2 非对称评分

C-MAPSS 官方评分：预测偏晚（RUL 高估）惩罚更重（晚发现故障更危险）。

```
d = RUL_pred - RUL_true
score_i = exp(-d/13) - 1   if d < 0   (提前，惩罚轻)
score_i = exp(d/10) - 1    if d >= 0  (滞后，惩罚重)
总分 = sum(score_i)，越低越好
```

作为**第二指标**与 RMSE 并列报告，不作为门控主判据（门控用 -RMSE）。

## 4. 影响文件

```
修改 src/harness/datasets/base.py          TaskAdapter 回归分支说明
新增 src/harness/datasets/cmapss.py         C-MAPSS 适配器 + 批次构造
修改 src/harness/models/industrial.py       支持 regression（回归器 + proba 分支隔离）
修改 src/harness/monitor/performance.py     回归指标 RMSE / 非对称评分 / 统一 score
修改 src/harness/monitor/report.py          回归漂移判定（目标分布漂移替代标签漂移）
修改 src/harness/monitor/distribution.py    目标值分布漂移（KS on y）
修改 src/harness/decide/rules.py            回归动作表（校准/阈值不适用，改用窗口/全量重训）
修改 src/harness/loop.py                     按 task_type 选择模型与判定分支
新增 configs/a3_cmapss.yaml                 A3 配置
新增 scripts/run_baseline_a3.py             A3 静态基线（或给现有脚本加 --config）
新增 scripts/run_loop_a3.py                 A3 闭环（或复用 run_loop.py）
新增 tests/test_cmapss.py                   适配器与回归指标测试
```

## 5. 边界条件

| 情况 | 处理 |
|---|---|
| 无效传感器（全程常数） | 加载时剔除，记录剔除列表 |
| RUL 截断阈值 | 配置项 `rul_cap`，默认 125 |
| 测试集只给末尾 RUL | 批次划分用训练轨迹；评测对齐官方 RUL 真值 |
| 不同子集工况数不同 | 工况设置作为特征输入，让模型自适应 |
| 回归无"类别缺失"问题 | 分类专用的缺类逻辑在回归分支跳过 |
| 分数方向 | 统一 -RMSE，仅在报告层转回正值 |

## 6. 预期结果

1. A3 静态基线：FD001→FD004 漂移流上，RMSE 随工况漂移显著上升（验证漂移存在）
2. A3 规则闭环：综合 RMSE 低于静态基线，恢复速度指标为正
3. 全程无人工干预跑完，与 A1 共用同一 `SelfEvolvingLoop`，仅配置不同
4. A1 的全部测试仍通过（扩展不破坏分类路径）
5. 产出 A3 的 E0/E2 对照与审计记录

**验收**：`python scripts/run_loop.py --config configs/a3_cmapss.yaml` 可跑通并产出对照指标；
`python -m pytest` 全绿。
