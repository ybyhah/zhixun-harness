# 工业模型自进化系统：第一阶段（MVP 骨架）

> 赛题：智训引擎 — 工业制造场景下的模型训练与超参数优化
> 本文档只覆盖**第一阶段**。大模型慢通路、策略层自改进、A2/A3 任务扩展、镜像打包属于后续阶段，另立 spec。
> 规则依据见 `赛题规则解析.md`，整体路线见 `方案.md`。

## 1. 本阶段目标

搭出一套**不依赖大模型**就能完整运行的流式自适应闭环，并在 A1（气体传感器漂移）上跑出可信的对照结果。

它要同时承担三个作用：

1. **保门槛**：无人干预跑完、记录完整、性能不低于静态基线。这三条是排名资格，必须先拿到。
2. **做底座**：后续大模型慢通路、策略自改进都挂在这套骨架上，接口现在就定好。
3. **出证据**：产出 E0（静态基线）和 E2（规则触发）两组对照数据，这是后面论证"大模型带来了什么"的参照系。

**本阶段明确不做**：调用大模型、策略层自改进、A2/A3、Docker 镜像、PyTRIO 接入。

## 2. 需求场景与处理逻辑

### 2.1 场景

系统面对一条按批次下发的工业数据流。A1 共 10 个批次，跨 36 个月，传感器老化导致特征分布持续偏移，同时各批次类别分布极不均衡（批次 3、4、5 完全没有 Toluene 样本，批次 10 每类 600 个）。

评测是串行的：必须对当前批次给出输出或下线判断，才能拿到下一批。

### 2.2 单批次处理逻辑

```
批次 t 到达
  │
  ├─ 1. 用当前在线模型预测，立即输出        ← 不阻塞
  │
  ├─ 2. 取得本批真实标签（赛题确认直接给）
  │
  ├─ 3. 监测：真实性能 + 分布漂移 + 误差流突变
  │       └→ 产出 DriftReport（含漂移类型判定）
  │
  ├─ 4. 决策：按规则表选动作（可能是"不动"）
  │
  ├─ 5. 若动作非"不动"：训练候选模型
  │
  ├─ 6. 门控：候选 vs 在线 vs 静态基线 vs 回放集
  │       ├→ 通过 → 候选上线，旧版本入历史
  │       └→ 不通过 → 丢弃候选；连续失败 → 回滚历史最优
  │
  └─ 7. 记录：决策记录写审计日志；训练过程写 SwanLab
```

关键原则：**第 1 步永远先执行**。任何后续环节异常都不影响本批次输出，保证闭环能跑完。

## 3. 架构与技术方案

### 3.1 分层

| 层 | 职责 | 是否依赖大模型 |
|---|---|---|
| 任务适配层 | 把不同数据集统一成"批次流 + 特征矩阵 + 标签" | 否 |
| 流与模拟层 | 串行批次下发；按需注入人工漂移 | 否 |
| 模型层 | 工业模型训练、预测、校准、版本管理 | 否 |
| 监测层 | 性能、分布、误差流三路信号，判定漂移类型 | 否 |
| 决策层 | 规则表选动作（本阶段）；后续挂大模型 | 否（预留接口） |
| 门控层 | 三道检查决定是否上线，失败回滚 | 否 |
| 记录层 | 审计日志 + SwanLab | 否 |

### 3.2 技术选型

| 用途 | 选择 | 理由 |
|---|---|---|
| 工业模型 | LightGBM | 4 核 8G 无 GPU，CPU 上快；评测环境禁用 embedding，深度模型出局 |
| 概率校准 | sklearn `CalibratedClassifierCV` | "重新校准"是成本最低的更新动作 |
| 漂移检测 | Frouros | 概念漂移与数据漂移齐全，BSD 协议，纯 Python |
| 分布距离 | 自实现 PSI + scipy KS | PSI 无现成轻量实现，自己写几十行更可控 |
| 多变量漂移 | sklearn PCA 重构误差 | 不引入额外依赖 |
| 实验记录 | SwanLab | 官方指定 |
| 配置 | YAML | 任务适配靠改配置而非改代码 |

当前环境：Python 3.11.9，已有 numpy 2.4.6、pandas 3.0.3、sklearn 1.9.0、scipy、matplotlib、yaml。**需补装 lightgbm、frouros、swanlab**。

### 3.3 漂移类型判定表（方法设计的核心依据）

| 分布变化 | 性能变化 | 判定 | 建议动作 |
|---|---|---|---|
| 无 | 无 | 稳定 | 不动 |
| 有 | 无 | 协变量漂移，模型仍适用 | 不动（只记录） |
| 有 | 下降 | 协变量漂移致失准 | 滑动窗口重训 / 增量训练 |
| 无 | 下降 | 概念漂移 | 重训，降低旧数据权重 |
| 类别占比变化 | 下降 | 标签漂移 | 调阈值 / 重校准（最便宜） |
| 任意 | 剧烈下降且重训无效 | 不可信 | 下线/拒判（待规则确认） |

## 4. 影响文件

全部为新建。根目录 `/Users/zhangtianshuo/Desktop/harness/`

```
requirements.txt                     依赖清单
configs/a1_gas.yaml                  A1 任务配置
data/raw/                            原始数据（batch1.dat ... batch10.dat）
src/harness/
  config.py                          配置加载与校验
  datasets/base.py                   TaskAdapter 协议：load_batches/metric/task_type
  datasets/gas_sensor.py             A1 适配器，解析 libsvm 风格的 .dat
  stream/batch_stream.py             BatchStream：串行卡点，必须 submit 才能 next
  stream/simulator.py                漂移注入：协变量/标签/概念，突变与渐变
  models/industrial.py               LightGBM 封装：fit/predict/calibrate/save/load
  models/registry.py                 版本库：online/candidate/history，上线与回滚
  monitor/performance.py             真实性能指标计算
  monitor/distribution.py            PSI、KS、PCA 重构误差
  monitor/error_stream.py            Frouros ADWIN/DDM 跑在误差序列上
  monitor/report.py                  DriftReport 数据类 + 漂移类型判定
  decide/actions.py                  动作枚举与执行器
  decide/rules.py                    规则决策器（实现 Decider 协议）
  gate/replay.py                     历史工况回放集维护
  gate/evaluator.py                  三道门控检查
  record/schema.py                   DecisionRecord 结构定义
  record/audit.py                    JSONL 审计日志
  record/swan.py                     SwanLab 封装（可离线降级）
  loop.py                            主闭环编排
scripts/download_data.py             下载并校验 A1 数据
scripts/run_baseline.py              E0 静态基线
scripts/run_loop.py                  E2 规则闭环
tests/                               单元测试
outputs/                             曲线图与结果表
```

## 5. 关键实现细节

### 5.1 串行卡点（硬约束的本地还原）

B 包要求"完成当前批次输出才能拿下一批"。本地流必须一样严格，否则会写出偷看未来数据的代码而不自知。

```python
class BatchStream:
    def __init__(self, batches): self._b, self._i, self._done = batches, 0, True
    def next_batch(self):
        if not self._done: raise RuntimeError("上一批未提交结果")
        if self._i >= len(self._b): return None
        self._done = False
        b = self._b[self._i]
        return BatchView(index=self._i, X=b.X)      # 只给 X，不给 y
    def submit(self, pred_or_offline):
        if self._done: raise RuntimeError("没有待提交的批次")
        y = self._b[self._i].y
        self._done, self._i = True, self._i + 1
        return y                                     # 提交后才返回标签
```

`next_batch` 只返回 X。标签在 `submit` 之后才给。这样"预测时用到标签"这类错误会直接抛异常。

### 5.2 决策器接口（为大模型预留）

规则决策器和后续的大模型决策器实现同一个协议，主闭环不需要改动：

```python
class Decider(Protocol):
    def decide(self, report: DriftReport, history: list[DecisionRecord]) -> Decision: ...

@dataclass
class Decision:
    action: Action              # NOOP/RECALIBRATE/ADJUST_THRESHOLD/INCREMENTAL/
                               # WINDOW_RETRAIN/FULL_RETRAIN/FEATURE_SUBSET/OFFLINE
    params: dict                # 如 {"window": 3, "decay": 0.9}
    reason: str                 # 依据，进审计日志
    hypothesis: str | None      # 规则版为 None；大模型版填假设
```

第二阶段只需新增 `decide/llm.py`，并在配置里切换 decider 类型。

### 5.3 三道门控

```python
def gate(cand, online, baseline, recent, replay, cfg) -> GateResult:
    s_c, s_o = score(cand, recent), score(online, recent)
    if s_c - s_o < cfg.min_gain:          return reject("提升不足阈值")
    if s_c < score(baseline, recent):     return reject("低于静态基线")
    if score(cand, replay) < score(online, replay) - cfg.forget_tol:
        return reject("回放集退化，疑似遗忘")
    return accept()
```

`min_gain` 防止噪声导致的无意义切换；回放集检查对应"稳定性与抗遗忘"5 分。

### 5.4 回放集

每个历史批次抽取少量代表性样本（分层抽样，保证类别覆盖），总量设上限，超出时按时间衰减概率淘汰。候选模型必须在回放集上不显著退化。

### 5.5 审计记录结构

同时满足门槛"至少 3 次读历史→提假设→做实验→做决策"闭环：

```python
@dataclass
class DecisionRecord:
    batch_index: int
    trigger: str                  # 触发原因
    evidence: dict                # 性能、PSI、KS、ADWIN 信号
    drift_type: str               # 判定结果
    history_read: list[int]       # 读了哪些历史记录（闭环证据）
    hypothesis: str | None        # 假设（闭环证据）
    experiments: list[dict]       # 做了哪些实验及结果（闭环证据）
    decision: Decision            # 最终决策（闭环证据）
    gate_result: dict
    outcome: dict                 # 下一批的实际表现，事后回填
    cost: dict                    # 训练次数、耗时、Token
```

`outcome` 在后续批次回填，用于复盘与策略改进。字段设计留扩展位，等 11-09 官方《实验记录规范》出来后对齐。

## 6. 边界条件与异常处理

| 情况 | 处理 |
|---|---|
| 某批次缺少部分类别（A1 批次 3/4/5 无 Toluene） | Macro-F1 只在本批出现的类别上计算，并记录缺类情况；训练时固定全局类别集合，避免模型输出维度跳变 |
| 批次样本量过小 | 低于阈值时跳过更新决策，只累积数据，避免小样本噪声触发误更新 |
| 训练失败或超时 | 丢弃候选，保留在线模型，记一次失败事件，不影响本批输出 |
| 连续多批门控失败 | 回滚到历史最优版本 |
| 回放集为空（第一批） | 跳过回放检查，只做前两道门控 |
| SwanLab 不可用 | 降级为本地离线记录，不中断闭环 |
| 首批无在线模型 | 冷启动：用首批训练初始模型，该批预测用全局先验或多数类 |
| 特征含 NaN 或量纲异常 | 适配层统一做缺失填充与标准化，scaler 只用已见数据拟合 |

**绝对不能做的事**：用未来批次的数据拟合任何东西（包括 scaler 和 PCA）。串行卡点在接口层强制这一点。

## 7. 数据流路径

```
data/raw/batchN.dat
    │ gas_sensor.py 解析（libsvm 风格：label feat:val ...）
    ▼
Batch(X, y, index)  ──→ simulator.py（可选，注入人工漂移）
    ▼
BatchStream（串行卡点）
    │ next_batch() 只给 X
    ▼
registry.online.predict(X) ──→ stream.submit(pred) ──→ 返回 y
    ▼
monitor（performance + distribution + error_stream）→ DriftReport
    ▼
decide（规则表）→ Decision
    ▼
actions 执行 → candidate 模型
    ▼
gate（recent / baseline / replay）
    ├─ accept → registry.promote()
    └─ reject → registry.discard() / rollback()
    ▼
record（audit.jsonl + SwanLab）→ 供下一批读取历史
```

## 8. 预期结果

**量化目标：**

1. E0 静态基线：画出 A1 批次 2 到 10 的性能曲线，确认漂移导致的下降幅度（文献上该数据集跨批次准确率下降可达二三十个百分点）
2. E2 规则闭环：综合性能显著优于 E0，且在回放集上不退化
3. 10 个批次全程无人工干预跑完，中途人为注入训练失败也能跑完
4. 产出至少 3 条包含完整闭环字段的审计记录
5. 单批次端到端耗时在 8 核机器上控制在秒级，为评测环境的 4 核预留余量

**交付物：**

- `outputs/e0_baseline.csv` 与 `outputs/e2_rule_loop.csv`：逐批次指标
- `outputs/curve_e0_e2.png`：对照曲线
- `outputs/audit.jsonl`：审计记录
- SwanLab 项目链接
- 一份简短的结论说明：漂移有多严重、规则闭环救回了多少

**本阶段验收标准**：`python scripts/run_baseline.py` 和 `python scripts/run_loop.py` 两条命令可重复执行并产出上述文件，结果在固定随机种子下可复现。



