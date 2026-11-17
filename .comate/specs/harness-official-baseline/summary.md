# Harness 驱动官方 base/ 续训完成总结

## 结果：三任务全部对齐官方口径并超过官方基线

Harness 重构为驱动官方 `base/` 续训的编排器，入口 `scripts/run_official.py`。每包 10 轮续训、门控上线、固定种子可复现、0 降级、每包 9 条完整闭环（≥3 门槛）。

| 任务 | 指标 | 官方基线 | 10 轮后最优 | 变化 | 上线/续训 | 用时 |
|---|---|---|---|---|---|---|
| A 气体识别 | Macro-F1↑ | 0.7140 | **0.7264** | +0.0124 | 2/10 | 5s |
| B 轴承 RUL（分钟） | RMSE↓ | 681.16 | **666.88** | −14.28 | 1/10 | 11s |
| C 发动机 RUL（循环） | RMSE↓ | 52.87 | **51.38** | −1.49 | 6/10 | 78s |

三者均不低于官方基线（C10 满足）。

## 新增模块（`src/driver/`，与旧 `src/harness/` 并存）

| 文件 | 职责 |
|---|---|
| `package.py` | 定位 base 包、校验 checkpoint sha256、给出 profile/task/数据路径 |
| `trainer.py` | 子进程调用官方 `baselines.train`，首轮 `--from-baseline`、续轮 `--resume`，B 自动带 `--label-provenance`，解析 `training.json` |
| `metric.py` | 官方 `validation_after` → 统一越大越好分数（回归 -RMSE），复用 `harness.metrics` |
| `gate.py` | 不低于官方基线且较当前最优有提升才上线 |
| `policy.py` | 规则策略（被拒减半 lr 再探）；预留 `Decider` 给 LLM 慢通路 |
| `loop.py` | 10 轮编排 + 审计（复用 `harness.record` 四要素）+ SwanLab；异常降级不中断 |
| `scripts/run_official.py` | 入口 `--package --rounds [--swanlab]` |
| `tests/test_driver.py` | 12 项：包定位、指标方向、门控、策略、trainer 命令拼装（mock 子进程） |

旧 `src/harness/**`、`run_loop.py` 原样保留作对照基线。全部 231 个测试通过。

## 严格遵守的官方续训约束

- 输出必须新目录；`--from-baseline`/`--resume` 保权重+Adam+归一化+种子，不重新初始化。
- 不改 hidden/特征schema/task/profile/seed（官方 `train.py` 会拒绝）。
- B 必带 label-provenance（status=draft，method 写明 RUL=(N−i) 分钟确定性计算）。
- 归一化/目标缩放只在初始训练集拟合，续训沿用；验证集不参与。
- 门控用官方验证指标；平台独立跑 `predict.py` 于隐藏测试，自报指标不替代官方评测。

## 与 doc/tasks 的偏差

- 未建 `configs/official_{a,b,c}.yaml`：脚本以 `--package` 直接驱动，避免无人读取的死配置（YAGNI）。
- `audit.py` 未单独成文件：直接在 `loop.py` 复用 `harness.record` 的 DecisionRecord/AuditLog/SwanRecorder，省掉一层薄封装。

## 重要纠偏（已写入 CONCLUSIONS 顶部）

- **B 是 RUL 回归（分钟），不是三分类**。此前 A2 的 Health/Degrading/Severe 三分类与官方 B 不符，相关"Severe 召回/关键类门控"结论对官方 B 不适用，仅作方法探索保留。
- 工业模型是官方 NumPy MLP（A 4326 / B 865 / C 833 参），SwanLab 记录对象正是它的每轮续训。

## 下一步（分值与契合度排序）

1. **LLM 慢通路**：PyTRiO Qwen 读 `history.jsonl/training.json` → 提下一轮超参或改 `baselines/` 代码 → 跑实验 → 决策。`policy.Decider` 接口已留。对应方法设计 30 分 + 策略层自改进。
2. **模型层优化**：在官方续训约束内调 lr/epochs/batch 与（合规的）代码改动，做对照实验找更优续训轨迹。
3. **SwanLab 公开 + Docker 打包**：复用上一个 spec 成果，打包对象改为 base/ + driver；SwanLab 记录官方 MLP 续训并设公开。
