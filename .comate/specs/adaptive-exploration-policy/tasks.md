# 自适应探索策略任务计划

> 对应 `.comate/specs/adaptive-exploration-policy/doc.md`。
> 原则：默认 `--policy rule` 行为与现状完全一致；新策略不得降低已有最优指标；
> NOOP 轮照常写审计（闭环数仍 ≥ 3）；判定按"指标不退且训练次数下降"才采纳。

- [x] Task 1: RoundPlan 支持跳过
    - 1.1: `policy.py` 的 `RoundPlan` 增加 `skip: bool = False` 与 `reason: str = ""`
    - 1.2: `expand_candidates` 对 `skip=True` 的 plan 直接原样返回（不扩候选）
    - 1.3: 测试：skip 字段默认 False，不影响既有调用

- [x] Task 2: AdaptivePolicy
    - 2.1: 新增 `AdaptivePolicy`（实现同一 `Decider` 协议），内部记录已试过的 `(parent, epochs, lr)` 与最近验证值
    - 2.2: 停滞检测：上一轮验证值与前一轮差 < tol（默认 1e-9）→ 判定 lr 进入无效区
    - 2.3: 换轴：停滞时 epochs ×2（上限 50）、lr 回弹到上次有效值，而不是继续减半
    - 2.4: 去重：`(parent, epochs, lr)` 已试过 → 返回 `skip=True` 的 NOOP plan
    - 2.5: 常规路径：有提升则保持；被拒但有变化则按原逻辑减半 lr
    - 2.6: 兜底：轴耗尽则 NOOP；但第 1 轮永不跳过
    - 2.7: 保证返回的超参始终在官方合法区间内

- [x] Task 3: 闭环支持 NOOP 轮
    - 3.1: `loop.py` 在 `plan.skip` 为真时不训练、不计 `trainings`、不新建轮次目录
    - 3.2: 该轮仍写审计（动作记 NOOP、含跳过原因、四要素齐全）
    - 3.3: SwanLab 记录该轮为跳过（不污染指标曲线）
    - 3.4: 历史里记录该轮为"未训练"，供策略后续判断

- [x] Task 4: 入口与回归
    - 4.1: `run_official.py` 加 `--policy {rule,adaptive}`，默认 `rule`
    - 4.2: 默认路径复跑 A，确认与改动前逐轮结果完全一致（0.7140→0.7264）
    - 4.3: 新增 `tests/test_policy.py`：停滞换轴、重复判 NOOP、有进展保持、轴耗尽 NOOP、超参合法、首轮不跳过
    - 4.4: 运行完整 `python3 -m pytest` 全绿

- [x] Task 5: 对照实验
    - 5.1: A/B/C 各跑 `--policy rule` 与 `--policy adaptive`，固定种子
    - 5.2: 记录最优指标、实际训练次数、耗时、单位增益训练次数、完整闭环数
    - 5.3: 各复跑一次确认可复现

- [x] Task 6: 判定与收尾
    - 6.1: 判定（指标不退 且 训练次数下降 → 采纳为推荐默认；指标退则不采纳）
    - 6.2: 更新 `outputs/CONCLUSIONS.md`：资源效率对照表与采纳结论
    - 6.3: 如实记录；若某包无浪费可省（B/C 本就零停滞）照实写明
    - 6.4: 明确下一步转向专家分与提交硬门槛（Docker、技术报告、演示视频、来源声明）
