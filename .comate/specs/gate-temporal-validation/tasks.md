# 门控时序验证任务计划

> 对应 `.comate/specs/gate-temporal-validation/doc.md`。
> 原则：默认行为（`in_batch`）必须与改动前完全一致；新增 `holdout_recent` 用 A/B + bench 量化后再决定去留。每步末尾跑 A1/A3 回归验证。

- [x] Task 1: split_last 增加时序验证口径
    - 1.1: `TrainingData.split_last` 增加 `val_mode` 参数（默认 `in_batch`，保持现状）
    - 1.2: 新增 `holdout_recent` 分支：训练池 = `X[:-1]`，验证集 = 整个 `X[-1]`
    - 1.3: 只有 1 个批次时 `holdout_recent` 回退 `in_batch`；未知取值抛 `ValueError`
    - 1.4: `tests/test_decide.py` 补两种模式的单元测试：验证集不与训练池重叠、holdout 下训练池不含最新批、单批回退

- [x] Task 2: 闭环与配置接入
    - 2.1: `loop.py` 的 `_after_submit` 调 `split_last` 时传 `self.cfg.gate.get("val_mode", "in_batch")`
    - 2.2: 三个配置 `gate` 段显式加 `val_mode: in_batch`
    - 2.3: `run_loop.py` 的 `VARIANTS` 增加 `holdout-gate`（`gate.val_mode=holdout_recent`）
    - 2.4: 跑 A1/A3 默认配置，确认逐批分数与改动前完全一致

- [x] Task 3: bench 增加门控口径对比
    - 3.1: `bench_external.py` 的 `run_estimate` 增加 `holdout_recent` 口径的高估量，与 `in_batch` 并列
    - 3.2: A1/A2 输出两种口径的平均高估，确认 holdout 的高估明显更低
    - 3.3: 复跑 `tests/test_bench_external.py` 相关纯函数测试

- [x] Task 4: 端到端 A/B 实验
    - 4.1: A1/A2/A3 分别跑 `--variant full` 与 `--variant holdout-gate`，记录综合性能、最差阶段、恢复、上线/训练次数
    - 4.2: 固定种子各跑两次确认可复现
    - 4.3: 运行完整 `python3 -m pytest` 确认全绿

- [x] Task 5: 结论与去留决策
    - 5.1: 汇总 A/B 结果，判断 `holdout_recent` 是否改善上线决策
    - 5.2: 更新 `outputs/COMPARISON.md` 第 3 节：补 holdout 口径的高估下降与端到端影响
    - 5.3: 按结果决定默认 `val_mode`（改善则采纳，否则保留 `in_batch` 并把"提升后全窗口重训"列入下一步）
    - 5.4: 如实记录收益不大的情形与原因
