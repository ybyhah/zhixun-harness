# 自主闭环决策质量对照评测

- [x] Task 1: 搭建 `scripts/compare_deciders.py` 骨架
    - 1.1: CLI 参数 `--packages/--arms/--rounds/--policy/--candidates`
    - 1.2: `load_package` 与每条臂独立输出目录 `outputs/compare/{pkg}/{arm}/`
    - 1.3: 计算静态基线 `baseline_value` 与 `base_score=to_score`

- [x] Task 2: 实现三条决策臂
    - 2.1: notrain 参照臂（scores 恒为 base_score，资源 0）
    - 2.2: rule 臂（`RulePolicy`/`AdaptivePolicy` + `OfficialDriver.run`）
    - 2.3: llm 臂（`LLMDecider`+`LLMClient`，沿用 .env 的 27B）

- [x] Task 3: 指标汇总
    - 3.1: 逐轮值→`to_score`→`metrics.summarize(baseline_composite=base_score,...)`
    - 3.2: 从各臂 `audit.jsonl` 累加 `cost.tokens` 与降级轮数
    - 3.3: 收集显著上线、训练次数、耗时、交付值

- [x] Task 4: 生成对照报告
    - 4.1: 写 `outputs/COMPARISON_deciders.md`（三臂对照表 + 简短结论）
    - 4.2: 写 `outputs/compare_deciders.csv`（同列，便于复核）

- [x] Task 5: 运行与验证
    - 5.1: 先用 notrain+rule 快跑（小 rounds）验证表格/CSV 正确
    - 5.2: 接入 llm(27B) 臂后台跑一遍，确认不降级、增益列有值
    - 5.3: `pytest -q` 确认未影响既有用例
