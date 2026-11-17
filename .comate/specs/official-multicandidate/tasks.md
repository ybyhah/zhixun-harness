# 官方续训多候选搜索任务计划

> 对应 `.comate/specs/official-multicandidate/doc.md`。
> 原则：默认 `--candidates 1` 与现状完全一致；不改官方 `baselines/`；多候选只多次调用官方 `train.py`；候选数有上限，结果诚实对照单候选。

- [x] Task 1: 候选扩展（纯函数）
    - 1.1: `policy.py` 新增 `expand_candidates(base_plan, k, round_index)`：以 base 为中心，lr ∈ {base, base×0.5, base×2}，必要时 epochs ∈ {base, base-2}
    - 1.2: 全部裁剪到官方合法区间（epochs 1..50，lr 0<lr≤0.1），去重，最多 k 个；k=1 只返回 base
    - 1.3: base lr 为 None 时只扰动 epochs，不制造非法 lr
    - 1.4: `tests/test_driver.py` 补：数量上限、去重、裁剪、k=1 等于 base、lr=None 处理

- [x] Task 2: 闭环支持每轮多候选
    - 2.1: `loop.py` 增加 `candidates` 参数（默认 1）；k>1 时本轮从当前被采纳 checkpoint 出发，对每个候选各训一次，输出 `round-k/cand-j`
    - 2.2: 取验证最优候选，过门控（不低于官方基线、较当前最优有提升）才上线；否则不上线
    - 2.3: 单个候选失败/超时则跳过，全失败本轮不上线；训练次数计入全部候选
    - 2.4: 审计记录本轮全部候选及验证分、选中项；SwanLab 记录选中值与候选数
    - 2.5: `tests/test_driver.py` 补多候选择优、部分失败、全部不如当前最优的 mock 测试

- [x] Task 3: 入口与回归
    - 3.1: `run_official.py` 加 `--candidates K`（默认 1）
    - 3.2: K=1 复跑 A，确认与改动前逐轮结果完全一致（A 0.7140→0.7264）
    - 3.3: 运行完整 `python3 -m pytest` 全绿

- [x] Task 4: A/B/C 对照实验
    - 4.1: 三包各跑 `--candidates 1` 与 `--candidates 3`，记录 best 指标、上线/训练次数、耗时
    - 4.2: 固定种子各复跑一次确认可复现
    - 4.3: 判断多候选是否带来提升；若无额外提升如实记录并说明原因

- [x] Task 5: 结论与去留
    - 5.1: 更新 `outputs/CONCLUSIONS.md`：单候选 vs 多候选对照与资源代价
    - 5.2: 按结果决定 `--candidates` 推荐默认值；收益不显著则保留 1 并写明
    - 5.3: 标注后续（LLM 一次提多候选、改 baselines 代码）入下一轮
