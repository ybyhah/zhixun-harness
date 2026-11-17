# 自主闭环决策质量对照评测（decider-comparison-eval）

## 需求场景与处理逻辑
赛题内核是「自学习智能体」，客观分主体是「训 vs 不训的平均增益」与「最差阶段/恢复速度」。要用**同等条件对照实验**证明自主闭环确实有效，而不是靠外层网格搜索刷单一指标。

在每个官方包（A/B/C）的逐轮续训流上，横向对比三条「决策臂」：
1. **不训（静态基线 E0）**：模型永不更新，逐轮值恒为 `baseline_value`，增益恒为 0，作参照零点。
2. **规则快通路**：`RulePolicy`（可选 `AdaptivePolicy`）。
3. **27B 慢通路**：`LLMDecider + LLMClient`（读 history→提假设→超参）。

对每条臂统计：平均增益(gain_vs_baseline)、综合(composite)、最差阶段(worst_stage)、恢复(recovery)、显著上线次数、训练次数、耗时、token，交付值。输出一张对照表 + 结论。

## 架构与技术方案
- 不改核心：直接**复用** `OfficialDriver`（driver/loop.py）跑规则/LLM 臂，复用 `src/harness/metrics.summarize`（已含 composite/worst_stage/recovery/gain_vs_baseline/资源效率）。
- 新增一个编排脚本 `scripts/compare_deciders.py`：按包×臂依次用 `OfficialDriver` 跑，收集逐轮值与资源，调 `metrics.summarize(scores, baseline_composite=to_score(pkg,baseline), ...)` 算增益/恢复，汇总成 Markdown + CSV。
- 不训臂不需要真跑：其 scores 恒为 baseline，增益 0、恢复无跌落，作为参照行直接写入。
- LLM 臂沿用已修好的 `.env`（27B + certifi + enable_thinking）。27B 冷启动约 158s/轮，故 `--rounds` 默认小（5），`--packages`/`--arms` 可选，长跑放后台。

## 影响文件
- 新增 `/Users/zhangtianshuo/Desktop/harness/scripts/compare_deciders.py`（唯一新增代码；编排+汇总+出报告）。
- 新增产物 `/Users/zhangtianshuo/Desktop/harness/outputs/COMPARISON_deciders.md`、`outputs/compare_deciders.csv`。
- 复用（不改）：`src/driver/loop.py`(OfficialDriver)、`src/driver/metric.py`(to_score/round_value)、`src/driver/policy.py`、`src/driver/llm.py`、`src/harness/metrics.py`(summarize)、`src/driver/package.py`(load_package)。
- 每条臂用独立 `rounds_dir`/`audit_path`（`outputs/compare/{pkg}/{arm}/`），避免覆盖官方正式产物。

## 实现细节
核心编排（按包×臂）：
```python
pkg = load_package(ROOT / f"base/{name}")
base = pkg_baseline_value(pkg)                 # = OfficialDriver 内 _baseline_value
base_score = to_score(pkg, base)
for arm in arms:                               # notrain / rule / llm
    if arm == "notrain":
        scores, sec, tok, prom, tr = [base_score]*rounds, 0.0, 0, 0, 0
    else:
        policy = RulePolicy(...) if arm=="rule" else LLMDecider(..., client=LLMClient())
        drv = OfficialDriver(pkg, rounds_dir=out/arm, rounds=rounds, policy=policy,
                             gate_mode="significance", audit_path=out/arm/"audit.jsonl")
        res = drv.run()
        scores = [to_score(pkg, r.value) for r in res.rows]
        sec, prom, tr = res.seconds, res.significant_promotions, res.trainings
        tok = sum_tokens(out/arm/"audit.jsonl")      # 累加 cost.tokens
    s = metrics.summarize(scores, baseline_composite=base_score,
                          trainings=tr, seconds=sec, promotions=prom)
    rows.append((name, arm, s.gain_vs_baseline, s.composite, s.worst_stage,
                 s.recovery_mean, prom, tr, sec, tok, res_delivered))
```
报告列：`包 | 臂 | 平均增益 | 综合 | 最差阶段 | 恢复 | 显著上线 | 训练次 | 耗时s | tokens | 交付值`。

## 边界与异常
- LLM 臂某轮超时/降级：`LLMDecider` 内部回退规则且不抛错；该轮照常计入（`llm_degraded` 可从 audit 读，报告标注降级轮数）。
- 无 PyTRiO 端点：LLM 臂整列退化为规则，脚本不报错（与离线语义一致）。
- 不训臂 scores 恒为 baseline → 增益 0、`recovery` 无跌落、资源 0，作参照行。
- CLI：`--packages a,b,c`、`--arms notrain,rule,llm`、`--rounds N`（默认 5）、`--policy rule|adaptive`、`--candidates N`。

## 数据流
`compare_deciders.py` → 逐（包,臂）跑 `OfficialDriver.run()` → `res.rows`(逐轮值)+`audit.jsonl`(tokens/降级) → `to_score` → `metrics.summarize(baseline_composite=...)` → 行汇总 → 写 `COMPARISON_deciders.md` + `compare_deciders.csv`。

## 预期结果
- 一张三臂对照表：量化「27B 慢通路 / 规则 / 不训」在平均增益、最差阶段、恢复、资源上的差异，直接作为「同等条件对照实验」证据（对应加分项与实验分析分）。
- 不改任何核心逻辑与官方产物；`pytest` 不受影响。
- 结论能回答「自主闭环是否真的优于不训与规则」，并指出下一步该强化闭环的哪一环。

## 已知范围 / 后续
- 本轮在**官方包逐轮续训流**上对比（评分对象一致）；每轮是续训而非严格不同漂移批次，`recovery/worst_stage` 为代理指标。
- 真·漂移流（A1/A2/A3 配置、`bench_external.py` 的 batch stream）上的三臂对比，因 LLM 决策器目前只在 driver 路径，需额外桥接，列为后续。

