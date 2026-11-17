# 统计显著性门控任务计划

> 对应 `.comate/specs/significance-gate/doc.md`。
> 工业级红线：①指标口径必须与官方 `metrics()` 逐位一致（单测强制）；②双轨保证结果不劣于现状；
> ③bootstrap 异常一律降级、绝不抛错（C1 无人干预）；④确定性可复现；⑤默认 `--gate plain` 行为不变。

- [x] Task 1: 官方口径指标的纯函数版（最先做，后面全依赖它）
    - 1.1: 新增 `src/driver/official_metric.py`：`classification_macro_f1(pred, y, idx)` 严格按官方公式——
      在 `set(expected) ∪ set(argmax(pred))` 上宏平均（**不是只看真实出现的类**）
    - 1.2: `regression_rmse(pred, y, idx)`；统一 `score(pred, y, idx, task)` 返回"越大越好"
    - 1.3: 单测（阻塞性）：对真实 checkpoint + 全量验证集，与官方 `baselines.train.metrics()` **逐位相等**（atol=0），A 与 C 各一例
    - 1.4: 单测：空/单类/含 NaN 输入不抛错

- [x] Task 2: 配对 bootstrap
    - 2.1: 新增 `src/driver/significance.py` 的 `paired_bootstrap(pred_a, pred_b, y, task, B, seed)`
      → `(p_better, ci_low, ci_high, mean_diff)`；同一组 idx 同时用于两个模型
    - 2.2: RNG 由 `(seed, round, cand)` 确定性派生；同种子两次结果完全一致
    - 2.3: 任何异常/退化输入返回 `None`（上层降级），绝不抛错
    - 2.4: 单测：同模型比自己 → p=0、差值恒 0；确定性；B 可配；降级路径

- [x] Task 3: 门控接入
    - 3.1: `gate.py` 的 `OfficialGate` 增加可选显著性检验（需要候选与当前最优的预测）
    - 3.2: `GateDecision` 增加 `p_better` / `ci_low` / `ci_high`，纳入 `to_dict`
    - 3.3: "不低于官方基线"仍用普通数值比较（这是门槛不是比较）
    - 3.4: 显著性不可用（返回 None）时退回普通比较，并在 reason 标注降级
    - 3.5: 平局（p 落在 [1−阈值, 阈值]）时偏向 `final_steps` 更小的候选

- [x] Task 4: 双轨选择与外部裁决
    - 4.1: `loop.py` 维护 `best_plain`（现状）与 `best_significant` 两条链
    - 4.2: 续训起点仍用 `best_plain`（不因保守而停滞探索）
    - 4.3: 最终交付优先 `best_significant`，不存在则回退 `best_plain`（不劣性保证）
    - 4.4: 支持 `--verdicts verdicts.json` 外部裁决：被拒轮次不更新 best 链（承接官方"平台未采纳则从仍被选中的模型继续"）
    - 4.5: 审计与 SwanLab 记录 p_better、CI、B、种子、双轨当前值

- [x] Task 5: 入口与回归
    - 5.1: `run_official.py` 加 `--gate {plain,significance}`（默认 plain）、`--bootstrap`、`--p-threshold`、`--verdicts`
    - 5.2: 默认模式复跑 A/B/C，确认与改动前逐轮结果完全一致
    - 5.3: 运行完整 `python3 -m pytest` 全绿

- [x] Task 6: 端到端 A/B 与耗时
    - 6.1: 三包各跑 `--gate plain` 与 `--gate significance`，记录最终交付 checkpoint、验证指标、上线次数、每轮 p_better
    - 6.2: 实测 bootstrap 耗时（C 验证集 23049 行 × B=500），超预算则降 B 并记录
    - 6.3: 固定种子复跑确认一致

- [x] Task 7: 判定与收尾
    - 7.1: 核对不劣性：显著性模式交付物不差于 plain；若 plain 验证分更高，给出其 p 值证明它是噪声峰
    - 7.2: 更新 `outputs/CONCLUSIONS.md`：配对 bootstrap 证据、双轨设计、A/B 结果与耗时
    - 7.3: 按结果定推荐默认值；如实记录"以可信度换少量验证分"的取舍
