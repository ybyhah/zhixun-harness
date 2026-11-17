# 统计显著性门控（工业级设计）

> 证据：配对 bootstrap 显示我们逐轮的"上线"决策有一部分落在噪声里——A 的 round-7 相对 round-1
> 只差 **+0.0006**，而配对噪声标准差是 **0.0086**；C 也有 +0.0075 对 0.057 的类似情形。
> 平台在**隐藏测试**打分，我们在**公开验证集**挑模型；追验证集的噪声峰会同时伤害
> 平均性能提升（15）、最差阶段与恢复速度（10）、稳定性（5）。

## 0. 先把工业级的顾虑想清楚（设计约束）

| 顾虑 | 约束/对策 |
|---|---|
| **指标口径必须与官方一致** | 官方 `metrics()` 的 Macro-F1 在 `set(expected) ∪ set(predicted)` 上做宏平均（**不是只看真实出现的类**）。我们的 bootstrap 必须用同一公式，否则门控优化的东西和平台打分的东西不是一回事。**强制措施**：单测断言"我们的实现在全量验证集上与官方 `metrics()` 逐位相等"。 |
| **绝不能让结果变差** | 采用**双轨**：同时维护"验证最优"与"显著性确认"两条链。最终产物优先取显著性确认的；没有则回退验证最优。**数学上保证不劣于现状。** |
| **无人干预不能崩**（C1） | bootstrap 任何异常（退化样本、NaN、单类）都降级为普通数值比较，绝不抛错；门控永远返回决策。 |
| **可复现** | bootstrap RNG 由 `(cfg.seed, 轮次, 候选序号)` 确定性派生，不用全局随机。 |
| **时间预算** | 只在预测结果上重采样，不重训、不重推理。C 的验证集 23049 行 × B=500 需实测耗时并写入报告；超预算则自动降 B。 |
| **过度保守风险** | 若全程无显著提升 → 双轨回退到验证最优，等价于现状；不会出现"一次都不上线"导致零增益。 |
| **默认行为不变** | 新门控为 opt-in（`--gate significance`），默认 `plain` 保持现状，A/B 验证后再定推荐值。 |
| **可审计** | 每次决策记录 P(更好)、置信区间、B、种子，写入审计与 SwanLab。 |

## 1. 方案

### 1.1 配对 bootstrap 检验（核心）

比较候选与当前最优时，**在同一批验证样本上**重采样：

```
预测一次：pred_cand, pred_best（官方 Model.predict，不重复推理）
for b in 1..B:
    idx = rng.integers(0, n, n)                  # 同一组 idx 同时用于两个模型
    d_b = official_metric(pred_cand, y, idx) − official_metric(pred_best, y, idx)
    （回归取负，统一成"正=更好"）
P(更好) = mean(d_b > 0)；CI = d 的 2.5/97.5 分位
通过条件：P(更好) ≥ p_threshold（默认 0.95）
```

配对是关键：两个模型共享样本，差值的方差远小于各自估计的方差（实测 A：单点 ±0.0249 vs 配对 ±0.0086）。

### 1.2 双轨选择（不劣于现状的保证）

```
best_plain        ：验证指标最优的 checkpoint（= 现在的行为）
best_significant  ：最近一个"显著优于其上一个显著模型"的 checkpoint
最终提交          ：best_significant 若存在，否则 best_plain
续训起点          ：仍从 best_plain 继续（保持探索能力，不因保守而停滞）
```

这样显著性只决定**最终交哪个**与**如何记录**，不削弱搜索过程。

### 1.3 平台侧裁决（承接官方要求）

官方 `TRAINING_GUIDE` 第 28 行："**若平台没有采纳某轮候选，应从仍被选中的模型继续**"。
说明评测是分阶段 + 平台侧 accept/reject。驱动器增加可选的**外部裁决**输入
（`--verdicts verdicts.json`，形如 `{"round-3": "rejected"}`）：被外部拒绝的轮次不更新 best 链，
下一轮仍从仍被选中的模型继续。本地无裁决文件时按自身门控运行（现状）。

### 1.4 平局时偏向更少步数

显著性无法区分（P 在 [1−p, p] 之间）时，选 `final_steps` 更小的那个——更少的续训步数意味着更低的过拟合风险。零成本 tie-break。

## 2. 影响文件

| 文件 | 改动 |
|---|---|
| `src/driver/official_metric.py` | 新增：与官方 `metrics()` 完全一致的纯函数版（给定 pred/y/idx 算指标），**单测对齐官方** |
| `src/driver/significance.py` | 新增：`paired_bootstrap(pred_a, pred_b, y, task, B, seed)` → `(p_better, ci_low, ci_high, mean)`；任何异常返回 `None` 交由上层降级 |
| `src/driver/gate.py` | `OfficialGate` 增加可选显著性检验；`GateDecision` 增加 `p_better`/`ci` 字段 |
| `src/driver/loop.py` | 维护双轨（`best_plain` / `best_significant`）；支持外部裁决；审计与 SwanLab 记录显著性 |
| `scripts/run_official.py` | `--gate {plain,significance}`（默认 plain）、`--bootstrap B`、`--p-threshold`、`--verdicts` |
| `tests/test_significance.py` | 官方口径对齐、配对性、确定性、退化输入降级、P 单调性 |
| `tests/test_driver.py` | 双轨选择、外部裁决、平局偏少步数 |

## 3. 验证

1. **口径对齐**（最重要）：随机 checkpoint + 全量验证集，我们的指标函数输出与官方 `metrics()` **逐位相等**。
2. **配对性**：同一模型比自己 → P(更好)=0、CI 包含 0、差值恒为 0。
3. **确定性**：同种子两次 bootstrap 结果完全一致。
4. **降级**：构造 NaN/单类/空样本 → 返回 None 且门控不抛错。
5. **端到端 A/B**：三包各跑 `--gate plain` 与 `--gate significance`，对比最终交付的 checkpoint、
   验证指标、上线次数、每轮 P(更好)、bootstrap 耗时。
6. **不劣性**：显著性模式下最终指标 ≥ plain 模式下的"保守等价物"；若 plain 更高则必须是因为
   plain 选了噪声峰（报告里给出 P 值作为依据）。

## 4. 预期与如实记录

- 预期：上线次数减少，最终交付的 checkpoint 可能验证分略低（A 可能 0.7264 → 0.7258），
  但附带"P(更好)=100%、95%CI [+0.0021,+0.0335]"这类可信证据。
- 如果显著性模式在三包上都只是"少上线、指标略低"，就如实写成"以可信度换 0.0006 的验证分"，
  由报告论证取舍，并把默认值留在 plain（双轨保证两者都可交）。
- bootstrap 耗时必须报出来（资源效率项要用）。
