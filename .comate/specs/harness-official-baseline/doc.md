# Harness 驱动官方 base/ 续训（对齐评分口径）

> 重大对齐：官方 `base/` 三个包才是评分对象——CPU NumPy MLP，从随包 `checkpoint.npz` **续训**，用 `baselines/` 的官方训练/预测代码，公开数据是 `public/*.jsonl`。
> 本轮把 Harness 重构为**驱动官方续训**的编排器：每个数据集 10 轮续训，自动决策每轮超参与是否上线，门控用官方验证指标，记录进 SwanLab，保留 ≥3 次"读历史→做实验→依据结果决策"。
> 旧 `src/harness/`（自研 HistGBDT + 适配器）**保留作对照基线**，不删除。

## 0. 为什么必须转向（证据）

- `base/<pkg>/baselines/`：官方 `train.py / model.py / data.py / predict.py / configs`。模型是 `Model`（NumPy MLP + Adam），A:128→32→6（4326 参），B:34→24→1 回归（865 参），C:24→32→1 回归（833 参）。
- 续训命令（实测可跑）：`python -m baselines.train --profile A --train public/train.jsonl --validation public/validation.jsonl --from-baseline checkpoint.npz --epochs 5 --output round-1 --stage-id round-1`，后续 `--resume round-k/checkpoint.npz`。
- 官方基线验证指标（不得低于，对应 C10）：**A Macro-F1 0.714**，**B RMSE 681 分钟 / MAE 563**，**C RMSE 52.9 循环 / MAE 38.9**。
- 关键纠偏：**B 是 RUL 回归（分钟），不是三分类**。`TASK.md` 明确"当前任务不使用正常/早期退化/严重退化"。我们之前的 A2 三分类与官方 B 不符，已记废（见旧 spec，保留作对照）。
- 平台独立跑 `predict.py` 于隐藏测试；训练日志自报指标不算数。

## 1. 目标与范围

新增一个**任务无关、配置驱动**的 Harness 驱动器，编排官方续训闭环：
- 对每个包（A/B/C）跑 **10 轮**续训；
- 每轮：读历史（`training.json`/`history.jsonl`/本地审计）→ 提假设（下一轮超参/是否训）→ 做实验（调用官方 `train.py`）→ 读 `validation_after` → 门控决策（上线/不上线/回滚）→ 记录；
- 门控与指标全部用**官方验证口径**（A Macro-F1↑，B/C RMSE↓）；
- 保留 ≥3 次完整闭环记录；SwanLab 记录官方工业模型的每轮训练（正好契合"SwanLab 记录工业模型"）；
- 不碰官方 `baselines/` 代码逻辑（本轮只驱动；改代码/改超参属后续优化，受官方续训约束：不可改架构、续训须保种子与特征schema）。

## 2. 设计

新增 `src/driver/`（与旧 `src/harness/` 并存）：

| 模块 | 职责 |
|---|---|
| `package.py` | 定位 `base/<pkg>`：读 `baseline-info.json`（profile/task/param_count）、校验 checkpoint sha、给出 train/val 路径与 profile config |
| `trainer.py` | 封装一次官方续训：构造并运行 `python -m baselines.train ...`（子进程，cwd=包目录），解析返回的 `training.json`；B 自动带 `--label-provenance`（status=draft, method 说明）；输出目录 `rounds/round-k` |
| `metric.py` | 从 `validation_after` 取主指标，统一成"越大越好"分数（回归用 -RMSE），复用 `harness.metrics` 的 composite/worst_stage/recovery |
| `policy.py` | 决策每轮超参与是否上线：规则版（按验证趋势调 epochs/lr，退化则回滚到上一个被采纳 checkpoint）；预留 LLM 慢通路接口 `Decider` |
| `gate.py` | 上线门控：验证指标较当前最优提升≥阈值且不低于官方基线才采纳；否则不上线并从上一个采纳 checkpoint 继续 |
| `audit.py` | 复用/对接 `harness.record`：每轮写 history_read/hypothesis/experiment/decision 四要素 + SwanLab |
| `loop.py` | 10 轮编排，串起上面各步；全程无人干预，异常降级不中断 |

入口 `scripts/run_official.py --package base/cpu-a-guide-20261110-v1 --rounds 10 [--swanlab]`。

## 3. 关键约束（来自官方代码，必须遵守）

- 输出目录必须是**新目录**；`--resume/--from-baseline` 恢复权重+Adam+归一化+种子，不重新初始化。
- 续训不得改 hidden/特征schema/task/profile/seed（`train.py` 会拒绝）。
- B 的 `--label-provenance` 必填：提供 `{"status":"draft","method":"官方 RUL=（N-i）分钟，确定性计算"}`。
- 归一化/目标缩放只在初始训练集拟合，续训沿用；验证/隐藏集不参与拟合。
- 每包用**自己的** `configs/<profile>.json` 与自己的 `checkpoint.npz`（注意 A/C 包里混带的旧 B.json 是三分类，不用；B 用 B 包的回归 config）。
- CPU 线程受 `BASELINE_CPU_THREADS`（1–8）约束，默认 1。

## 4. 影响文件

```
新增 src/driver/{__init__,package,trainer,metric,policy,gate,audit,loop}.py
新增 scripts/run_official.py            驱动入口
新增 configs/official_{a,b,c}.yaml      指向各 base 包 + 轮数/门控阈值/SwanLab
新增 tests/test_driver.py               用最小合成包/mock 子进程测试编排与门控（不跑真训练）
保留 src/harness/**                      旧自研实现，作对照基线，不改不删
修改 outputs/CONCLUSIONS.md             记录对齐官方口径后的结论与对照
```

旧 `run_loop.py`、`src/harness/` 保留；README/结论里标注"自研实现为对照，官方评分走 `run_official.py`"。

## 5. 验证

1. A：`run_official.py --package base/cpu-a-guide-20261110-v1 --rounds 10`，10 轮续训跑通，验证 Macro-F1 ≥ 官方基线 0.714 且逐轮有据可查；审计含 ≥3 完整闭环。
2. B：带 label-provenance 跑通，RMSE（分钟）不高于 681；确认回归路径正确。
3. C：跑通，RMSE（循环）不高于 52.9。
4. 固定种子可复现；SwanLab offline 记录生成。
5. `python3 -m pytest` 全绿（旧测试不受影响；新加 driver 测试用 mock，不跑真实多轮训练以控时）。

## 6. 边界与风险

| 情况 | 处理 |
|---|---|
| 某轮续训让验证退化 | 不上线，从上一个被采纳 checkpoint 继续（回滚语义） |
| 子进程失败/超时 | 记录降级，跳过该轮候选，闭环继续 |
| B 缺 label-provenance | 驱动器自动生成 draft provenance，内容如实写 RUL 定义 |
| 误把验证/隐藏集拟合统计量 | 不触碰官方归一化逻辑，只传数据路径 |
| 10 轮耗时 | A/B 秒级、C 单轮约 5s；10 轮可接受，记录耗时供资源效率项 |
| Docker/SwanLab | 复用上一个 spec 成果，打包对象改为 base/ + driver；SwanLab 记录官方 MLP 训练 |

## 7. 不在本轮（后续）

- 改官方 `baselines/` 代码做模型层优化（受续训约束，单独 spec）。
- LLM 慢通路（PyTRiO Qwen 读 history→提超参/改代码）：本轮留接口，先上规则策略。
- 策略层自改进（受控 RSI 第二层）。
