# Harness 驱动官方 base/ 续训任务计划

> 对应 `.comate/specs/harness-official-baseline/doc.md`。
> 原则：不改官方 `baselines/` 代码，只驱动；旧 `src/harness/` 保留作对照不动。新驱动器全程无人干预、异常降级不中断。先在 A 跑通，再扩 B/C。

- [x] Task 1: 包定位与官方续训封装
    - 1.1: 新增 `src/driver/package.py`：读 `baseline-info.json`（profile/task/param_count）、校验 `checkpoint.npz` sha256、给出 train/val 路径与 profile config 路径
    - 1.2: 新增 `src/driver/trainer.py`：以子进程在包目录运行 `python -m baselines.train ...`，首轮 `--from-baseline`、后续 `--resume`，输出到 `rounds/round-k`，解析返回 `training.json`
    - 1.3: B 自动生成并传 `--label-provenance`（status=draft，method 写明 RUL=(N-i) 分钟确定性计算）
    - 1.4: 新增 `tests/test_driver.py`：mock 子进程验证命令拼装（首轮/续轮、B 带 provenance、输出新目录）

- [x] Task 2: 指标与门控
    - 2.1: 新增 `src/driver/metric.py`：从 `validation_after` 取主指标，统一成越大越好（回归 -RMSE），复用 `harness.metrics`
    - 2.2: 新增 `src/driver/gate.py`：验证较当前最优提升≥阈值且不低于官方基线才上线；否则不上线
    - 2.3: 回滚语义：未上线则下一轮从上一个被采纳 checkpoint 继续
    - 2.4: 测试：提升则上线、退化则不上线并回滚、低于官方基线拒绝

- [x] Task 3: 决策策略与审计
    - 3.1: 新增 `src/driver/policy.py`：规则版决定每轮 epochs/lr 与是否训；预留 `Decider` 接口给 LLM 慢通路
    - 3.2: 新增 `src/driver/audit.py`：对接 `harness.record`，每轮写 history_read/hypothesis/experiment/decision 四要素
    - 3.3: SwanLab 记录每轮官方验证指标（offline，复用 SwanRecorder）
    - 3.4: 测试：审计四要素齐全、≥3 完整闭环计数正确

- [x] Task 4: 10 轮编排闭环
    - 4.1: 新增 `src/driver/loop.py`：串起 读历史→决策→续训→门控→记录，跑满 rounds
    - 4.2: 异常/子进程失败/超时降级不中断，记录降级事件
    - 4.3: 新增 `scripts/run_official.py --package <dir> --rounds 10 [--swanlab]` 入口
    - 4.4: 偏差：未建 `configs/official_*.yaml`（脚本以 `--package` 驱动，避免无用配置文件，YAGNI）

- [x] Task 5: A 端到端验证
    - 5.1: A 10 轮跑通，Macro-F1 0.7140 → 0.7264（≥ 官方基线），逐轮有据
    - 5.2: 审计 9 条完整闭环；SwanLab offline 可生成
    - 5.3: 固定种子复跑一致

- [x] Task 6: 扩展 B/C
    - 6.1: B（RUL 分钟）带 label-provenance 跑 10 轮，RMSE 681.16 → 666.88（≤ 基线）
    - 6.2: C（RUL 循环）跑 10 轮，RMSE 52.87 → 51.38（≤ 基线）
    - 6.3: 三包耗时 5/11/78s 已记录

- [x] Task 7: 收尾
    - 7.1: `python3 -m pytest` 全绿（231 项，旧测试不受影响）
    - 7.2: 更新 `outputs/CONCLUSIONS.md`：顶部加官方口径结论 + "自研为对照"说明
    - 7.3: 后续（改 baselines 优化、LLM 慢通路、策略层）标注入下一轮
