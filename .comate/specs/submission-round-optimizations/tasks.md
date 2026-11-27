# 提交物单轮优化任务计划

- [ ] Task 1: 关闭思考并加兜底（S1）
    - 1.1: `_ask_model` 调用 `rnd.model(..., enable_thinking=False)`
    - 1.2: 捕获 `HarnessError` 后不带该选项重试一次
    - 1.3: `local_sdk_test.py` 补测：正常传参、宿主拒绝时重试

- [ ] Task 2: 用 /state 保存决策轨迹（S2）
    - 2.1: `train()` 开头 `state_workspace` 并读取 `trail.jsonl`，异常时置空
    - 2.2: 提示词历史改为"轨迹最近 6 行 + 平台记录紧凑 JSON（每条 ≤300 字节）"
    - 2.3: 正常分支与续训失败分支都在 `finish` 前追加轨迹并 `save_state`，异常不影响本轮
    - 2.4: `local_sdk_test.py` 补测：首轮无状态、跨轮读到上一轮轨迹、状态损坏
    - 2.5: 跑全部本地测试

- [ ] Task 3: 代理漂移评测脚本（S4，先只评估现有机制）
    - 3.1: 新增 `scripts/proxy_drift_eval.py`，把 `data/raw/batch*.dat` 转成 A 包格式 JSONL
    - 3.2: 构造阶段 s=1..8（训练 s / 验证 s+1 / 测试 s+2），用 FakeRound 逐阶段驱动真实 `train()`
    - 3.3: 实现策略：静态 / 每轮必更新 / 门控 0.95 / 门控 0.8
    - 3.4: 输出各策略测试 Macro-F1 的阶段均值、最差阶段、相对静态的平均增益

- [ ] Task 4: 多候选 + 半分验证原型并对照（S3）
    - 4.1: `decision.py` 新增 `split_halves(ids, seed)`（按 id 的 sha256 确定性切半，样本过少返回 None）
    - 4.2: `contest_submission.py` 支持解析 1–3 组超参，逐组续训，选择半挑最优、门控半做 bootstrap，只提交胜者
    - 4.3: 在代理评测中加入"多候选+半分 0.95 / 0.8"两个策略并运行
    - 4.4: 按采纳规则（测试均值更高且最差阶段不更差）决定：采纳则保留代码并补测试；否则回退 S3 代码，保持单候选与现有阈值

- [ ] Task 5: 回归验证与镜像重建
    - 5.1: 跑 `local_sdk_test.py` 全部用例
    - 5.2: 重建 linux/amd64 单平台镜像，离线验证 `describe`、`predict`
    - 5.3: `docker save` 重新导出未压缩 tar，记录 MD5 与大小，确认 ≤2GB

- [ ] Task 6: 更新作品文档并推送
    - 6.1: `作品文档.md` 更新 S1/S2（及采纳时的 S3）机制描述
    - 6.2: 加入代理漂移实验对照表与局限说明
    - 6.3: 提交并推送到 GitHub（推送前做密钥扫描）
