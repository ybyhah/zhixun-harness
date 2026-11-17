# 完成总结：接入 contest-harness SDK（真正的提交物）

## 做了什么
把比赛的真实接口（`contest.harness.workspace.v2`）接上了，产出可提交的 adapter，纠正了之前「在静态验证集上刷分」的方向偏差。

- 新增 `submission/contest_submission.py`：
  - `train(rnd)`：`rnd.parent.path("checkpoint.npz")` 取平台父检查点 → `rnd.model(...)` 真实调用一次模型让它读历史提续训超参（失败回退官方默认，但调用已发生，满足「每轮必有 model_call」）→ 子进程跑官方 `baselines.train` 从父检查点续训 `/data` 投影 → `rnd.submit_candidate(...)`（checkpoint + predict_argv）→ `rnd.finish(...)`（真实 reason + 非空 model_call_ids）。
  - `predict(sample, model_dir)`：从 `CONTEST_ARTIFACT_PATH` 加载官方 `Model`，复用官方 `baselines.predict`，分类/回归按合同输出；检查点按样本缓存。
  - `TASKS=("classification","regression")`。
- 新增 `submission/baselines/`：官方训练/预测代码（`train.py` 取 B 包的超集版本，`configs/` 用各自数据集的正确配置：B 为回归 target_unit=minutes），**不含权重**。
- 新增 `submission/Dockerfile`：装 SDK + baselines + 提交模块 + `harness.py`→`/app`；零密钥、非 root、不打包 `base/`。
- 新增 `scripts/local_sdk_test.py`：describe / train 一轮 / predict 三段离线自测。

## 为什么这样接（关键事实，已核对源码）
- 模型只经 `rnd.model()`/`rnd.transport()`，密钥在平台密钥栏，镜像内零密钥；之前 `src/driver/llm.py` 用 `PYTRIO_*`+certifi 直连只是本地自测。
- `finish` 要求 `model_call_ids` 非空 → 每轮必须真实调模型；之前「停滞才唤醒 + 熔断少调」在提交物里不成立（保留为本地成本分析）。
- 自评无效：候选分数由平台 `submit_candidate` 后测量。
- 官方 baselines：`model.py/data.py/predict.py` 三包一致，仅 `train.py` 与 `B.json` 不同；B 的 `train.py` 是超集（含 RUL 回归），故镜像里采用它。

## 验证
- `python3 scripts/local_sdk_test.py` 三段全过：describe 协议正确；train 真实跑官方续训、产出 132KB 候选、finish 带 `model_call_ids`；predict 对 3 个样本输出合法（top=Ethanol）。
- `pytest -q` 308 条全绿（未改 `src/`）。

## 定位调整
`src/driver`（run_official/compare/adjudicate/hybrid/熔断）与 `src/harness` 降级为**本地方法探索与证据**，写进技术报告/答辩（对照实验、split-half 裁决、漂移流分析），不进提交物；真正提交走本 SDK adapter。

## 后续 / 未在本地覆盖
- `train` 的完整平台往返依赖 `/previous`、`/model` 等平台挂载（SDK 里是硬编码绝对路径），本地用 FakeRound 验证了逻辑与官方续训子进程；端到端需在镜像/平台试运行。
- 建议：构建镜像 `docker build -f submission/Dockerfile -t zhixun-submit .`，先按 SDK 文档离线试运行，再上平台试提交；模型访问与计费由平台提供，务必只在平台密钥栏填 PyTRiO Key。
