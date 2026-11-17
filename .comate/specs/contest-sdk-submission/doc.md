# 接入 contest-harness SDK：真正的提交物

## 要解决的问题（前面工作的偏差）
读完 `contest-harness-sdk-1.0.2/HARNESS_SDK.md` 和源码后确认，之前围绕 `src/driver/run_official.py` 的工作与真实比赛接口有三处硬冲突，必须纠正：
1. **流程由平台逐轮驱动**：平台反复调用 `/app/harness.py train`，每轮给 `rnd`（父检查点在 `/previous`、当前阶段数据投影在 `/data/`、冻结 `rnd.task`）。不是我们在本地 for 循环刷同一份 train/val。我之前的「10 轮续训刷验证集」是单批过拟合，不是流式适应。
2. **每轮必须至少一次真实模型调用**（`finish` 要求 `model_call_ids` 非空且来自 `rnd.model()`/transport）。我加的「停滞才唤醒 + 熔断少调」直接违规——提交物里每轮都要调模型。
3. **模型只能走 SDK 网关**（`rnd.model()` / `rnd.transport()`），密钥在平台密钥栏，镜像内不得出现。`src/driver/llm.py` 用 `PYTRIO_*`+certifi 直连 pytrio.com 的做法只能当本地自测，不是提交调用方式。另外**自报分数无效**，由平台 `submit_candidate` 后测量。

## 目标
产出真正可提交的 SDK adapter：`contest_submission.py`（`train`/`predict`）+ 提交用 `Dockerfile` + 本地离线自测脚本。复用官方 `baselines/` 训练与预测代码；`src/driver`/`src/harness` 降级为「本地方法探索与证据」，不进提交物。

## 关键事实（已核对）
- 官方 `baselines/` 代码：A 与 C 完全一致（同哈希），B 不同（回归 + label-provenance）。故**每轮候选要把父产物自带的 baselines 代码一并带下去**（`code_paths`），预测时从产物加载，而不是假设镜像里一份通用代码。
- `baselines/predict.py` 的 `predict(model, inputs)`：`Model.load(ckpt)` → 校验特征 schema → 分类返回 `[{label,score}]`，回归返回 `{value}`。正是 `predict(sample, model_dir)` 要包的逻辑。
- 续训走官方 `python -m baselines.train --resume <父ckpt> --train /data/train.jsonl --validation /data/validation.jsonl`，梯度只用 train；B 需 `--label-provenance`。

## 影响文件
- 新增 `/Users/zhangtianshuo/Desktop/harness/submission/contest_submission.py`：`train(rnd)`、`predict(sample, model_dir)`、`TASKS=("classification","regression")`。
- 新增 `/Users/zhangtianshuo/Desktop/harness/submission/Dockerfile`：装 SDK（`pip install ./contest-harness-sdk-1.0.2`）、放 `template/harness.py`→`/app/harness.py`、放 `contest_submission.py`→`/app/`；**不含密钥、不含 base/ 权重**；`USER` 非 root。
- 新增 `/Users/zhangtianshuo/Desktop/harness/scripts/local_sdk_test.py`：用 SDK 的 JSONL 宿主替身（仿 `tests/support.py`）在本地跑 `describe`/一轮 `train`/若干 `predict`，断言产出候选、调用过模型、预测形状合法。
- 复用（不改）：官方各包 `baselines/`；`src/driver/*` 与 `src/harness/*` 保留为本地证据，不进镜像。

## 实现细节
`train(rnd)`：
1. `parent_ckpt = rnd.parent.path("checkpoint.npz")`；按父 `code_paths` 把 `baselines/*` 取到 `/workspace/pkg/`（`rnd.parent.path` 逐个校验摘要）。
2. 读 `rnd.history()`，用 `rnd.model([...本轮数据摘要/历史/问题...], temperature=0)` 让 27B 提 `{epochs, learning_rate, hypothesis}`；解析+裁剪区间，失败则回退官方默认超参（**但模型调用已发生，满足每轮必调**）。
3. 子进程跑官方续训：`python -m baselines.train --profile <task 对应> --train <rnd train 投影> --validation <val 投影> --resume <parent_ckpt> --epochs E --learning-rate LR --output /workspace/cand`（B 加 `--label-provenance`）。
4. `rnd.submit_candidate(/workspace/cand, hypothesis=..., change_summary=..., configuration={epochs,lr,model_call_id}, seed, checkpoint_paths=["checkpoint.npz"], code_paths=[带下去的 baselines 文件], predict_argv=["python3","/app/harness.py","predict"])`。
5. `rnd.finish(selected_artifact_id=..., reason=<基于模型回复与平台返回摘要的真实理由>, history_ids=..., model_call_ids=list(rnd.call_ids))`。

`predict(sample, model_dir)`：把 `model_dir`（/model）与其中 baselines 代码加入 `sys.path`，`Model.load(os.environ["CONTEST_ARTIFACT_PATH"])`，按 `baselines.predict.predict` 的逻辑对 `sample.inputs` 产出；分类 `[{label,score}]`、回归 `{value}`（非负）。

## 边界与异常
- 模型调用失败/超时/坏 JSON：回退官方默认超参续训，但**本轮仍已调用过模型**（调用在前、解析在后）；绝不让一轮没有 model_call。
- 父摘要不符：`rnd.parent.path` 抛 `PARENT_DIGEST_MISMATCH`，不吞掉，让平台按失败处理。
- `/workspace` 候选：≤64 文件、≤64MiB；只放普通文件；输出目录新建。
- 预测：每行先写结果再读下一行；schema 不符按官方 `predict` 抛错（该次测量失败，不伪造）。
- 密钥：镜像与代码零密钥；模型只经 `rnd.model()`/`rnd.transport()`。

## 本地测试方式
- `scripts/local_sdk_test.py` 用 JSONL 宿主替身喂一帧 `start`（含假父产物与小 train/val 投影），跑 `train` 一轮，断言：有 `candidate_request`、有 `model_request`、`round_complete` 的 `model_call_ids` 非空；再喂几行样本跑 `predict`，断言输出形状合法。
- `python3 /app/harness.py describe` 返回协议 JSON 且 `tasks` 正确。
- 镜像构建后用 SDK 文档的离线方式试运行；最后平台试提交。

## 预期结果
- `describe/train/predict` 三入口在本地替身下全通过；每轮有真实模型调用、产出合法候选、预测形状正确。
- 提交物不含密钥、不含权重，续训严格从平台父检查点继续。
- `pytest` 既有用例不受影响（未改 `src/`）。

## 定位调整（需确认）
`src/driver`（run_official、compare_deciders、adjudicate_deciders、hybrid/熔断）与 `src/harness` 从「提交物」**降级为本地方法探索与证据**：对照实验、split-half 裁决、漂移流分析写进技术报告与答辩材料，但真正提交走本 SDK adapter。其中「每轮必调模型」使熔断/停滞省调用的逻辑不进提交物（可在报告里作为「本地成本分析」保留）。

