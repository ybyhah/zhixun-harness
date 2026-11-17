# 接入 contest-harness SDK：真正的提交物

- [x] Task 1: 搭建 `submission/contest_submission.py` 骨架
    - 1.1: 定义 `TASKS=("classification","regression")`、`train`、`predict` 三个入口，`describe` 可通过
    - 1.2: 公共工具：从 `rnd.task` 取 task/metric/方向；定位 task 对应的官方 profile

- [x] Task 2: 实现 `train(rnd)`
    - 2.1: 取父检查点 `rnd.parent.path("checkpoint.npz")`，按父 `code_paths` 取 baselines 代码到 `/workspace`
    - 2.2: `rnd.history()` + `rnd.model(...)` 让 27B 提超参（失败回退官方默认，但保证本轮已调用模型）
    - 2.3: 子进程跑官方 `baselines.train` 续训（B 加 `--label-provenance`）到 `/workspace/cand`
    - 2.4: `submit_candidate`（checkpoint+code+predict_argv）与 `finish`（真实 reason、model_call_ids 非空）

- [x] Task 3: 实现 `predict(sample, model_dir)`
    - 3.1: 从 `CONTEST_ARTIFACT_PATH` 加载 `Model`，`model_dir` 下 baselines 代码加入 `sys.path`
    - 3.2: 分类输出 `[{label,score}]`、回归输出非负 `{value}`；schema 不符按官方逻辑抛错

- [x] Task 4: 提交用 `submission/Dockerfile`
    - 4.1: 安装 SDK、放 `template/harness.py`→`/app/harness.py`、`contest_submission.py`→`/app/`
    - 4.2: 零密钥、不打包 base/ 权重、非 root 用户、`ENTRYPOINT python3 /app/harness.py`

- [x] Task 5: 本地离线自测 `scripts/local_sdk_test.py`
    - 5.1: JSONL 宿主替身喂 `start` + 小投影，跑 `train` 一轮，断言有候选、有模型调用、finish 的 model_call_ids 非空
    - 5.2: 跑 `describe` 校验协议 JSON；喂样本跑 `predict` 校验输出形状

- [x] Task 6: 验证
    - 6.1: `local_sdk_test.py` 全通过；`python3 harness.py describe` 正确
    - 6.2: `pytest -q` 既有用例不受影响（未改 src/）
