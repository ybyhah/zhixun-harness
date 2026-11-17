# SwanLab 公开记录 + Docker 一键运行打包

> 补齐资格硬门槛：C8（全部训练/评估记录在 SwanLab 且公开）、C5（Docker ≤2GB、不打包模型权重）、C11（代码可复现、一键运行）。
> 不改闭环算法，只补"记录"与"打包运行"两块工程设施。

## 0. 与比赛要求对齐

| 要求 | 本轮落实 |
|---|---|
| C8 训练/评估记录在 SwanLab，且**必须公开** | 真正启用 SwanRecorder，逐批上报工业模型指标；提供离线记录 + 事后上传并设公开的流程 |
| C4 评测环境无外网 | SwanLab 用 **offline 模式**在评测时落本地，赛后 `swanlab upload` 同步并设公开（对应汇总文档 12.4 Q19 的待确认，先按离线+事后上传实现） |
| C5 镜像 ≤2GB、**严禁打包模型权重** | slim 基础镜像 + 仅运行期依赖；`.dockerignore` 排除 data/、third_party/、outputs/、权重；工业模型运行时训练，不入镜像 |
| C11 一键运行、可复现 | `run.sh` 一条命令跑通；固定种子；复现文档 |
| "SwanLab 记录对象是工业模型" | 只上报工业模型训练/评估指标，不记 Harness 自身调用 |

## 1. SwanLab 接入

现状：`src/harness/record/swan.py` 的 `SwanRecorder` 已实现、异常自动降级，但三个配置 `swanlab.enabled: false`，`loop.py` 只 log 了 `train_samples/train_seconds` 与每批主指标，从未真正建过 run。

改动：
- `loop.py`：`SwanRecorder` 的 `config` 传入任务名/种子/关键超参；每批 `log` 增补统一 `score`、主指标 `value`、`drift_type` 编码、动作成本，`finish()` 已在 run 结尾调用。
- 配置：新增 `swanlab.public: true`（写入 run 的可见性意图，作为文档/上传提示）与 `swanlab.mode`（默认 `offline`，安全）；默认 `enabled` 仍 false，由 `run.sh`/环境变量或 `--set record.swanlab.enabled=true` 打开，避免本地无账号时报错。
- `run_loop.py`：加 `--swanlab` 开关，等价 `--set record.swanlab.enabled=true`；并把 run_name 设为 `{task}-{variant}`。
- 新增 `scripts/swanlab_sync.py`（或在复现文档写明命令）：评测后把本地 `swanlog/` 上传并提示在网页端设为公开。

验证：本地用 `mode=offline` 跑一次，确认生成 `swanlog/` 且 `SwanRecorder.active` 为真、无降级；cloud 模式因需账号，只在文档给步骤，不在 CI 强跑。

## 2. Docker 一键运行

新增文件：
- `Dockerfile`：`python:3.11-slim` 基础；装 `requirements.txt`（river/nannyml 已注释，不入镜像）；拷贝 `src/ scripts/ configs/`；`ENTRYPOINT ["/app/run.sh"]`。
- `run.sh`：一键入口。默认依次跑 A1/A3（有随镜像可获取的数据时），A2 依赖特征缓存，按环境变量 `TASKS` 选择；参数透传 `--config`。可复现固定 `PYTHONHASHSEED=0`、种子来自配置。
- `.dockerignore`：排除 `data/`、`third_party/`、`outputs/`、`.git/`、`__pycache__/`、`*.rar`、`features.npz`、虚拟环境——保证 ≤2GB 且不含权重/大数据。
- `README.docker` 段落或 `scripts/` 内运行说明：build、run（数据以 volume 挂载 `-v`）、SwanLab 上传。

数据策略：官方数据上传到平台固定目录，**镜像不打包数据**；本地演示用 `-v $PWD/data:/app/data` 挂载。镜像只带代码与依赖。

验证：
- `docker build` 成功，`docker images` 确认大小 ≤2GB。
- 冒烟：`docker run --rm -v $PWD/data:/app/data <img> --config configs/a1_gas.yaml --no-plot` 跑通并在挂载的 outputs 看到结果。
- 若 build 需联网拉基础镜像，按需申请升级权限。

## 3. 影响文件

| 文件 | 改动 |
|---|---|
| `src/harness/loop.py` | SwanRecorder 传入 config 与更全的 per-batch 指标 |
| `src/harness/record/swan.py` | 增加 `public` 意图字段与离线目录提示（最小改动，保留降级） |
| `configs/*.yaml` | `swanlab` 段加 `public`、`mode: offline` 默认 |
| `scripts/run_loop.py` | 加 `--swanlab` 开关与 run_name |
| `scripts/swanlab_sync.py` | 新增：离线记录上传并设公开的脚本/说明 |
| `Dockerfile`、`run.sh`、`.dockerignore` | 新增：一键运行镜像 |
| `tests/test_record.py` | 补：offline 模式建 run、log 不崩、降级路径 |
| `outputs/CONCLUSIONS.md` 或运行说明 | 记录 SwanLab 与 Docker 使用步骤 |

## 4. 边界与风险

| 情况 | 处理 |
|---|---|
| 本地无 SwanLab 账号 / 无网 | 默认 offline，落本地 swanlog；cloud 仅文档说明，不强跑 |
| 评测环境离线 | SwanLab offline 记录，赛后上传并设公开 |
| 镜像超 2GB | 用 slim 基础 + .dockerignore 排除数据/第三方；build 后校验大小，超则裁依赖 |
| 误打包权重/大数据 | .dockerignore 明确排除 data/、*.rar、features.npz、third_party/ |
| docker/网络不可用 | 文件照常产出，build/run 验证按需申请升级；失败则给出命令供手动执行 |

## 5. 预期产出

1. 三任务跑一次 SwanLab offline 记录成功、无降级，含上传并设公开的步骤说明。
2. 可构建的 Docker 镜像（≤2GB，不含权重/大数据），`run.sh` 一键跑通，volume 挂载数据。
3. `python3 -m pytest` 全绿；闭环算法行为不变。
