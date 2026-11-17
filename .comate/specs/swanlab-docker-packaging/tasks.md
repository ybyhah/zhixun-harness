# SwanLab 公开记录 + Docker 一键运行任务计划

> 对应 `.comate/specs/swanlab-docker-packaging/doc.md`。
> 原则：不改闭环算法，只补记录与打包；闭环逐批分数与改动前一致。SwanLab 默认 offline 不依赖账号/网络；Docker build/run 需网络时再申请升级。

- [ ] Task 1: SwanLab 真正接入闭环
    - 1.1: `loop.py` 构造 `SwanRecorder` 时传入 config（任务名、种子、关键超参）与 run_name
    - 1.2: 每批 `log` 增补统一 `score`、主指标 `value`、动作成本、drift 编码；保留只记工业模型指标的口径
    - 1.3: `record/swan.py` 增加 `public` 意图字段与离线目录提示，保留异常降级
    - 1.4: 三个配置 `swanlab` 段加 `public: true`、`mode: offline`，`enabled` 仍默认 false

- [ ] Task 2: 运行开关与离线记录验证
    - 2.1: `run_loop.py` 加 `--swanlab` 开关（等价 enabled=true），run_name=`{task}-{variant}`
    - 2.2: 本地以 offline 跑 A1，确认生成 `swanlog/`、`SwanRecorder.active` 为真、无降级
    - 2.3: `tests/test_record.py` 补：offline 建 run、log 不崩、未安装/异常走降级
    - 2.4: A1/A3 默认（不开 swanlab）逐批结果与改动前完全一致

- [ ] Task 3: SwanLab 上传与公开说明
    - 3.1: 新增 `scripts/swanlab_sync.py` 或运行说明：离线 `swanlog/` 上传命令
    - 3.2: 写清在网页端把记录设为**公开**的步骤（C8 硬性要求）
    - 3.3: 记录"评测离线→赛后上传→设公开"流程，对应汇总文档 12.4 Q19 的假设

- [ ] Task 4: Docker 一键运行
    - 4.1: 新增 `.dockerignore`：排除 data/、third_party/、outputs/、.git/、__pycache__/、*.rar、features.npz
    - 4.2: 新增 `Dockerfile`：python:3.11-slim + requirements.txt（river/nannyml 已注释不入镜像）+ 拷贝 src/scripts/configs + ENTRYPOINT run.sh
    - 4.3: 新增 `run.sh`：一键入口，`PYTHONHASHSEED=0`，按 `TASKS`/`--config` 选任务，数据 volume 挂载
    - 4.4: `docker build`，`docker images` 确认 ≤2GB、不含权重/大数据

- [ ] Task 5: 镜像冒烟与文档
    - 5.1: `docker run -v $PWD/data:/app/data <img> --config configs/a1_gas.yaml --no-plot` 跑通并产出结果
    - 5.2: 运行说明写 build/run/数据挂载/SwanLab 上传完整步骤
    - 5.3: 运行完整 `python3 -m pytest` 全绿

- [ ] Task 6: 结论与门槛核对
    - 6.1: 核对 C8（SwanLab 可公开）、C5（≤2GB 不含权重）、C11（一键可复现）是否满足
    - 6.2: 更新运行说明/结论，如实记录仍待确认项（离线上传口径以官方规范为准）
