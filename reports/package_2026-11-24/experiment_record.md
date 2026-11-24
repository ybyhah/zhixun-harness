# 2026-11-24 队友版本打包与提交记录

## 版本来源

- 来源目录：`D:/人工智能+(1)/浜哄伐鏅鸿兘+/harness`
- 版本说明：队友最新修改版
- 主要改动范围：C 隐藏集近似评估与截断、统一早停、平台历史指标解析、A/B/C 策略测试补充

## 验收结果

### 单元测试

执行：

```powershell
python -m pytest tests/test_submission_contract.py tests/test_submission_baseline_optimization.py -q
```

结果：86 项通过，0 项失败。

### Docker 构建

执行：

```powershell
docker buildx build --platform linux/amd64 --load --sbom=false --provenance=false `
  -f submission/Dockerfile `
  -t zhixun-submit:linux-amd64-20261124 .
```

结果：构建成功；镜像使用 `python:3.12-slim-bookworm`，未包含数据、权重和密钥。

### 容器烟测

执行：

```powershell
docker run --rm zhixun-submit:linux-amd64-20261124 describe
```

结果：返回 `contest.harness.workspace.v2`、SDK `1.0.2`，任务类型包含 classification 和 regression。

## 提交镜像

```powershell
docker save -o artifacts/submission_image_2026-11-24/zhixun-submit-linux-amd64.tar `
  zhixun-submit:linux-amd64-20261124
```

产物：

`artifacts/submission_image_2026-11-24/zhixun-submit-linux-amd64.tar`

大小约 71.4 MB。已检查 `manifest.json`，归档只有一个镜像标签 `zhixun-submit:linux-amd64-20261124`，适合平台要求的 Linux/amd64 单个未压缩 tar 上传格式。

## GitHub 提交

目标远端：`https://github.com/ybyhah/zhixun-harness.git`

最终源码提交：`70aa05c`（已推送到 `main`）。

本记录对应源码应提交以下内容：

- `submission/` 正式 SDK 代码与 A/B/C 基线
- `tests/` 契约和策略测试
- `scripts/c_hidden_like_experiment.py`
- `reports/` 中的实验记录和已有报告
- `RUNBOOK.md`、作品文档和 SDK 相关说明

不提交：镜像 tar、`base/` 数据包、权重、`.env`、临时输出和缓存目录。

## 复现注意事项

1. PowerShell 多行命令必须把 `docker buildx build` 放在第一行，反引号放在行尾。
2. 平台正式门控应使用 SDK 返回的 `submitted["summary"]["metrics"]`；本地指标只作诊断。
3. 新镜像上传前，应在目标平台执行一次 smoke test，并保留平台返回的 call ID、summary 和最终审计记录。
