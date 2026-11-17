# 完成总结：让 PyTRiO 慢通路稳定使用 Qwen3.8-27B

## 目标
让 LLM 慢通路固定用 `Qwen/Qwen3.8-27B`，解决冷启动超时与思考模式 `content=null`，并打通实际调用，不再逢轮必降级；无端点时仍自动降级（保持 C4 离线约束）。

## 改动清单（`src/driver/llm.py`）
- `__init__`：`timeout_s` 可空，默认读 `PYTRIO_TIMEOUT_S`（默认 300），吸收 27B 冷启动。
- `complete`：
  - 请求体加**顶层** `enable_thinking: false`（关思考，直出 JSON）。实测端点**不支持** `chat_template_kwargs`（传了返回 400），`reasoning_effort` 也压不住思考；只有顶层 `enable_thinking` 生效。
  - 用 `certifi` 的 CA 证书构造 SSL context 传给 `urlopen`，修复 macOS 的 `CERTIFICATE_VERIFY_FAILED`。
  - 取文本：`content` 为空回退 `message.reasoning`，皆空抛错交上层降级。
- 配套：`.env` 加 `PYTRIO_TIMEOUT_S=300`；`tests/test_llm.py` 加 2 条超时用例；`RUNBOOK.md` 补环境变量行。

## 测试中发现并修复的两个真 bug
1. **SSL 校验失败**：Python urllib 缺系统 CA → 每次调用瞬间 `URLError` → 每轮降级。首次「1.4s 跑完、llm_degraded=true」即此问题。用 certifi 修复。
2. **关思考参数用错**：原设计的 `chat_template_kwargs` 被端点拒绝（HTTP 400），反而弄坏请求。改为顶层 `enable_thinking: false`。

## 验证结果
- `pytest tests/test_llm.py`：13 条全绿。
- 端到端实跑（27B，1 轮）：`决策器：LLM 慢通路（PyTRiO 可用）`，用时 157.7s，**降级 0**；audit 记录 `hypothesis=[LLM] ...`、`llm_degraded=False`、`tokens=209`、LLM 给出 epochs=20 / lr=0.01。即 27B 真正驱动了调参。
- 该轮结果 REJECT（macro_f1 0.6629 < 基线 0.7140）：是 LLM 首轮激进超参的模型效果问题，显著性门控正确拦下，属正常行为、非 bug。

## 使用方式
```bash
cd /Users/zhangtianshuo/Desktop/harness
set -a; source .env; set +a
python3 scripts/run_official.py --package base/cpu-a-guide-20261110-v1 --rounds 3 --decider llm
```

## 已知限制 / 后续
- 27B 冷启动约 128s~180s（一轮总耗时 ~158s 即受此主导），`PYTRIO_TIMEOUT_S=300` 能等到，但每轮慢是共享端点的固有成本；彻底消除需平台侧常驻/独占部署。
- `certifi` 依赖：本地已装（随 swanlab/requests）；Docker 镜像同理。无 certifi 时回退系统默认 context。
- 首轮 LLM 无历史、易给激进超参被 gate 拒；多跑几轮（≥3）配合 history 后更能体现慢通路价值。
