# 让 PyTRiO 慢通路稳定使用 Qwen3.8-27B

## 背景与需求
- 决策：LLM 慢通路固定使用 `Qwen/Qwen3.8-27B`（不换小模型）。
- 实测问题（已联网验证）：
  - 27B 在共享端点是 vLLM 冷启动，首字节 128s、第二次 >180s；生成本身很快。
  - 27B 默认开思考模式，小 `max_tokens` 下 `choices[0].message.content` 为 `null`，token 全烧在 `reasoning`。
- 现状 `src/driver/llm.py`：默认 `timeout_s=20.0`、直接取 `message["content"]`。→ 27B 每轮必超时或取不到文本 → `LLMDecider.decide` 走 `except` 降级规则，LLM 实际没生效。
- 目标：用最小改动让 27B 真正返回可解析 JSON、不再逢轮必降级；无端点时仍自动降级（保持 C4 离线约束与「慢通路绝不中断闭环」语义）。

## 技术方案
1. 超时可配置并调大：新增环境变量 `PYTRIO_TIMEOUT_S`（float，默认 300），吸收冷启动等待。
2. 关闭思考模式：请求体加**顶层** `enable_thinking=false`（实测端点不支持 `chat_template_kwargs`，传了会 400；`reasoning_effort` 也压不住思考），让 27B 直出 JSON 而非 reasoning，省 token、提速、规避 content=null。
3. 健壮取文本：`complete()` 读 `content` 为空时回退 `message.reasoning`，两者皆空才算失败（交上层降级）。
4. TLS 校验：用 `certifi` 的 CA 证书构造 SSL context 传给 `urlopen`（macOS/Python.framework 缺系统 CA 会 `CERTIFICATE_VERIFY_FAILED`，导致每轮瞬间 URLError 降级）。
5. 不改决策/门控/循环逻辑，不改 `run_official.py` 的 CLI；`LLMDecider` 的 try/except 降级语义保持不变。

## 影响文件
- `/Users/zhangtianshuo/Desktop/harness/src/driver/llm.py`（核心改动）
  - `LLMClient.__init__`：`timeout_s` 改为可空，默认读 `PYTRIO_TIMEOUT_S`（默认 300）。
  - `LLMClient.complete`：payload 增加 `chat_template_kwargs`；取文本增加 reasoning 回退与空值校验。
- `/Users/zhangtianshuo/Desktop/harness/.env`：模型已是 `Qwen/Qwen3.8-27B`；补一行 `PYTRIO_TIMEOUT_S=300` 并加注释说明冷启动。
- `/Users/zhangtianshuo/Desktop/harness/tests/test_llm.py`：补 1 条单测，验证 `PYTRIO_TIMEOUT_S` 被读入 `LLMClient.timeout_s`。
- `/Users/zhangtianshuo/Desktop/harness/RUNBOOK.md`：环境变量表补 `PYTRIO_TIMEOUT_S` 一行（文档同步，可选）。

## 实现细节
`LLMClient.__init__`（超时可配置）：
```python
def __init__(self, base_url=None, model=None, api_key=None, timeout_s=None):
    self.base_url = (base_url or os.environ.get("PYTRIO_BASE_URL", "")).rstrip("/")
    self.model = model or os.environ.get("PYTRIO_MODEL", "")
    self.api_key = api_key or os.environ.get("PYTRIO_API_KEY", "")
    if timeout_s is None:
        timeout_s = float(os.environ.get("PYTRIO_TIMEOUT_S", "300"))
    self.timeout_s = timeout_s
```

`LLMClient.complete`（关思考 + 健壮取文本）：
```python
payload = json.dumps({
    "model": self.model,
    "messages": [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
    "temperature": 0.2, "max_tokens": 512,
    "enable_thinking": False,
}).encode("utf-8")
...
# urlopen 传 certifi 构造的 _SSL_CTX：
#   urllib.request.urlopen(req, timeout=self.timeout_s, context=_SSL_CTX)
msg = body["choices"][0]["message"]
text = msg.get("content") or msg.get("reasoning") or ""
if not text.strip():
    raise ValueError("PyTRiO 返回空文本")
tokens = int(body.get("usage", {}).get("total_tokens", 0))
return text, tokens
```

## 边界与异常
- 无端点/缺 env：`available=False`，`complete` 抛 `RuntimeError` → 上层降级（语义不变）。
- 冷启动超过超时：urllib 抛超时 → 当轮降级规则，下一轮继续重试，闭环不中断。
- 服务器忽略 `enable_thinking`：content 可能仍为 null → reasoning 回退兜底；若 reasoning 内含 JSON，`_extract_json` 仍可抽取。
- 热模型（如 4B）：300s 超时仅在异常时才会等满，正常亚秒返回，无副作用。

## 数据流
`run_official.py --decider llm` → `LLMDecider.decide(history)` → `LLMClient.complete(system,user)` → `POST {base}/v1/chat/completions`（`enable_thinking=false`）→ 解析 `content`/`reasoning` → `_extract_json` → `_clamp` → `RoundPlan`；任一步异常 → `fallback.decide`（规则兜底）。

## 预期结果
- 配置 27B + `PYTRIO_TIMEOUT_S=300` 后，慢通路能拿到真实 JSON 超参，`last_degraded=False`（冷启动那一轮仍可能降级，属预期）。
- `pytest tests/test_llm.py` 全绿；无端点时 `verify_contest_conditions.py` 离线自检仍通过。
- 代价：每轮最坏需等到冷启动完成（分钟级），是共享端点用 27B 的固有成本；彻底消除需平台侧常驻/独占部署。

