# 让 PyTRiO 慢通路稳定使用 Qwen3.8-27B

- [x] Task 1: 改造 `src/driver/llm.py` 支持 27B
    - 1.1: `LLMClient.__init__` 的 `timeout_s` 改为可空，默认读 `PYTRIO_TIMEOUT_S`（float，默认 300）
    - 1.2: `complete()` 请求体加入 `chat_template_kwargs={"enable_thinking": False}` 关闭思考模式
    - 1.3: `complete()` 取文本改为 `content` 为空回退 `message.reasoning`，两者皆空抛异常
    - 1.4: 确认 `available` 判定与无端点时 `RuntimeError` 降级语义不变

- [x] Task 2: 更新本地配置 `.env`
    - 2.1: 确认 `PYTRIO_MODEL=Qwen/Qwen3.8-27B`
    - 2.2: 新增 `PYTRIO_TIMEOUT_S=300` 并加注释说明冷启动原因

- [x] Task 3: 补充单元测试 `tests/test_llm.py`
    - 3.1: 新增用例验证 `PYTRIO_TIMEOUT_S` 被读入 `LLMClient.timeout_s`
    - 3.2: 确认既有用例（可用性、JSON 抽取、降级路径）不受影响

- [x] Task 4: 同步文档 `RUNBOOK.md`
    - 4.1: 环境变量表补 `PYTRIO_TIMEOUT_S` 一行（默认 300，说明用于 27B 冷启动）

- [x] Task 5: 验证
    - 5.1: 运行 `pytest tests/test_llm.py` 全绿
    - 5.2: 运行 `verify_contest_conditions.py` 确认无端点时离线自检仍通过
