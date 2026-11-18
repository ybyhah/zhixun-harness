# C Platform Feedback Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Workspace policy forbids subagents, so execution must remain inline in the current agent.

**Goal:** Make C use all platform rounds for one-epoch, non-repeating learning-rate experiments while keeping platform public summaries authoritative and hidden-report results as the post-run acceptance gate.

**Architecture:** Keep all cross-round behavior in `submission/contest_submission.py`. Add only optional arguments to existing helpers so A/B retain their current behavior, force C to one epoch, keep all compact C history in the Qwen prompt, prevent repeated C learning rates under one parent, and exempt C from resource skipping. Update focused contract tests first, then document the observable boundary between public platform summaries and post-run hidden metrics.

**Tech Stack:** Python 3.11/3.12, pytest, contest-harness-sdk 1.0.2, Docker.

---

## File map

- Modify `submission/contest_submission.py`: C-specific search policy and helper wiring.
- Modify `tests/test_submission_contract.py`: pure-policy and train-path regression tests.
- Modify `作品文档.md`: document C single-step search and the public/hidden metric boundary.
- Modify `RUNBOOK.md`: add the post-trial hidden C acceptance check.

### Task 1: Specify C search behavior with failing tests

**Files:**
- Modify: `tests/test_submission_contract.py`
- Test: `tests/test_submission_contract.py`

- [x] **Step 1: Replace the latest-three-only prompt test with default and C-all-history coverage**

Keep the existing default assertion and add an explicit all-history assertion:

```python
def test_history_prompt_defaults_to_latest_three_but_can_include_all_rounds():
    history = [
        {"id": f"r{i}", "round_index": i,
         "summary": {"metrics": {"rmse": float(i)}}, "configuration": {}}
        for i in range(1, 6)
    ]
    default_prompt = cs._history_prompt(history)
    assert "轮1:" not in default_prompt and "轮2:" not in default_prompt
    assert all(f"轮{i}:" in default_prompt for i in (3, 4, 5))

    full_prompt = cs._history_prompt(history, include_all=True)
    assert all(f"轮{i}:" in full_prompt for i in range(1, 6))
```

- [x] **Step 2: Add pure tests for C single-step, no-skip, and learning-rate de-duplication**

```python
def test_c_strategy_forces_one_epoch_and_keeps_qwen_learning_rate():
    result = cs._apply_profile_strategy("C", 4, 20, 0.0005, [], "parent-c")
    assert result == (1, 0.0005, False, "C_platform_search:单步续训便于平台反馈归因")


def test_c_resource_policy_never_skips_a_platform_round():
    history = [{
        "id": f"r{i}", "selected_artifact_id": "parent-c",
        "configuration": {
            "parent_artifact_id": "parent-c",
            "loop_evidence": {"complete": True},
            "platform_decision": {"reason": "candidate_not_better"},
            "platform_drift": {"status": "STABLE"},
        },
    } for i in range(1, 4)]
    assert cs._should_skip_training(history, "parent-c", dataset_key="C") is False


def test_c_avoid_repeat_compares_learning_rate_after_epoch_is_fixed():
    history = [{
        "configuration": {
            "parent_artifact_id": "parent-c", "epochs": 9,
            "learning_rate": 0.001,
        }
    }]
    epochs, learning_rate = cs._avoid_repeat(
        "parent-c", 1, 0.001, history, dataset_key="C")
    assert epochs == 1
    assert learning_rate == pytest.approx(0.0005)
```

- [x] **Step 3: Add an integration test proving C trains instead of skipping and passes `--epochs 1`**

```python
def test_train_c_uses_one_epoch_and_never_resource_skips(monkeypatch, tmp_path):
    history = [{
        "id": f"r{i}", "round_index": i,
        "selected_artifact_id": "parent-1",
        "summary": {"metrics": {"rmse": 52.0, "nasa_asymmetric_score": 100.0}},
        "configuration": {
            "parent_artifact_id": "parent-1", "epochs": 5,
            "learning_rate": 0.001,
            "loop_evidence": {"complete": True},
            "platform_decision": {"reason": "candidate_not_better"},
            "platform_drift": {"status": "STABLE"},
        },
    } for i in range(1, 4)]
    rnd = _Round(history, {"rmse": 51.9, "nasa_asymmetric_score": 99.0}, round_index=4)
    rnd.dataset_key = "C"
    rnd.task = {"task": "regression", "metric": "rmse", "metric_direction": "lower"}
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        cand = tmp_path / "cand"
        cand.mkdir()
        (cand / "checkpoint.npz").write_bytes(b"checkpoint")
        (cand / "training.json").write_text(json.dumps({
            "seed": 42,
            "validation_before": {"rmse": 52.0},
            "validation_after": {"rmse": 51.9},
        }), encoding="utf-8")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    monkeypatch.setattr(cs.subprocess, "run", run)
    monkeypatch.setattr(cs, "_decide_promote", lambda *a, **k: (True, {"p": 1.0}))
    cs.train(rnd)

    index = seen["cmd"].index("--epochs")
    assert seen["cmd"][index + 1] == "1"
    assert rnd.finished["configuration"].get("resource_action") is None
```

- [x] **Step 4: Run the new tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_submission_contract.py -k "history_prompt_defaults or c_strategy_forces or c_resource_policy or c_avoid_repeat or train_c_uses" -q
```

Expected: failures because the helpers do not accept `include_all`/`dataset_key`, C does not force one epoch, and C can still resource-skip.

### Task 2: Implement the minimal C platform-feedback policy

**Files:**
- Modify: `submission/contest_submission.py`
- Test: `tests/test_submission_contract.py`

- [x] **Step 1: Let `_history_prompt` optionally retain all history**

Change the signature and iteration only:

```python
def _history_prompt(history, include_all=False):
    lines = []
    records = history if include_all else history[-3:]
    for record in records:
        # existing formatting remains unchanged
```

- [x] **Step 2: Exempt C from `_should_skip_training` without changing A/B**

Change the signature and add the early return:

```python
def _should_skip_training(history, parent_artifact_id, dataset_key=None):
    if str(dataset_key).upper() == "C":
        return False
    # existing A/B resource policy remains unchanged
```

Pass the dataset key from `train()`:

```python
if _should_skip_training(history, rnd.parent.artifact_id,
                         dataset_key=rnd.dataset_key):
```

- [x] **Step 3: Make C de-duplicate learning rates independently of proposed epochs**

Extend `_avoid_repeat` with `dataset_key=None`. For C, normalize the candidate epoch to 1, compare tried configurations by learning rate under the same parent, and choose the first unused bounded multiplier:

```python
def _avoid_repeat(parent_artifact_id, epochs, lr, history, dataset_key=None):
    if lr is None:
        return epochs, lr
    c_search = str(dataset_key).upper() == "C"
    if c_search:
        epochs = 1
    tried = set()
    for record in history:
        config = record.get("configuration") if isinstance(record, dict) else None
        if not isinstance(config, dict) or config.get("parent_artifact_id") != parent_artifact_id:
            continue
        old_epochs, old_lr = config.get("epochs"), config.get("learning_rate")
        if isinstance(old_epochs, int) and _finite_number(old_lr):
            tried.add(float(old_lr) if c_search else (old_epochs, float(old_lr)))
    current = float(lr) if c_search else (epochs, float(lr))
    if current not in tried:
        return epochs, float(lr)
    if c_search:
        alternatives = []
        for power in range(1, 11):
            alternatives.extend((
                float(_clamp(lr / (2 ** power), *LR_RANGE)),
                float(_clamp(lr * (2 ** power), *LR_RANGE)),
            ))
        replacement = next((value for value in alternatives if value not in tried), float(lr))
        return 1, replacement
    alternatives = [
        (epochs, float(_clamp(lr / 2, *LR_RANGE))),
        (epochs, float(_clamp(lr * 2, *LR_RANGE))),
        (max(EPOCH_RANGE[0], epochs // 2), float(lr)),
        (min(EPOCH_RANGE[1], epochs * 2), float(lr)),
    ]
    return next((item for item in alternatives if item not in tried), current)
```

- [x] **Step 4: Force C to one epoch and wire the C prompt/fallback**

Add this first branch to `_apply_profile_strategy`:

```python
if profile == "C":
    return 1, lr, False, "C_platform_search:单步续训便于平台反馈归因"
```

In `_ask_model`, use all C history, request only a learning rate for C, default C to one epoch and `0.001`, pass `dataset_key` into `_avoid_repeat`, and retain the existing A/B prompt and parsing:

```python
c_search = str(rnd.dataset_key).upper() == "C"
hist = _history_prompt(history, include_all=c_search)
if c_search:
    request = '{"learning_rate": 0到0.1小数, "hypothesis": "一句话理由"}'
    epochs, lr = 1, 0.001
else:
    request = '{"epochs": 整数1-50, "learning_rate": 0到0.1小数, "hypothesis": "一句话理由"}'
    epochs, lr = 5, None
# build the existing prompt using request
```

After parsing:

```python
if not c_search:
    epochs = int(_clamp(int(data["epochs"]), *EPOCH_RANGE))
lr = float(_clamp(float(data["learning_rate"]), *LR_RANGE))
```

Pass the profile into repeat protection:

```python
adjusted = _avoid_repeat(
    rnd.parent.artifact_id, epochs, lr, history, dataset_key=rnd.dataset_key)
```

The existing `_apply_profile_strategy()` call then fixes C to one epoch and appends its reason.

- [x] **Step 5: Run the focused tests**

Run:

```powershell
python -m pytest tests/test_submission_contract.py -k "history_prompt or resource_skip or avoid_repeat or profile_strategy or train_c_uses" -q
```

Expected: all selected tests pass.

- [x] **Step 6: Run the complete submission contract suite**

Run:

```powershell
python -m pytest tests/test_submission_contract.py -q
```

Expected: all tests pass.

- [x] **Step 7: Commit the implementation**

```powershell
git add -- submission/contest_submission.py tests/test_submission_contract.py
git commit -m "feat: drive C search with platform feedback"
```

### Task 3: Document, regress, and build the formal image

**Files:**
- Modify: `作品文档.md`
- Modify: `RUNBOOK.md`
- Verify: `submission/Dockerfile`

- [x] **Step 1: Document the two-layer C feedback loop**

Add this content to the existing C strategy section in `作品文档.md`:

```markdown
- **C 平台反馈搜索**：每轮固定续训 1 个 epoch，Qwen 读取截至当前轮的全部压缩平台公开验证历史并提出下一学习率；同一父产物不重复学习率，C 不跳过训练轮次。SDK 每轮返回的是公开验证摘要，不是隐藏测试指标。隐藏集效果只在整次试运行报告中验收。
```

- [x] **Step 2: Add the post-trial C acceptance check to `RUNBOOK.md`**

Add an acceptance block near the submission verification commands:

```markdown
### C 隐藏集验收

SDK 运行中的 `summary.metrics` 是平台公开验证摘要。新镜像试运行完成后，必须在平台报告中检查 C 隐藏测试：RMSE 应低于第一次试运行的 `56.455008`；NASA 非对称分应低于 `96,699,827,373.39221`，至少不得出现数量级恶化。未拿到新平台报告前，只能确认工程策略已生效，不能宣称隐藏集性能已经提升。
```

- [x] **Step 3: Run all relevant tests**

Run:

```powershell
python -m pytest tests/test_submission_contract.py tests/test_submission_baseline_optimization.py tests/test_objective_score_proxy.py -q
python -m pytest contest-harness-sdk-1.0.2/tests -q
```

Expected: project suites pass; SDK live tests may be skipped when live platform variables are absent, with no failures.

- [x] **Step 4: Build the formal SDK image**

Run:

```powershell
docker build -f submission/Dockerfile -t zhixun-submit:latest .
```

Expected: build completes successfully.

- [x] **Step 5: Verify the image contract**

Run the repository's existing Docker `describe` command from `RUNBOOK.md` against `zhixun-submit:latest`.

Expected: protocol `contest.harness.workspace.v2`, datasets A/B/C, and no nonzero exit code.

- [x] **Step 6: Commit documentation**

```powershell
git add -- '作品文档.md' RUNBOOK.md
git commit -m "docs: explain C platform feedback acceptance"
```

- [ ] **Step 7: Record the only external acceptance gate**

Submit the rebuilt image as a platform trial. Compare the new C hidden metrics with the first report:

```text
RMSE < 56.455008
NASA asymmetric score < 96,699,827,373.39221
```

Expected: RMSE passes the strict goal. If it does not, retain the new report as the next cross-submission observation and do not claim the C performance issue is solved.
