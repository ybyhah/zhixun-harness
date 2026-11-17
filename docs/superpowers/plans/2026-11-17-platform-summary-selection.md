# Platform Summary Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the formal SDK submission select artifacts from authoritative platform summaries, feed platform history to Qwen, avoid duplicate experiments, and align Docker and RunBook with the v2 protocol.

**Architecture:** Keep all formal-submission behavior in `submission/contest_submission.py`. Add small pure helpers for extracting platform metrics, reading parent history, choosing a candidate, and avoiding repeated hyperparameters; cover them with focused tests before wiring them into `train()`. Preserve local training metrics only as diagnostics.

**Tech Stack:** Python 3.12, NumPy 1.26, contest-harness SDK 1.0.2, pytest, Docker.

---

### Task 1: Platform metric extraction and artifact selection

**Files:**
- Create: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] **Step 1: Write failing tests for authoritative platform selection**

Add tests that import `submission/contest_submission.py` and assert:

```python
def test_first_round_accepts_valid_platform_candidate_without_parent_metric():
    selected, detail = cs._select_from_platform(
        task="classification", metric="macro_f1", direction="higher",
        candidate={"macro_f1": 0.8}, parent={}, first_round=True)
    assert selected is True
    assert detail["source"] == "platform"

def test_later_round_rejects_candidate_not_better_than_parent():
    selected, _ = cs._select_from_platform(
        task="regression", metric="rmse", direction="lower",
        candidate={"rmse": 10.1}, parent={"rmse": 10.0}, first_round=False)
    assert selected is False

def test_missing_candidate_metric_is_conservative():
    selected, detail = cs._select_from_platform(
        task="classification", metric="macro_f1", direction="higher",
        candidate={}, parent={"macro_f1": 0.7}, first_round=False)
    assert selected is False
    assert detail["reason"] == "candidate_metric_missing"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_submission_contract.py -q
```

Expected: failures because `_select_from_platform` and metric helpers do not exist.

- [ ] **Step 3: Implement minimal pure helpers**

Add helpers that:

```python
def _summary_metrics(value): ...
def _metric_value(metrics, name): ...
def _select_from_platform(*, task, metric, direction, candidate, parent, first_round): ...
```

They accept only finite numeric metrics, use `higher`/`lower` direction, accept a valid first-round candidate without a parent metric, and conservatively retain the parent in later rounds when the authoritative parent metric is unavailable.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same targeted pytest command. Expected: all Task 1 tests pass.

### Task 2: Platform history parsing and duplicate experiment avoidance

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] **Step 1: Write failing history tests**

Cover nested platform records and repeated configurations:

```python
history = [{
    "id": "r1", "round_index": 1, "selected_artifact_id": "parent-1",
    "summary": {"metrics": {"rmse": 9.5}},
    "configuration": {"epochs": 5, "learning_rate": 0.001,
                      "parent_artifact_id": "parent-0"},
}]
assert cs._history_metrics(history[0]) == {"rmse": 9.5}
assert "rmse" in cs._history_prompt(history)
assert cs._avoid_repeat("parent-0", 5, 0.001, history) != (5, 0.001)
```

- [ ] **Step 2: Verify RED**

Run the targeted test file and confirm missing helper failures.

- [ ] **Step 3: Implement history helpers**

Add:

```python
def _history_metrics(record): ...
def _parent_metrics(rnd, history): ...
def _history_prompt(history): ...
def _avoid_repeat(parent_artifact_id, epochs, lr, history): ...
```

Read `summary.metrics`, direct `metrics`, and `configuration.platform_metrics`. Match the current parent artifact when possible. Try deterministic alternatives in this order: halve learning rate, double learning rate, halve epochs, double epochs, all within existing bounds.

- [ ] **Step 4: Verify GREEN**

Run targeted tests and confirm all pass.

### Task 3: Wire platform summaries into the train round

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `scripts/local_sdk_test.py`
- Modify: `submission/contest_submission.py`

- [ ] **Step 1: Write failing round integration tests**

Use a fake round whose `submit_candidate()` returns `{"artifact_id": "cand", "summary": {"metrics": ...}}`. Assert:

- a local diagnostic improvement cannot promote when the platform metric is worse;
- a platform improvement promotes even if local diagnostic data would reject;
- final `configuration.platform_metrics` equals the returned platform summary;
- final `reason` names the platform metric and values;
- the next-round model prompt contains historical platform metrics.

- [ ] **Step 2: Verify RED**

Run the integration tests and confirm current code selects from `_decide_promote()` instead.

- [ ] **Step 3: Implement the minimal train wiring**

Keep `_decide_promote()` only as `local_diagnostic`. After `submit_candidate()`:

```python
candidate_metrics = _summary_metrics(submitted)
parent_metrics = _parent_metrics(rnd, history)
promote, platform_decision = _select_from_platform(...)
finish_configuration = {
    **configuration,
    "parent_artifact_id": rnd.parent.artifact_id,
    "platform_metrics": candidate_metrics,
    "parent_platform_metrics": parent_metrics,
    "platform_decision": platform_decision,
    "local_diagnostic": local_diagnostic,
}
```

Use the candidate only when `promote` is true. Build `reason` exclusively around the platform decision and identify local values as diagnostics.

- [ ] **Step 4: Improve the model request contract**

Use the platform history prompt, apply `_avoid_repeat()`, set `response_format={"type": "json_object"}`, and cap output at `min(256, rnd.max_output_tokens)`.

- [ ] **Step 5: Verify GREEN**

Run targeted tests and `python scripts/local_sdk_test.py`. Both must pass.

### Task 4: C-task stability veto

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] **Step 1: Write a failing C stability test**

Assert that a C regression candidate with better RMSE but worse platform NASA metric is rejected when both candidate and parent summaries contain the NASA metric.

- [ ] **Step 2: Verify RED**

Run only the C stability test and confirm the candidate currently promotes.

- [ ] **Step 3: Add the minimal veto**

Recognize normalized metric keys containing `nasa` and treat lower as better. Apply the veto only when both platform summaries contain finite NASA values. Do not substitute local metrics.

- [ ] **Step 4: Verify GREEN**

Run the targeted test file.

### Task 5: Formal Docker image and RunBook alignment

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/Dockerfile`
- Modify: `Dockerfile`
- Modify: `RUNBOOK.md`

- [ ] **Step 1: Write failing static packaging tests**

Assert that `submission/Dockerfile` copies `submission/decision.py`, uses `/app/harness.py`, and that the RunBook's primary build command contains `-f submission/Dockerfile`.

- [ ] **Step 2: Verify RED**

Run static tests and observe the missing copy and old build command failures.

- [ ] **Step 3: Apply surgical packaging changes**

Add:

```dockerfile
COPY submission/decision.py /app/decision.py
```

Mark the root Dockerfile as a local-development image in its header. Rewrite the first RunBook build/run section for the formal SDK image and keep the old local flow in a clearly labeled later section.

- [ ] **Step 4: Verify GREEN**

Run static tests and the SDK `describe` command with the submission and SDK directories on `PYTHONPATH`.

### Task 6: Full verification and Docker inspection

**Files:**
- No new production files.

- [ ] **Step 1: Run submission-focused tests**

```powershell
python -m pytest tests/test_submission_contract.py -q
python scripts/local_sdk_test.py
```

- [ ] **Step 2: Run project regression tests**

```powershell
python -m pytest tests -q
```

Record pre-existing failures separately; do not hide them.

- [ ] **Step 3: Start Docker Desktop and build the formal image**

Start the installed Docker Desktop application hidden if the daemon is not running, wait in bounded polls, then run:

```powershell
docker build --file submission/Dockerfile --tag zhixun-submit:test .
```

- [ ] **Step 4: Inspect and smoke-test the image**

Verify `decision.py` exists, no `.env` or `.npz` is present, the container runs as non-root, and `/app/harness.py describe` returns protocol v2.

- [ ] **Step 5: State the external platform boundary**

The repository cannot create a real host-gateway `call_id`. Cite the prior platform report as evidence that Qwen calls work, and require one new platform submission after these changes to validate the revised artifact-selection path.

