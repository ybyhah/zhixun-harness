# P1 Objective Score Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The repository rules prohibit subagents, so execution stays in the current primary agent.

**Goal:** Improve early-stage performance, stagnation recovery, C-package safety, and checkpoint stability without weakening platform-authoritative artifact selection.

**Architecture:** Keep within-run model optimization in `submission/baselines`; keep cross-round policy and platform artifact selection in `submission/contest_submission.py`. Every cross-artifact decision uses platform `summary.metrics`; local validation only selects the best epoch inside one candidate training run.

**Tech Stack:** Python 3.12, NumPy 1.26, contest-harness SDK 1.0.2, pytest, Docker.

---

### Task 1: Save the best epoch in each training run

**Files:**
- Create: `tests/test_submission_baseline_optimization.py`
- Modify: `submission/baselines/train.py`

- [ ] **Step 1: Write failing tests**

Add a unit test around a small pure comparison helper and an integration test that replaces `Model` with a deterministic fake whose validation scores improve and then regress. Assert that `checkpoint.npz`, `validation_after`, `selected_epoch`, and `best_validation` correspond to the best intermediate epoch while `history.jsonl` still contains every requested epoch.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_submission_baseline_optimization.py -q
```

Expected: FAIL because best-epoch selection and report fields do not exist.

- [ ] **Step 3: Implement minimum best-epoch selection**

In `train_run()`, deep-copy the model whenever the current epoch's primary validation metric is strictly better than the prior best. After all requested epochs, write the copied best model. Set:

```python
report["selected_epoch"] = best_model.meta["epochs"]
report["best_validation"] = best_validation
report["trained_final_epochs"] = trained_final_epochs
report["validation_after"] = best_validation
```

- [ ] **Step 4: Verify GREEN and commit**

Run the targeted test, then commit only the test and `train.py`.

### Task 2: Apply the A early-round guardrail

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] **Step 1: Write a failing policy test**

Assert `_apply_profile_strategy("A", round_index, epochs, lr, history)` returns `(10, 0.01, reason)` for rounds 1 and 2, and returns the Qwen proposal unchanged for round 3.

- [ ] **Step 2: Verify RED**

Run the three new test cases and confirm failure because the helper is absent.

- [ ] **Step 3: Implement and wire the helper**

Call the helper after Qwen JSON parsing and duplicate avoidance. Append its nonempty reason to `hypothesis` so the audit trail explains the override.

- [ ] **Step 4: Verify GREEN and commit**

Run `tests/test_submission_contract.py`, then commit the two files.

### Task 3: Switch B strategy after one stagnant round

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] **Step 1: Write failing stagnation tests**

Create a B history record whose `configuration.platform_decision.reason` is `candidate_not_better`. Assert the next configuration is the first current-parent-untried entry from `(5, .001)`, `(3, .0005)`, `(8, .0003)`. Also assert an improving previous round does not override Qwen.

- [ ] **Step 2: Verify RED**

Run the new cases and confirm the proposed Qwen configuration is still returned.

- [ ] **Step 3: Implement the deterministic switch**

Extend `_apply_profile_strategy()` with B-only stagnation detection and the three-entry sequence. Reuse `parent_artifact_id` and existing history configuration fields; add no new configuration source.

- [ ] **Step 4: Verify GREEN and commit**

Run the contract tests and commit.

### Task 4: Add C asymmetric overestimation loss and NASA diagnostics

**Files:**
- Modify: `tests/test_submission_baseline_optimization.py`
- Modify: `submission/baselines/model.py`
- Modify: `submission/baselines/train.py`

- [ ] **Step 1: Write failing numerical tests**

Assert the regression loss-weight helper returns `1.5` for positive C prediction error, `1.0` for negative C error, and all ones for B. Assert the NASA helper penalizes equal-magnitude overestimation more than underestimation.

- [ ] **Step 2: Verify RED**

Run the new numerical tests and confirm the helpers are absent.

- [ ] **Step 3: Implement the minimum loss change**

Set `model.meta["overestimate_weight"] = 1.5` only for profile C before training. In regression `fit()`, multiply squared error and its derivative by `np.where(error > 0, weight, 1.0)`. Treat a missing checkpoint field as `1.0`, validate a present field as finite and at least `1.0`, and include it when C is resumed. Add `nasa_asymmetric_score` to C validation metrics using the standard positive-error divisor 10 and negative-error divisor 13.

- [ ] **Step 4: Verify GREEN and commit**

Run baseline optimization tests plus existing model/metric tests, then commit.

### Task 5: Select an allowed historical best artifact

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] **Step 1: Write failing history-selection tests**

Assert a history artifact with a strictly better platform main metric is selected only when its ID appears in `rnd.artifacts`. Assert an unavailable artifact is ignored. Assert C rejects a historical artifact whose NASA score is worse than the current parent.

- [ ] **Step 2: Verify RED**

Run the new cases and confirm no history-selection helper exists.

- [ ] **Step 3: Implement platform-only ranking**

Add a pure `_best_allowed_history()` helper that returns an artifact ID, platform metrics, and history record ID. After candidate gating, rank eligible candidate, parent, and history best by the main metric; apply the existing C NASA veto relative to the parent. Record `selected_source` and `selected_history_id` in the final configuration and reason.

- [ ] **Step 4: Verify GREEN and commit**

Run all contract tests and commit.

### Task 6: Documentation and complete verification

**Files:**
- Modify: `作品文档.md`
- Modify: `RUNBOOK.md`

- [ ] **Step 1: Update only the implemented behavior**

Document best-epoch candidate generation, A/B profile guardrails, C asymmetric loss, and platform-authorized history recovery. Keep SwanLab and second-platform-run status explicitly incomplete.

- [ ] **Step 2: Run fresh verification**

```powershell
python -m pytest tests/test_submission_baseline_optimization.py tests/test_submission_contract.py -q
python scripts/local_sdk_test.py
$env:PYTHONUTF8='1'; python -m pytest contest-harness-sdk-1.0.2/tests -q; Remove-Item Env:PYTHONUTF8
python -m py_compile submission/contest_submission.py submission/baselines/model.py submission/baselines/train.py
docker buildx build --platform linux/amd64 --provenance=false --sbom=false --load -t zhixun-submit -f submission/Dockerfile .
docker run --rm zhixun-submit describe
```

Expected: all targeted tests and SDK tests pass, Docker build exits 0, and `describe` reports `contest.harness.workspace.v2`.

- [ ] **Step 3: Run full project tests and classify unrelated failures**

Run `python -m pytest tests -q --tb=short`. Do not change unrelated legacy modules merely to make this P1 branch green; report pre-existing failures separately.

- [ ] **Step 4: Commit documentation and hand off platform checks**

The remaining platform-only evidence is a new real Qwen `call_id`, disappearance of `pred_failed:ModuleNotFoundError`, populated platform history, and objective score movement in the second submission report.

