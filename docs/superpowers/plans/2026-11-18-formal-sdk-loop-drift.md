# Formal SDK Loop and Drift Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Repository rules prohibit subagents, so the primary agent executes every step inline.

**Goal:** Make complete-loop and drift evidence native to the formal workspace-v2 submission without adding runtime dependencies.

**Architecture:** Add small pure helpers to `submission/contest_submission.py` for platform-metric drift and SDK loop evidence. Persist both structures in `finish(configuration=...)`, feed prior drift evidence to Qwen, and re-scope legacy local-loop tests so they no longer represent formal submission acceptance.

**Tech Stack:** Python 3.12, contest-harness SDK 1.0.2, NumPy 1.26, pytest, Docker.

---

### Task 1: Platform metric drift detector

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] Write failing tests for higher-is-better degradation, lower-is-better degradation, stable/improved metrics, missing metrics, and C NASA worsening.
- [ ] Run the new tests and confirm failure because `_platform_metric_drift` does not exist.
- [ ] Implement `_platform_metric_drift(metric, direction, candidate, parent, dataset_key=None)` with a fixed 2% relative degradation threshold and no external dependency.
- [ ] Add prior `configuration.platform_drift` status/reason/gain to `_history_prompt()` and test the prompt.
- [ ] Run all submission contract tests and commit.

### Task 2: Structured formal-loop evidence

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`
- Modify: `scripts/local_sdk_test.py`

- [ ] Write failing tests for `_loop_evidence`, `_complete_sdk_loop_count`, three completed formal history records, successful training configuration, and training-failure configuration.
- [ ] Run the new tests and verify the missing evidence fields cause failure.
- [ ] Implement evidence helpers with explicit `history_read`, `model_call_ids`, `experiment_status`, `decision_source`, and `complete` fields.
- [ ] Add complete evidence to both successful and failed `rnd.finish()` paths; add `platform_drift` to the successful path.
- [ ] Run contract and local SDK tests and commit.

### Task 3: Re-scope legacy local tests

**Files:**
- Modify: `tests/test_loop.py`
- Modify: `tests/test_monitor.py`

- [ ] Change the local-loop test name and assertion so it verifies that the simulator records triggered experimental loops without claiming the formal SDK three-loop requirement.
- [ ] Make the optional ADWIN test skip when `ErrorStreamMonitor.kind == "none"`; retain the sudden-change assertion when `frouros` is installed.
- [ ] Run the two tests, then the complete test suite, and commit.

### Task 4: Documentation and release verification

**Files:**
- Modify: `RUNBOOK.md`
- Modify: `作品文档.md`

- [ ] Document `loop_evidence` and dependency-free `platform_metric_delta`; state that it is performance-drift evidence rather than PSI.
- [ ] Run submission/baseline tests, local SDK test, official SDK tests with UTF-8 mode, Python compilation, and `git diff --check`.
- [ ] Build `submission/Dockerfile`, run `describe`, and inspect the new helpers inside the non-root container.
- [ ] Run the full suite and report any remaining unrelated failure separately.
- [ ] Commit documentation only after verification.

