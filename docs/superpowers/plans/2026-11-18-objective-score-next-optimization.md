# Objective Score Next Optimization Implementation Plan

> **For agentic workers:** Execute inline in the current primary agent. Repository rules prohibit subagents. Follow TDD for every behavior change.

**Goal:** Improve the four objective-score dimensions while keeping platform metrics authoritative and using the existing official static baselines.

**Architecture:** Keep within-run epoch selection and optimizer controls in `submission/baselines`; keep cross-round policy and resource decisions in `submission/contest_submission.py`; add one standard-library reporting script for local proxy metrics.

**Tech Stack:** Python 3.12, NumPy, contest-harness SDK 1.0.2, pytest, Docker.

---

### Task 1: C multi-objective epoch selection and early stopping

**Files:**
- Modify: `tests/test_submission_baseline_optimization.py`
- Modify: `submission/baselines/train.py`

- [ ] Add failing tests for C RMSE/NASA tolerance, official weight 1.0, and patience-3 early stop.
- [ ] Run the targeted tests and verify expected failures.
- [ ] Implement the minimum profile-aware comparator, remove forced weight 1.5, and stop C after three non-improving epochs.
- [ ] Run the targeted tests until green.

### Task 2: Platform stability gates for A and C

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] Add failing tests for A minimum gain/Accuracy floor and C 5% NASA tolerance.
- [ ] Verify RED.
- [ ] Implement the platform-only guardrails.
- [ ] Verify GREEN.

### Task 3: B optimizer-moment reset

**Files:**
- Modify: `tests/test_submission_baseline_optimization.py`
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/baselines/train.py`
- Modify: `submission/contest_submission.py`

- [ ] Add failing policy and training-report tests.
- [ ] Verify RED.
- [ ] Carry a reset flag only for B stagnation switches and clear first/second moments before fitting.
- [ ] Verify GREEN.

### Task 4: Qwen and training resource controls

**Files:**
- Modify: `tests/test_submission_contract.py`
- Modify: `submission/contest_submission.py`

- [ ] Add failing tests for last-three history, 96-token non-thinking calls, and non-consecutive cooldown skip after three completed loops.
- [ ] Verify RED.
- [ ] Implement the minimum prompt and skip-policy changes.
- [ ] Verify GREEN.

### Task 5: Objective-score proxy

**Files:**
- Create: `scripts/objective_score_proxy.py`
- Create: `tests/test_objective_score_proxy.py`

- [ ] Add failing tests for baseline loading and classification/regression direction handling.
- [ ] Verify RED.
- [ ] Implement CSV summarization and CSV/Markdown outputs.
- [ ] Generate the current proxy report under `reports/objective_score_proxy_2026-11-18/`.
- [ ] Verify GREEN.

### Task 6: Documentation and full verification

**Files:**
- Modify: `RUNBOOK.md`

- [ ] Document the implemented guardrails, static-baseline source, proxy command, and platform-only validation items.
- [ ] Run targeted tests, full project tests, SDK tests, `py_compile`, local SDK test, Docker build, and `describe`.
- [ ] Classify any unrelated or platform-only failures without weakening tests.
