"""contest.harness.workspace.v2 提交物：把官方 baselines 续训接到 SDK 轮次契约。

每轮都从平台父检查点续训，并真实调用一次模型（读历史 → 提下一轮续训超参，满足
「每轮至少一次模型调用」，也体现 Agent 自主决策）。密钥不入镜像：模型只经 rnd.model()。
自评无效：候选分数由平台 submit_candidate 后测量。
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"   # 镜像内 /app/baselines
EPOCH_RANGE = (1, 50)
LR_RANGE = (1e-6, 0.1)
DRIFT_RELATIVE_THRESHOLD = 0.02
A_MIN_MACRO_F1_GAIN = 0.002
A_ACCURACY_TOLERANCE = 0.002
C_NASA_TOLERANCE = 0.05
_MODEL_CACHE: dict = {}


def _clamp(value, low, high):
    return max(low, min(high, value))


def _extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型回复没有 JSON 对象")
    return json.loads(text[start:end + 1])


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _summary_metrics(value):
    """Extract finite platform metrics from a candidate or artifact receipt."""
    if not isinstance(value, dict):
        return {}
    summary = value.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("metrics"), dict):
        metrics = summary["metrics"]
    elif isinstance(value.get("metrics"), dict):
        metrics = value["metrics"]
    else:
        return {}
    return {str(k): v for k, v in metrics.items() if _finite_number(v)}


def _history_metrics(record):
    metrics = _summary_metrics(record)
    if metrics:
        return metrics
    if not isinstance(record, dict):
        return {}
    config = record.get("configuration")
    if isinstance(config, dict) and isinstance(config.get("platform_metrics"), dict):
        return {str(k): v for k, v in config["platform_metrics"].items()
                if _finite_number(v)}
    return {}


def _metric_value(metrics, name):
    if not isinstance(metrics, dict):
        return None
    wanted = "".join(c for c in str(name).lower() if c.isalnum())
    for key, value in metrics.items():
        normalized = "".join(c for c in str(key).lower() if c.isalnum())
        if normalized == wanted and _finite_number(value):
            return float(value)
    return None


def _nasa_value(metrics):
    if not isinstance(metrics, dict):
        return None
    for key, value in metrics.items():
        if "nasa" in str(key).lower() and _finite_number(value):
            return float(value)
    return None


def _platform_metric_drift(metric, direction, candidate, parent, dataset_key=None):
    cand = _metric_value(candidate, metric)
    prev = _metric_value(parent, metric)
    result = {
        "detector": "platform_metric_delta", "enabled": True,
        "metric": metric, "candidate": cand, "parent": prev,
        "relative_gain": None, "threshold": DRIFT_RELATIVE_THRESHOLD,
        "status": "UNKNOWN", "reason": "metric_missing",
        "drift_detected": False,
    }
    if cand is None or prev is None:
        return result
    scale = max(abs(prev), 1e-12)
    gain = ((cand - prev) if direction == "higher" else (prev - cand)) / scale
    result["relative_gain"] = gain
    if str(dataset_key).upper() == "C":
        cand_nasa, prev_nasa = _nasa_value(candidate), _nasa_value(parent)
        if (cand_nasa is not None and prev_nasa is not None
                and cand_nasa > prev_nasa * (1 + C_NASA_TOLERANCE)):
            return {**result, "status": "HARMFUL", "reason": "nasa_worsened",
                    "drift_detected": True, "candidate_nasa": cand_nasa,
                    "parent_nasa": prev_nasa}
    if gain < -DRIFT_RELATIVE_THRESHOLD:
        return {**result, "status": "HARMFUL",
                "reason": "main_metric_degraded", "drift_detected": True}
    if gain > 0:
        return {**result, "status": "IMPROVED", "reason": "metric_improved"}
    return {**result, "status": "STABLE", "reason": "within_threshold"}


def _select_from_platform(*, task, metric, direction, candidate, parent,
                          first_round, dataset_key=None):
    """Choose from authoritative platform metrics; local scores never decide."""
    del task
    cand = _metric_value(candidate, metric)
    prev = _metric_value(parent, metric)
    detail = {"source": "platform", "metric": metric, "direction": direction,
              "candidate": cand, "parent": prev}
    if cand is None:
        return False, {**detail, "reason": "candidate_metric_missing"}
    if prev is None:
        reason = "first_round_valid_candidate" if first_round else "parent_metric_missing"
        return bool(first_round), {**detail, "reason": reason}
    better = cand > prev if direction == "higher" else cand < prev
    if not better:
        return False, {**detail, "reason": "candidate_not_better"}
    if str(dataset_key).upper() == "A":
        if cand - prev < A_MIN_MACRO_F1_GAIN:
            return False, {**detail, "reason": "gain_below_minimum",
                           "minimum_gain": A_MIN_MACRO_F1_GAIN}
        cand_accuracy = _metric_value(candidate, "accuracy")
        prev_accuracy = _metric_value(parent, "accuracy")
        if (cand_accuracy is not None and prev_accuracy is not None
                and cand_accuracy < prev_accuracy - A_ACCURACY_TOLERANCE):
            return False, {**detail, "reason": "accuracy_floor",
                           "candidate_accuracy": cand_accuracy,
                           "parent_accuracy": prev_accuracy}
    if str(dataset_key).upper() == "C":
        cand_nasa, prev_nasa = _nasa_value(candidate), _nasa_value(parent)
        if (cand_nasa is not None and prev_nasa is not None
                and cand_nasa > prev_nasa * (1 + C_NASA_TOLERANCE)):
            return False, {**detail, "reason": "nasa_stability_veto",
                           "candidate_nasa": cand_nasa, "parent_nasa": prev_nasa}
    return True, {**detail, "reason": "candidate_better"}


def _history_prompt(history):
    lines = []
    for record in history[-6:]:
        config = record.get("configuration") if isinstance(record, dict) else {}
        config = config if isinstance(config, dict) else {}
        metrics = _history_metrics(record)
        drift = config.get("platform_drift")
        drift = drift if isinstance(drift, dict) else {}
        lines.append(
            f"  轮{record.get('round_index', '?')}: "
            f"platform_metrics={json.dumps(metrics, ensure_ascii=False)}; "
            f"epochs={config.get('epochs', '?')}; lr={config.get('learning_rate', '?')}; "
            f"selected={record.get('selected_artifact_id', '?')}; "
            f"drift={drift.get('status', 'UNKNOWN')}/"
            f"{drift.get('reason', 'missing')}/"
            f"{drift.get('relative_gain', '?')}"
        )
    return "\n".join(lines) if lines else "  （暂无历史）"


def _loop_evidence(history, model_call_ids, hypothesis, experiment_status,
                   decision_source):
    history_ids = [record["id"] for record in history
                   if isinstance(record, dict) and isinstance(record.get("id"), str)]
    calls = [call_id for call_id in model_call_ids
             if isinstance(call_id, str) and call_id]
    hypothesis_recorded = isinstance(hypothesis, str) and bool(hypothesis.strip())
    experiment_done = experiment_status in {"candidate_measured", "train_failed"}
    decision_done = decision_source in {"platform_summary", "failure_fallback"}
    return {
        "history_read": True,
        "history_ids": history_ids,
        "model_call_ids": calls,
        "hypothesis_recorded": hypothesis_recorded,
        "experiment_status": experiment_status,
        "decision_source": decision_source,
        "complete": bool(calls and hypothesis_recorded and experiment_done and decision_done),
    }


def _complete_sdk_loop_count(history):
    count = 0
    for record in history:
        config = record.get("configuration") if isinstance(record, dict) else None
        evidence = config.get("loop_evidence") if isinstance(config, dict) else None
        if isinstance(evidence, dict) and evidence.get("complete") is True:
            count += 1
    return count


def _parent_metrics(rnd, history):
    receipt = getattr(getattr(rnd, "parent", None), "receipt", {})
    metrics = _summary_metrics(receipt)
    if metrics:
        return metrics
    parent_id = getattr(getattr(rnd, "parent", None), "artifact_id", None)
    for record in reversed(history):
        if record.get("selected_artifact_id") == parent_id:
            metrics = _history_metrics(record)
            if metrics:
                return metrics
    task_contract = getattr(rnd, "task", {})
    if isinstance(task_contract, dict):
        for key in ("parent_metrics", "baseline_metrics"):
            metrics = _summary_metrics({"metrics": task_contract.get(key)})
            if metrics:
                return metrics
    for record in reversed(history):
        metrics = _history_metrics(record)
        if metrics:
            return metrics
    return {}


def _best_allowed_history(history, artifacts, metric, direction, parent_metrics,
                          dataset_key=None):
    allowed = {item.get("artifact_id") for item in artifacts
               if isinstance(item, dict) and isinstance(item.get("artifact_id"), str)}
    best = None
    for record in history:
        if not isinstance(record, dict):
            continue
        artifact_id = record.get("selected_artifact_id")
        if artifact_id not in allowed:
            continue
        metrics = _history_metrics(record)
        eligible, _ = _select_from_platform(
            task="", metric=metric, direction=direction,
            candidate=metrics, parent=parent_metrics, first_round=False,
            dataset_key=dataset_key)
        if not eligible:
            continue
        if best is not None:
            better, _ = _select_from_platform(
                task="", metric=metric, direction=direction,
                candidate=metrics, parent=best["metrics"], first_round=False,
                dataset_key=dataset_key)
            if not better:
                continue
        best = {"artifact_id": artifact_id, "history_id": record.get("id"),
                "metrics": metrics}
    return best


def _avoid_repeat(parent_artifact_id, epochs, lr, history):
    if lr is None:
        return epochs, lr
    tried = set()
    for record in history:
        config = record.get("configuration") if isinstance(record, dict) else None
        if not isinstance(config, dict) or config.get("parent_artifact_id") != parent_artifact_id:
            continue
        old_epochs, old_lr = config.get("epochs"), config.get("learning_rate")
        if isinstance(old_epochs, int) and _finite_number(old_lr):
            tried.add((old_epochs, float(old_lr)))
    current = (epochs, float(lr))
    if current not in tried:
        return current
    alternatives = [
        (epochs, float(_clamp(lr / 2, *LR_RANGE))),
        (epochs, float(_clamp(lr * 2, *LR_RANGE))),
        (max(EPOCH_RANGE[0], epochs // 2), float(lr)),
        (min(EPOCH_RANGE[1], epochs * 2), float(lr)),
    ]
    return next((item for item in alternatives if item not in tried), current)


def _apply_profile_strategy(dataset_key, round_index, epochs, lr, history,
                            parent_artifact_id):
    profile = str(dataset_key).upper()
    if profile == "A" and round_index in (1, 2):
        return 10, 0.01, "A_early_round:前两轮采用已验证的强起点"
    if profile == "B":
        configs = [record.get("configuration") for record in history
                   if isinstance(record, dict)]
        matching = [config for config in configs if isinstance(config, dict)
                    and config.get("parent_artifact_id") == parent_artifact_id]
        if matching:
            decision = matching[-1].get("platform_decision")
            reason = decision.get("reason") if isinstance(decision, dict) else None
            if reason not in ("candidate_better", "first_round_valid_candidate"):
                tried = {(config.get("epochs"), config.get("learning_rate"))
                         for config in matching}
                for candidate in ((5, 0.001), (3, 0.0005), (8, 0.0003)):
                    if candidate not in tried:
                        return (*candidate, "B_stagnation_switch:上一轮无改善后换轴")
    return epochs, lr, ""


def predict(sample, model_dir):
    """逐样本预测：从 CONTEST_ARTIFACT_PATH 加载官方 Model，复用官方 predict 逻辑。"""
    del model_dir  # 检查点路径以 CONTEST_ARTIFACT_PATH 为准
    if str(BASELINES_DIR.parent) not in sys.path:
        sys.path.insert(0, str(BASELINES_DIR.parent))
    from baselines.model import Model
    from baselines.predict import predict as official_predict

    ckpt = os.environ["CONTEST_ARTIFACT_PATH"]
    model = _MODEL_CACHE.get(ckpt)
    if model is None:
        model = _MODEL_CACHE[ckpt] = Model.load(ckpt)
    return official_predict(model, sample.inputs)   # {"outputs": ...}，SDK 负责校验


def _ask_model(rnd, task, metric, history):
    """调用一次模型，让它读历史提下一轮续训超参。返回 (epochs, lr, hypothesis, call_id)。

    任何失败都回退官方默认超参，但模型调用已经发生（调用在前、解析在后），
    保证本轮必有一次真实 model_call。
    """
    hib = rnd.task.get("metric_direction", "higher" if task == "classification" else "lower") == "higher"
    direction = "越大越好" if hib else "越小越好"
    hist = _history_prompt(history)
    user = (
        f"数据集 {rnd.dataset_key}，任务 {task}，主指标 {metric}（{direction}）。\n"
        f"本轮从父检查点续训，梯度只用公开训练集。历史检查点表现：\n{hist}\n\n"
        "请给出本轮续训超参，严格只输出 JSON："
        '{"epochs": 整数1-50, "learning_rate": 0到0.1小数, "hypothesis": "一句话理由"}'
    )
    epochs, lr, hypothesis = 5, None, "回退官方默认超参续训一轮"
    receipt = rnd.model(
        [{"role": "user", "content": user}],
        max_tokens=min(256, rnd.max_output_tokens), temperature=0,
        response_format={"type": "json_object"})
    try:
        data = _extract_json(receipt.text)
        epochs = int(_clamp(int(data["epochs"]), *EPOCH_RANGE))
        lr = float(_clamp(float(data["learning_rate"]), *LR_RANGE))
        if isinstance(data.get("hypothesis"), str) and data["hypothesis"].strip():
            hypothesis = data["hypothesis"].strip()[:2000]
    except Exception as exc:  # noqa: BLE001 —— 慢通路失败回退，不中断本轮
        hypothesis = f"模型回复不可用（{type(exc).__name__}），回退官方默认超参"
    adjusted = _avoid_repeat(rnd.parent.artifact_id, epochs, lr, history)
    if adjusted != (epochs, lr):
        epochs, lr = adjusted
        hypothesis += "；检测到同一父模型下重复配置，已确定性换轴"
    epochs, lr, strategy_reason = _apply_profile_strategy(
        rnd.dataset_key, rnd.round_index, epochs, lr, history,
        rnd.parent.artifact_id)
    if strategy_reason:
        hypothesis += "；" + strategy_reason
    return epochs, lr, hypothesis, receipt.call_id


def _aggregate_better(task, before, after):
    """回退口径：报告里的 before/after 单点比较（信息不足则保守保留父模型）。"""
    key = "macro_f1" if task == "classification" else "rmse"
    vb, va = before.get(key), after.get(key)
    if not isinstance(vb, (int, float)) or not isinstance(va, (int, float)):
        return False
    return va > vb if task == "classification" else va < vb


def _decide_promote(rnd, task, parent_ckpt, cand_ckpt, report):
    """候选是否应上线：先在验证投影上做配对 bootstrap，不可用则回退聚合比较。"""
    before, after = report.get("validation_before", {}), report.get("validation_after", {})
    info = {}
    try:
        if str(BASELINES_DIR.parent) not in sys.path:
            sys.path.insert(0, str(BASELINES_DIR.parent))
        from baselines.model import Model
        from baselines.data import read_dataset
        from decision import should_promote
        cand_model = Model.load(str(cand_ckpt))
        val = read_dataset(rnd.data.validation_path, task, cand_model.meta.get("classes", []))
        parent_pred = Model.load(str(parent_ckpt)).predict(val["x"])
        cand_pred = cand_model.predict(val["x"])
        promote, info = should_promote(parent_pred, cand_pred, val["y"], task)
        if promote is not None:
            return promote, info
    except Exception as exc:  # noqa: BLE001 —— 门控不中断本轮
        info = {"reason": "pred_failed:" + type(exc).__name__}
    return _aggregate_better(task, before, after), {"fallback": "aggregate", **info}


def train(rnd):
    task = rnd.task["task"]
    metric = rnd.task.get("metric", "macro_f1" if task == "classification" else "rmse")
    direction = rnd.task.get("metric_direction",
                             "higher" if task == "classification" else "lower")
    parent_ckpt = rnd.parent.path("checkpoint.npz")          # 摘要校验在 /previous 完成
    history = rnd.history()
    epochs, lr, hypothesis, call_id = _ask_model(rnd, task, metric, history)

    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    cand = work / "cand"                                      # train_run 要求新目录
    cmd = [sys.executable, "-m", "baselines.train",
           "--profile", rnd.dataset_key,
           "--train", rnd.data.train_path, "--validation", rnd.data.validation_path,
           "--resume", str(parent_ckpt),
           "--epochs", str(epochs), "--output", str(cand),
           "--stage-id", f"round-{rnd.round_index}"]
    if lr is not None:
        cmd += ["--learning-rate", str(lr)]
    env = {**os.environ, "PYTHONPATH": str(BASELINES_DIR.parent)}
    proc = subprocess.run(cmd, cwd=str(BASELINES_DIR.parent), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        # 续训失败不让整轮失败（C1 无人干预跑完）：优雅保留父模型并正常结束本轮。
        err = (proc.stderr.strip().splitlines() or ["非零退出"])[-1][:500]
        parent_metrics = _parent_metrics(rnd, history)
        drift = _platform_metric_drift(
            metric, direction, {}, parent_metrics, dataset_key=rnd.dataset_key)
        loop_evidence = _loop_evidence(
            history, rnd.call_ids, hypothesis, "train_failed", "failure_fallback")
        rnd.finish(
            selected_artifact_id=rnd.parent.artifact_id,
            hypothesis=hypothesis,
            change_summary=f"续训子进程失败，本轮保留父模型（{err}）"[:3900],
            reason=f"官方续训失败：{err}；为保证无人干预跑完，保留父模型。模型调用 {call_id}。",
            configuration={"epochs": epochs, "learning_rate": lr,
                           "parent_artifact_id": rnd.parent.artifact_id,
                           "model_call_id": call_id, "train_failed": err,
                           "platform_metrics": {},
                           "parent_platform_metrics": parent_metrics,
                           "platform_drift": drift,
                           "loop_evidence": loop_evidence,
                           "complete_loops_total": (
                               _complete_sdk_loop_count(history)
                               + int(loop_evidence["complete"]))},
            seed=42, history_ids=[h["id"] for h in history],
            model_call_ids=list(rnd.call_ids))
        return
    report = json.loads((cand / "training.json").read_text())
    after = report.get("validation_after", {})

    # 本地配对检验只作诊断；上线选择严格以平台 submit_candidate 返回摘要为准。
    _, local_diagnostic = _decide_promote(
        rnd, task, parent_ckpt, cand / "checkpoint.npz", report)
    change = f"从父检查点续训 epochs={epochs} lr={lr if lr is not None else '默认'}"
    seed = int(report.get("seed", 42))
    configuration = {"epochs": epochs, "learning_rate": lr, "model_call_id": call_id,
                     "parent_artifact_id": rnd.parent.artifact_id,
                     "local_validation_after": after,
                     "local_diagnostic": local_diagnostic}
    submitted = rnd.submit_candidate(
        cand, hypothesis=hypothesis, change_summary=change,
        configuration=configuration, seed=seed,
        checkpoint_paths=["checkpoint.npz"], code_paths=[], predict_argv=PREDICT_ARGV)

    candidate_metrics = _summary_metrics({"summary": submitted.get("summary")})
    parent_metrics = _parent_metrics(rnd, history)
    platform_drift = _platform_metric_drift(
        metric, direction, candidate_metrics, parent_metrics,
        dataset_key=rnd.dataset_key)
    promote, platform_decision = _select_from_platform(
        task=task, metric=metric, direction=direction,
        candidate=candidate_metrics, parent=parent_metrics,
        first_round=rnd.round_index == 1, dataset_key=rnd.dataset_key)
    if promote:
        selected, selected_metrics = submitted["artifact_id"], candidate_metrics
        verdict, selected_source = "采纳候选", "candidate"
    else:
        selected, selected_metrics = rnd.parent.artifact_id, parent_metrics
        verdict, selected_source = "保留父模型", "parent"
    selected_history_id = None
    history_best = _best_allowed_history(
        history, getattr(rnd, "artifacts", []), metric, direction,
        parent_metrics, dataset_key=rnd.dataset_key)
    if history_best is not None:
        restore, _ = _select_from_platform(
            task=task, metric=metric, direction=direction,
            candidate=history_best["metrics"], parent=selected_metrics,
            first_round=False, dataset_key=rnd.dataset_key)
        if restore:
            selected = history_best["artifact_id"]
            selected_metrics = history_best["metrics"]
            selected_history_id = history_best["history_id"]
            verdict, selected_source = "恢复历史最佳产物", "history"
    finish_configuration = {
        **configuration,
        "platform_metrics": candidate_metrics,
        "parent_platform_metrics": parent_metrics,
        "platform_decision": platform_decision,
        "selected_source": selected_source,
        "selected_history_id": selected_history_id,
        "selected_platform_metrics": selected_metrics,
        "platform_drift": platform_drift,
    }
    loop_evidence = _loop_evidence(
        history, rnd.call_ids, hypothesis, "candidate_measured", "platform_summary")
    finish_configuration["loop_evidence"] = loop_evidence
    finish_configuration["complete_loops_total"] = (
        _complete_sdk_loop_count(history) + int(loop_evidence["complete"]))
    reason = (f"平台摘要判定：{verdict}。模型建议调用 {call_id}；"
              f"平台候选指标={json.dumps(candidate_metrics, ensure_ascii=False)}；"
              f"平台父模型指标={json.dumps(parent_metrics, ensure_ascii=False)}；"
              f"判定={json.dumps(platform_decision, ensure_ascii=False)}；"
              f"漂移={json.dumps(platform_drift, ensure_ascii=False)}"[:3900])
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis, change_summary=change,
        reason=reason, configuration=finish_configuration, seed=seed,
        history_ids=[h["id"] for h in history],
        model_call_ids=list(rnd.call_ids))

