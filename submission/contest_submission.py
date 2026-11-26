"""contest.harness.workspace.v2 提交物：把官方 baselines 续训接到 SDK 轮次契约。

每轮都从平台父检查点续训，并真实调用一次模型（读历史 → 提下一轮续训超参，满足
「每轮至少一次模型调用」，也体现 Agent 自主决策）。密钥不入镜像：模型只经 rnd.model()。
自评无效：候选分数由平台 submit_candidate 后测量。
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"   # 镜像内 /app/baselines
EPOCH_RANGE = (1, 50)
LR_RANGE = (1e-6, 0.1)
_MODEL_CACHE: dict = {}


def _clamp(value, low, high):
    return max(low, min(high, value))


def _extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型回复没有 JSON 对象")
    return json.loads(text[start:end + 1])


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
    lines = []
    for h in history[-6:]:
        v = h.get("validation_after") or h.get("metrics") or {}
        lines.append(f"  轮{h.get('round_index', '?')}: {json.dumps(v, ensure_ascii=False)}")
    hist = "\n".join(lines) if lines else "  （暂无历史）"
    user = (
        f"数据集 {rnd.dataset_key}，任务 {task}，主指标 {metric}（{direction}）。\n"
        f"本轮从父检查点续训，梯度只用公开训练集。历史检查点表现：\n{hist}\n\n"
        "请给出本轮续训超参，严格只输出 JSON："
        '{"epochs": 整数1-50, "learning_rate": 0到0.1小数, "hypothesis": "一句话理由"}'
    )
    epochs, lr, hypothesis = 5, None, "回退官方默认超参续训一轮"
    receipt = rnd.model([{"role": "user", "content": user}],
                        max_tokens=min(512, rnd.max_output_tokens), temperature=0)
    try:
        data = _extract_json(receipt.text)
        epochs = int(_clamp(int(data["epochs"]), *EPOCH_RANGE))
        lr = float(_clamp(float(data["learning_rate"]), *LR_RANGE))
        if isinstance(data.get("hypothesis"), str) and data["hypothesis"].strip():
            hypothesis = data["hypothesis"].strip()[:2000]
    except Exception as exc:  # noqa: BLE001 —— 慢通路失败回退，不中断本轮
        hypothesis = f"模型回复不可用（{type(exc).__name__}），回退官方默认超参"
    return epochs, lr, hypothesis, receipt.call_id


STALL_SKIP = 3              # 同一父模型连续多少轮无改进后跳过一轮训练
REGRESSION_EPOCH_CAP = 5    # B/C 第 2 轮起单轮续训 epochs 上限
NASA_TOLERANCE = 1.05
# 回归本轮内小网格（来自 scripts/sweep_bc.py 对 B/C 公开集的扫描：B 偏好较大 lr，C 偏好小 lr 短续训）
INNER_GRID = [(1, 0.0003), (5, 0.0003), (5, 0.001), (1, 0.01), (5, 0.01), (3, 0.003)]  # C：候选 NASA 非对称分不得比父模型差 5% 以上（高估寿命的安全护栏）


def _num(value):
    return isinstance(value, (int, float)) and value == value and not isinstance(value, bool)


def _nasa(metrics):
    for key, value in (metrics or {}).items():
        if "nasa" in str(key).lower() and _num(value):
            return float(value)
    return None


def _decide_promote(task, parent_metrics, cand_metrics, report):
    """候选是否上线。优先用平台 submit_candidate 返回的公开验证摘要（与父模型的平台摘要比），
    父模型平台摘要缺失（首轮）时回退 training.json 的 before/after。
    回归任务另加 NASA 护栏：主指标变好但 NASA 明显恶化时拒绝（2026-11-25 正式评测 C 第 9 轮的教训）。
    """
    key = "macro_f1" if task == "classification" else "rmse"
    source = "platform"
    before, after = parent_metrics or {}, cand_metrics or {}
    if not (_num(before.get(key)) and _num(after.get(key))):
        source = "training_json"
        before, after = report.get("validation_before", {}), report.get("validation_after", {})
    vb, va = before.get(key), after.get(key)
    info = {"source": source, "metric": key, "parent": vb, "candidate": va}
    if not (_num(vb) and _num(va)):
        return False, {**info, "reason": "metric_missing"}
    better = va > vb if task == "classification" else va < vb
    if not better:
        return False, {**info, "reason": "not_better"}
    nb, na = _nasa(before), _nasa(after)
    if nb is not None and na is not None:
        info.update(parent_nasa=nb, candidate_nasa=na)
        if na > nb * NASA_TOLERANCE:
            return False, {**info, "reason": "nasa_veto"}
    return True, {**info, "reason": "better"}


def _load_state(rnd):
    """读取上一轮 /state 里的 harness_state.json（失败则空状态，不中断）。"""
    try:
        if rnd.state_path:
            path = Path(rnd.state_path) / "harness_state.json"
            if path.is_file():
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    return data
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_state(rnd, state):
    try:
        work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
        target = work / f"state-r{rnd.round_index}"
        if target.exists():
            return
        target.mkdir(parents=True)
        (target / "harness_state.json").write_text(json.dumps(state, ensure_ascii=False))
        rnd.save_state(target)
    except Exception as exc:  # noqa: BLE001 —— 状态只是优化，不让本轮失败
        print(f"save_state failed: {exc!r}", file=sys.stderr)


def _dedupe(epochs, lr, default_lr, tried):
    """同一父模型、同一 (epochs, lr) 在 seed=42 下结果逐位相同，重跑纯属浪费：
    确定性换轴（先减半学习率，再缩短 epochs），直到出现未试过的组合。"""
    cur_lr = lr if lr is not None else default_lr
    seen = {(int(e), round(float(l), 10)) for e, l in tried}
    e, l = epochs, cur_lr
    for step in range(8):
        if (int(e), round(float(l), 10)) not in seen:
            break
        if step % 2 == 0:
            l = max(LR_RANGE[0], l / 2)
        else:
            e = max(1, e - 2)
    changed = (e, l) != (epochs, cur_lr)
    return e, (l if changed or lr is not None else None), changed


def _run_train(rnd, parent_ckpt, epochs, lr, out):
    cmd = [sys.executable, "-m", "baselines.train",
           "--profile", rnd.dataset_key,
           "--train", rnd.data.train_path, "--validation", rnd.data.validation_path,
           "--resume", str(parent_ckpt),
           "--epochs", str(epochs), "--output", str(out),
           "--stage-id", f"round-{rnd.round_index}"]
    if lr is not None:
        cmd += ["--learning-rate", str(lr)]
    env = {**os.environ, "PYTHONPATH": str(BASELINES_DIR.parent)}
    return subprocess.run(cmd, cwd=str(BASELINES_DIR.parent), env=env,
                          capture_output=True, text=True)


def _nasa_np(y, p):
    import numpy as np
    d = np.clip(p - y, -500, 500)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def _local_score(task, report, out, parent_ckpt, rnd):
    """本地验证集打分（越小越好）。回归：RMSE，且 NASA 不得比父模型差 NASA_TOLERANCE 以上。"""
    before, after = report.get("validation_before", {}), report.get("validation_after", {})
    if task != "regression":
        return {"ok": False}
    rb, ra = before.get("rmse"), after.get("rmse")
    if not (_num(rb) and _num(ra)):
        return {"ok": False}
    info = {"rmse": ra, "key": ra, "ok": ra < rb}
    try:
        if str(BASELINES_DIR.parent) not in sys.path:
            sys.path.insert(0, str(BASELINES_DIR.parent))
        from baselines.model import Model
        from baselines.data import read_dataset
        if "val" not in _MODEL_CACHE:
            _MODEL_CACHE["val"] = read_dataset(rnd.data.validation_path, "regression", [])
        val = _MODEL_CACHE["val"]
        if "parent_nasa" not in _MODEL_CACHE:
            _MODEL_CACHE["parent_nasa"] = _nasa_np(val["y"], Model.load(str(parent_ckpt)).predict(val["x"]))
        na = _nasa_np(val["y"], Model.load(str(out / "checkpoint.npz")).predict(val["x"]))
        info["nasa_local"] = na
        if rnd.dataset_key == "C" and na > _MODEL_CACHE["parent_nasa"] * NASA_TOLERANCE:
            info["ok"] = False
    except Exception as exc:  # noqa: BLE001
        info["nasa_error"] = type(exc).__name__
    return info


def train(rnd):
    task = rnd.task["task"]
    metric = rnd.task.get("metric", "macro_f1" if task == "classification" else "rmse")
    parent_ckpt = rnd.parent.path("checkpoint.npz")          # 摘要校验在 /previous 完成
    history = rnd.history()
    epochs, lr, hypothesis, call_id = _ask_model(rnd, task, metric, history)
    state = _load_state(rnd)
    parent_id = rnd.parent.artifact_id
    parent_metrics = (state.get("metrics") or {}).get(parent_id)
    tried = (state.get("tried") or {}).get(parent_id, [])
    try:
        default_lr = float(json.loads((BASELINES_DIR / "configs" / f"{rnd.dataset_key}.json")
                                      .read_text())["learning_rate"])
    except Exception:  # noqa: BLE001
        default_lr = 0.003
    stall = int(state.get("stall", 0))
    if stall >= STALL_SKIP and len(history) >= 3:
        # 资源效率：同一父模型连续 STALL_SKIP 轮无改进 → 本轮不训练，保留父模型，
        # 并清零计数，下一轮再以去重后的新组合尝试一次（避免永久停训丢掉后续提升）。
        _save_state(rnd, {**state, "stall": 0})
        rnd.finish(
            selected_artifact_id=parent_id, hypothesis=hypothesis,
            change_summary=f"连续 {stall} 轮候选未优于父模型，本轮跳过训练以节省资源",
            reason=(f"资源门控：父模型 {parent_id} 已连续 {stall} 轮未被超越，跳过训练、保留父模型；"
                    f"模型建议调用 {call_id}。下一轮恢复搜索。")[:3900],
            configuration={"training_skipped": True, "stall": stall, "model_call_id": call_id},
            seed=42, history_ids=[h["id"] for h in history],
            model_call_ids=list(rnd.call_ids))
        return
    if task == "regression" and rnd.round_index > 1:
        epochs = min(epochs, REGRESSION_EPOCH_CAP)   # B/C 历史上 >5 epoch 续训几乎都被拒，纯耗时
    epochs, lr, deduped = _dedupe(epochs, lr, default_lr, tried)
    if deduped:
        hypothesis = (hypothesis + f"；该组合已在同一父模型上试过，确定性换轴为 epochs={epochs} lr={lr}")[:2000]

    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    configs = [(epochs, lr)]
    if task == "regression":
        # 回归续训很便宜（B≈0.1s/C≈1s 每 epoch）：本轮内对 Qwen 建议 + 小网格各训一次，
        # 只提交验证集最好的一个。梯度仍只用训练集，验证集仅用于选择。
        for e, l in INNER_GRID:
            if (e, l) not in configs and [e, l] not in tried:
                configs.append((e, l))
    proc, cand, chosen, inner = None, None, None, []
    for idx, (e, l) in enumerate(configs):
        out = work / f"cand{idx}"
        p = _run_train(rnd, parent_ckpt, e, l, out)
        if p.returncode != 0:
            proc = proc or p
            continue
        rep = json.loads((out / "training.json").read_text())
        score = _local_score(task, rep, out, parent_ckpt, rnd)
        inner.append({"epochs": e, "learning_rate": l, **score})
        if score.get("ok") and (chosen is None or score["key"] < chosen[2]["key"]):
            chosen = (e, l, score, out)
    if chosen is None and inner and task == "regression":
        # 本轮所有候选在验证集上都不优于父模型：不提交、保留父模型（省平台测量，避免劣化上线）
        tried_map = dict(state.get("tried") or {})
        tried_map[parent_id] = list(tried) + [[i["epochs"], i["learning_rate"] if i["learning_rate"] is not None
                                               else default_lr] for i in inner]
        _save_state(rnd, {**state, "tried": tried_map, "stall": stall + 1})
        rnd.finish(
            selected_artifact_id=parent_id, hypothesis=hypothesis,
            change_summary=f"本轮试了 {len(inner)} 组续训超参，验证集均未优于父模型，未提交候选",
            reason=(f"门控判定：保留父模型。模型建议调用 {call_id}；本轮内搜索="
                    f"{json.dumps(inner[:8], ensure_ascii=False)}")[:3900],
            configuration={"inner_search": inner[:16], "model_call_id": call_id, "submitted": False},
            seed=42, history_ids=[h["id"] for h in history],
            model_call_ids=list(rnd.call_ids))
        return
    if chosen is None and inner:
        first = next((i for i, c in enumerate(configs) if (work / f"cand{i}" / "training.json").exists()), None)
        if first is not None:
            chosen = (configs[first][0], configs[first][1], {}, work / f"cand{first}")
    if chosen is not None:
        epochs, lr, _, cand = chosen
        proc = None
    if proc is not None or cand is None:
        proc = proc or SimpleNamespace(returncode=1, stderr="no candidate")
        # 续训失败不让整轮失败（C1 无人干预跑完）：优雅保留父模型并正常结束本轮。
        err = (proc.stderr.strip().splitlines() or ["非零退出"])[-1][:500]
        rnd.finish(
            selected_artifact_id=rnd.parent.artifact_id,
            hypothesis=hypothesis,
            change_summary=f"续训子进程失败，本轮保留父模型（{err}）"[:3900],
            reason=f"官方续训失败：{err}；为保证无人干预跑完，保留父模型。模型调用 {call_id}。",
            configuration={"epochs": epochs, "learning_rate": lr,
                           "model_call_id": call_id, "train_failed": err},
            seed=42, history_ids=[h["id"] for h in history],
            model_call_ids=list(rnd.call_ids))
        return
    report = json.loads((cand / "training.json").read_text())
    after = report.get("validation_after", {})

    change = f"从父检查点续训 epochs={epochs} lr={lr if lr is not None else '默认'}"
    seed = int(report.get("seed", 42))
    configuration = {"epochs": epochs, "learning_rate": lr, "model_call_id": call_id,
                     "validation_after": after, "deduped": deduped, "inner_search": inner[:16]}
    submitted = rnd.submit_candidate(
        cand, hypothesis=hypothesis, change_summary=change,
        configuration=configuration, seed=seed,
        checkpoint_paths=["checkpoint.npz"], code_paths=[], predict_argv=PREDICT_ARGV)
    summary = submitted.get("summary") if isinstance(submitted, dict) else None
    cand_metrics = summary.get("metrics") if isinstance(summary, dict) else None
    cand_metrics = cand_metrics if isinstance(cand_metrics, dict) else {}

    # 门控：以平台公开验证摘要为准；不优于父模型（或 NASA 明显恶化）则保留父模型
    promote, decision = _decide_promote(task, parent_metrics, cand_metrics, report)
    configuration["decision"] = decision
    configuration["platform_metrics"] = cand_metrics

    if promote:
        selected, verdict = submitted["artifact_id"], "采纳候选"
    else:
        selected, verdict = rnd.parent.artifact_id, "保留父模型"

    metrics_map = dict(state.get("metrics") or {})
    if cand_metrics:
        metrics_map[submitted["artifact_id"]] = cand_metrics
    tried_map = dict(state.get("tried") or {})
    tried_map[parent_id] = list(tried) + [[epochs, lr if lr is not None else default_lr]] + \
        [[i["epochs"], i["learning_rate"]] for i in inner if i["learning_rate"] is not None]
    _save_state(rnd, {"metrics": metrics_map, "tried": tried_map,
                      "stall": 0 if promote else stall + 1})
    reason = (f"门控判定：{verdict}。模型建议调用 {call_id}；"
              f"候选 vs 父模型比较={json.dumps(decision, ensure_ascii=False)}"[:3900])
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis, change_summary=change,
        reason=reason, configuration=configuration, seed=seed,
        history_ids=[h["id"] for h in history],
        model_call_ids=list(rnd.call_ids))

