"""本地模拟平台：用 base/ 公开数据把 submission.train() 跑 N 轮（state 在轮间传递）。
平台指标用 baselines.train.metrics 在公开验证集上计算（近似平台公开摘要）。
用法：python scripts/sim_submission.py B 10
"""
import json, shutil, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
import contest_submission as cs  # noqa: E402
from baselines.model import Model  # noqa: E402
from baselines.data import read_dataset  # noqa: E402
from baselines.train import metrics  # noqa: E402

PKG = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}


def _llm(messages, options):
    """设置了 QWEN_API_KEY 时调用真实 Qwen（OpenAI 兼容接口），否则返回固定建议。
    环境变量：QWEN_API_KEY、QWEN_BASE_URL（默认 DashScope 兼容模式）、QWEN_MODEL（默认 qwen-plus）。"""
    import os, urllib.request
    keyv = os.environ.get("QWEN_API_KEY")
    if not keyv:
        return '{"epochs": 5, "learning_rate": 0.001, "hypothesis": "sim"}'
    base = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    body = json.dumps({"model": os.environ.get("QWEN_MODEL", "qwen-plus"), "messages": messages,
                       "max_tokens": options.get("max_tokens", 512),
                       "temperature": options.get("temperature", 0)}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {keyv}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001 —— 与平台行为一致：回复不可用则由提交代码回退
        print(f"qwen call failed: {exc!r}", file=sys.stderr)
        return ""


def main(key, rounds):
    base = ROOT / "base" / PKG[key]
    task = "classification" if key == "A" else "regression"
    tmp = Path(tempfile.mkdtemp(prefix="sim-"))
    arts = {"baseline": base / "checkpoint.npz"}
    parent, state_dir, hist, trajectory, trained = "baseline", None, [], [], 0
    val_cache = {}
    t0 = time.time()
    for r in range(1, rounds + 1):
        work = tmp / f"r{r}"; work.mkdir()
        cs.os.environ["CONTEST_WORK"] = str(work)
        out = {}

        def submit(d, **kw):
            aid = f"{key}-r{r}"; dst = tmp / aid; shutil.copytree(d, dst)
            arts[aid] = dst / "checkpoint.npz"
            m = Model.load(str(arts[aid]))
            if "v" not in val_cache:
                val_cache["v"] = read_dataset(str(base / "public/validation.jsonl"), task, m.meta.get("classes", []))
            met = metrics(m, val_cache["v"]); out["sub"] = met
            return {"artifact_id": aid, "summary": {"metrics": met}}

        def save_state(d):
            nonlocal state_dir
            state_dir = tmp / f"state{r}"; shutil.copytree(d, state_dir); return "sha"

        def finish(**kw):
            out["fin"] = kw

        rnd = SimpleNamespace(
            task={"task": task, "metric": "macro_f1" if task == "classification" else "rmse",
                  "metric_direction": "higher" if task == "classification" else "lower"},
            dataset_key=key, round_index=r, max_output_tokens=512, call_ids=[f"call-{r}"],
            parent=SimpleNamespace(artifact_id=parent, path=lambda _n, p=parent: arts[p]),
            data=SimpleNamespace(train_path=str(base / "public/train.jsonl"),
                                 validation_path=str(base / "public/validation.jsonl")),
            state_path=str(state_dir) if state_dir else None,
            history=lambda: list(hist),
            model=lambda msgs, **k: SimpleNamespace(text=_llm(msgs, k), call_id=f"call-{r}"),
            submit_candidate=submit, save_state=save_state, finish=finish)
        cs.train(rnd)
        fin = out["fin"]; parent = fin["selected_artifact_id"]
        if "sub" in out:
            trained += 1
        if "inner_search" in fin["configuration"] and "sub" not in out:
            fin["configuration"].setdefault("decision", {"reason": "local_no_improve"})
        sel = Model.load(str(arts[parent]))
        if "v" not in val_cache:
            val_cache["v"] = read_dataset(str(base / "public/validation.jsonl"), task, sel.meta.get("classes", []))
        m = metrics(sel, val_cache["v"]); trajectory.append(m)
        hist.append({"id": f"h{r}", "round_index": r, "metrics": m})
        d = fin["configuration"].get("decision", {})
        print(f"r{r:02d} sel={parent:10s} skip={fin['configuration'].get('training_skipped', False)} "
              f"ep={fin['configuration'].get('epochs')} lr={fin['configuration'].get('learning_rate')} "
              f"reason={d.get('reason')} sel_metrics={json.dumps(m)}", flush=True)
    print(f"trained={trained}/{rounds} seconds={time.time()-t0:.1f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
