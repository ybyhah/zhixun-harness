"""提交侧「候选是否显著优于父模型」门控（numpy-only，零外部依赖）。

口径与官方 baselines.train.metrics 一致：分类 Macro-F1 在 set(expected)∪set(argmax) 上宏平均，
回归 -RMSE（统一「越大越好」）。配对 bootstrap 在同一批验证样本上比较两模型的差值分布。
任何退化输入/异常都返回 promote=None，交 train() 回退聚合比较；绝不抛错中断本轮。
自评仅用于本地「要不要上线」的决策，平台仍独立测量，不对外另报分数。
"""
import numpy as np


def classification_macro_f1(pred, y, idx=None):
    labels = np.asarray(pred).argmax(axis=1)
    expected = np.asarray(y)
    if idx is not None:
        labels, expected = labels[idx], expected[idx]
    if expected.size == 0:
        return float("nan")
    observed = sorted(set(expected.tolist()) | set(labels.tolist()))
    f1 = []
    for label in observed:
        denom = np.sum(labels == label) + np.sum(expected == label)
        f1.append(2 * np.sum((labels == label) & (expected == label)) / denom)
    return float(np.mean(f1))


def regression_rmse(pred, y, idx=None):
    p, e = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if idx is not None:
        p, e = p[idx], e[idx]
    if e.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - e) ** 2)))


def score(pred, y, task, idx=None):
    """统一成「越大越好」：分类 Macro-F1，回归 -RMSE。"""
    if task == "classification":
        return classification_macro_f1(pred, y, idx)
    r = regression_rmse(pred, y, idx)
    return -r if r == r else float("nan")


def should_promote(parent_pred, cand_pred, y, task, *, p_threshold=0.95,
                   bootstrap=500, seed=42):
    """候选相对父模型做配对 bootstrap。返回 (promote, info)。

    promote=True 表示 p(候选更好) ≥ 阈值；False 表示不显著更好（应保留父模型）；
    None 表示无法判定（形状不符/非有限/异常），由上层回退聚合比较。
    """
    try:
        y = np.asarray(y)
        a, b = np.asarray(cand_pred), np.asarray(parent_pred)
        n = y.shape[0]
        if n == 0 or a.shape[0] != n or b.shape[0] != n or bootstrap < 1:
            return None, {"reason": "shape_mismatch"}
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return None, {"reason": "nonfinite"}
        rng = np.random.default_rng(seed)
        diffs = np.empty(bootstrap, dtype=np.float64)
        for i in range(bootstrap):
            idx = rng.integers(0, n, n)
            sa, sb = score(a, y, task, idx), score(b, y, task, idx)
            if sa != sa or sb != sb:
                return None, {"reason": "nan_score"}
            diffs[i] = sa - sb                     # score 已统一越大越好
        p_better = float(np.mean(diffs > 0))
        info = {"p_better": p_better, "mean_diff": float(np.mean(diffs)),
                "ci_low": float(np.percentile(diffs, 2.5)),
                "ci_high": float(np.percentile(diffs, 97.5)),
                "bootstrap": int(bootstrap), "p_threshold": p_threshold}
        return (p_better >= p_threshold), info
    except Exception as exc:  # noqa: BLE001 —— 门控绝不中断闭环
        return None, {"reason": type(exc).__name__}
