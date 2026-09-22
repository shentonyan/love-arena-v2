"""统计工具（仅依赖 numpy）。"""
import numpy as np


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):                      # 并列取平均秩
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b):
    a, b = rankdata(a), rankdata(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def auc(scores, labels):
    """Mann–Whitney 形式的 AUC。"""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def kripp_alpha_interval(matrix):
    """Krippendorff's α（interval 度量）。matrix: 评分者 × 条目，缺失为 nan。"""
    m = np.asarray(matrix, float)
    pairs_within, all_vals = [], []
    for col in m.T:
        v = col[~np.isnan(col)]
        if len(v) < 2:
            continue
        all_vals.append(v)
        n_u = len(v)
        d = (v[:, None] - v[None, :]) ** 2
        pairs_within.append(d.sum() / (n_u - 1))
    if not all_vals:
        return float("nan")
    vals = np.concatenate(all_vals)
    n = len(vals)
    do = sum(pairs_within) / n
    de = ((vals[:, None] - vals[None, :]) ** 2).sum() / (n * (n - 1))
    return float(1 - do / de) if de > 0 else float("nan")


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, float), -30, 30)))


def platt_fit(raw_p, y, l2=1.0, iters=200):
    """Platt 校准：p_cal = sigmoid(a * logit(raw) + b)。
    logit 截断在 ±6.9（p∈[0.001,0.999]），带 L2 正则与回溯线搜索，小样本也能稳定收敛。"""
    x = logit(raw_p, eps=1e-3)
    y = np.asarray(y, float)
    X = np.column_stack([x, np.ones_like(x)])
    def loss(w):
        z = np.clip(X @ w, -30, 30)
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * w[0] ** 2)
    w = np.zeros(2)
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + np.array([l2 * w[0], 0.0])
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag([l2, 1e-6])
        step = np.linalg.solve(H, g)
        t, f0 = 1.0, loss(w)
        while loss(w - t * step) > f0 and t > 1e-6:
            t /= 2
        w = w - t * step
        if np.abs(t * step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def platt_apply(raw_p, ab):
    a, b = ab
    return sigmoid(a * logit(raw_p, eps=1e-3) + b)


def loo_brier(raw_p, y):
    """留一法：校准前后的 Brier 分数（越低越好）。"""
    raw_p, y = np.asarray(raw_p, float), np.asarray(y, float)
    cal = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cal[i] = platt_apply(raw_p[i:i + 1], platt_fit(raw_p[m], y[m]))[0]
    return float(((raw_p - y) ** 2).mean()), float(((cal - y) ** 2).mean())


def fit_effects(y, contestant, judge, sentence, self_flag, contestants, judges, use_self=True):
    """y = contestant + judge + sentence (+ β_self·self) + ε 的最小二乘。
    返回 dict(contestant_means, judge_offsets(中心化), self_beta)。
    这是多面 Rasch（Linacre 1989）思路的线性近似：把判官宽严与自偏好从选手效应中分离。"""
    y = np.asarray(y, float)
    sents = sorted(set(sentence))
    cols = []
    for c in contestants:
        cols.append(np.array([ci == c for ci in contestant], float))
    for j in judges[1:]:
        cols.append(np.array([ji == j for ji in judge], float))
    for s in sents[1:]:
        cols.append(np.array([si == s for si in sentence], float))
    if use_self:
        cols.append(np.asarray(self_flag, float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    nc, nj = len(contestants), len(judges)
    j_raw = np.concatenate([[0.0], beta[nc:nc + nj - 1]])
    j_off = j_raw - j_raw.mean()
    self_beta = float(beta[-1]) if use_self else 0.0
    return {"judge_offsets": dict(zip(judges, j_off.tolist())), "self_beta": self_beta,
            "rank": int(np.linalg.matrix_rank(X)), "ncol": X.shape[1]}
