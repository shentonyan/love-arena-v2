"""
jev_local.py — 在本地 Ollama 上实现类 Jev 的类型化判定（v0.2）

原理（G-Eval, Liu et al. 2023）：判官只输出一个标记，读取该位置的
top_logprobs，把概率质量归到合法标记并归一化。
去位置偏差（Zheng et al. 2023; Wang et al. 2023）：
  - Score 正序、倒序各问一次，取平均
  - Noul  陈述与其否定各问一次：p = (p(陈述) + 1 - p(否定)) / 2
coverage = 合法标记所占原始概率质量；< 0.5 视为未按格式作答。
这里得到的是原始置信度，不是校准后的概率（校准在 run.py validate 中做）。

依赖：仅标准库。Ollama >= 0.12.11。
环境变量 LOVE_ARENA_FAKE=1 时使用模拟后端（仅供测试流程）。
"""
from __future__ import annotations
import hashlib, json, math, os, random, urllib.error, urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL
FAKE = os.environ.get("LOVE_ARENA_FAKE") == "1"
FAKE_NEGATIONS: set[str] = set()   # 由 config 注册，仅模拟后端使用
NUM_CTX = 2048        # 上下文窗口；越小，KV 缓存越省显存，可并发的请求越多
NUM_THREAD = None     # 部分层在 CPU 上时可设为 CPU 性能核数（i7-14700KF 为 8）

JUDGE_SYSTEM = "你是一个严格、中立的评审。你只输出一个标记（一个字母或一个数字），不输出任何解释、标点或其他文字。"


def _post(path, body, timeout=900):
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本地请求绕过 FlClash 代理
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama {e.code}: {e.read().decode('utf-8', 'replace')}") from None


def _generate(model, prompt, system=None, *, temperature=0.0, num_predict=3,
              seed=None, logprobs=False):
    if FAKE:
        return _fake(model, prompt, logprobs, seed)
    options = {"temperature": temperature, "num_predict": num_predict, "num_ctx": NUM_CTX}
    if NUM_THREAD:
        options["num_thread"] = NUM_THREAD
    if seed is not None:
        options["seed"] = seed
    body = {"model": model, "prompt": prompt, "stream": False, "think": False,
            "options": options, "keep_alive": "15m"}
    if system:
        body["system"] = system
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 20
    try:
        return _post("/api/generate", body)
    except RuntimeError as e:
        if "think" in str(e).lower():
            body.pop("think")
            return _post("/api/generate", body)
        raise


def unload(model):
    if FAKE:
        return
    try:
        _post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def generate_text(model, prompt, system, temperature=0.7, seed=None, max_tokens=300):
    r = _generate(model, prompt, system, temperature=temperature,
                  num_predict=max_tokens, seed=seed)
    return r.get("response", "").strip()


def _norm(t):
    return t.strip().strip("*`\"'“”‘’（）()[]【】:：.。，,").upper()


def _dist(model, prompt, labels):
    r = _generate(model, prompt, JUDGE_SYSTEM, logprobs=True)
    positions = r.get("logprobs") or []
    if not positions:
        raise RuntimeError("Ollama 未返回 logprobs，请升级到 0.12.11 以上")
    for pos in positions:
        mass = dict.fromkeys(labels, 0.0)
        for c in pos.get("top_logprobs", []):
            k = _norm(c["token"])
            if k in mass:
                mass[k] += math.exp(c["logprob"])
        cov = sum(mass.values())
        if cov > 0.05:
            return {k: v / cov for k, v in mass.items()}, cov
    return {k: 1 / len(labels) for k in labels}, 0.0


def score(model, state, instruction, levels, orders=2, guide=None):
    """levels 从低到高。返回 (期望分 0..len-1, 最小 coverage)。"""
    n = len(levels)
    vals, covs = [], []
    for rev in ([False, True] if orders == 2 else [False]):
        idx = list(range(n))[::-1] if rev else list(range(n))
        shown = "\n".join(f"{k}. {levels[i]}" for k, i in enumerate(idx))
        order_note = "（从高到低排列）" if rev else "（从低到高排列）"
        g = f"【评判说明】\n{guide}\n\n" if guide else ""
        p = (f"{g}{state}\n\n【问题】{instruction}\n\n【量表{order_note}】\n{shown}\n\n只回答一个数字：")
        probs, cov = _dist(model, p, [str(k) for k in range(n)])
        vals.append(sum(p_ * idx[int(k)] for k, p_ in probs.items()))
        covs.append(cov)
    return sum(vals) / len(vals), min(covs)


def noul(model, state, statement, negation, orders=2, guide=None):
    """返回 (陈述为真的概率, 最小 coverage)。"""
    def ask(stmt):
        g = f"【评判说明】\n{guide}\n\n" if guide else ""
        p = f"{g}{state}\n\n【陈述】{stmt}\n\n该陈述为真回答 Y，为假回答 N。只回答一个字母："
        probs, cov = _dist(model, p, ["Y", "N"])
        return probs["Y"], cov
    p1, c1 = ask(statement)
    if orders == 1:
        return p1, c1
    p2, c2 = ask(negation)
    return (p1 + 1 - p2) / 2, min(c1, c2)


# ---------------- 模拟后端：带一个隐藏“质量”，让流程可测 ----------------
def _h(*xs):
    return int(hashlib.md5("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def _fake(model, prompt, logprobs, seed):
    if not logprobs:
        q = (_h(model, prompt, seed) % 100) / 100
        return {"response": f"[{model} q={q:.2f}] 我对这件事有些失望，想和你一起找到改进的办法。"}
    import re
    m = re.search(r"q=([0-9.]+)", prompt)
    q = float(m.group(1)) if m else 0.5
    jm = {"gemma4:26b": 0.0, "qwen2.5:7b": 0.15, "llama3.1:8b": -0.1}.get(model, 0)
    if re.search(r"\[" + re.escape(model) + " ", prompt):
        jm += 0.15                                   # 自偏好
    rng = random.Random(_h(model, prompt))
    x = min(max(q + jm + rng.gauss(0, 0.15), 0.01), 0.99)
    if "只回答一个数字" in prompt:
        labels = [l.split(".")[0] for l in prompt.split("【量表")[1].split("\n")[1:] if l[:1].isdigit()]
        n = len(labels)
        desc = "从高到低" in prompt
        neg = "残留" in prompt                         # 恨语残留：质量越高分越低
        target = (1 - x if neg else x) * (n - 1)
        if desc:
            target = (n - 1) - target
        w = [math.exp(-((k - target) ** 2)) for k in range(n)]
    else:
        neg_stmt = any(n in prompt for n in FAKE_NEGATIONS)
        py = x if not neg_stmt else 1 - x
        stmt_part = prompt.split("【陈述】")[1]
        if "居高临下" in stmt_part or "敌意、贬低、给人贴标签" in stmt_part:
            py = 1 - py
        labels, w = ["Y", "N"], [py, 1 - py]
    s = sum(w)
    tops = [{"token": l, "logprob": math.log(max(v / s * 0.92, 1e-9))} for l, v in zip(labels, w)]
    tops.append({"token": "\n", "logprob": math.log(0.08)})
    return {"logprobs": [{"token": labels[0], "top_logprobs": tops}]}
