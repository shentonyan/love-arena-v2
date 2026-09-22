"""
爱语擂台 v0.2 —— 分阶段运行：
  check     检查 Ollama 与三个模型能否按格式作答
  generate  三个模型改写（每句 SAMPLES_PER_MODEL 次）
  annotate  生成人工盲评表（两位标注者各一份）
  tune      在开发集上为每个判官×维度自动选择最优提示词变体
  judge     全交叉判定（每个判官判所有改写，含自己的）；可中断续跑
  validate  判官 vs 人工：一致性、校准、准入评审团
  analyze   扣除判官宽严与自偏好 → J 分 + bootstrap 置信区间
  pairs     仅在判官通过验证后产出偏好样本对
通用参数：--run 目录名（默认 main）  --limit N（只用前 N 句，试跑用）
"""
from __future__ import annotations
import argparse, csv, json, math, random, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import config as C
import jev_local as J
import stats_utils as S

ROOT = Path(__file__).parent


# ---------------- 通用 ----------------
def rdir(args):
    d = ROOT / "runs" / args.run
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
_LOCK = threading.Lock()


def append_jsonl(p, rec):
    with _LOCK:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pmap(fn, items, label, every=50):
    """并发执行 fn(item)；fn 自行写结果。显示进度与剩余时间。"""
    items = list(items)
    if not items:
        return
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(max_workers=C.WORKERS) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for f in as_completed(futs):
            f.result()
            n += 1
            if n % every == 0 or n == len(items):
                el = time.time() - t0
                print(f"  {label} {n}/{len(items)}  已用 {el / 60:.1f} 分钟，"
                      f"预计还需 {el / n * (len(items) - n) / 60:.1f} 分钟，{n / el:.1f} 项/秒")


def sentences(args):
    lines = [l.strip() for l in (ROOT / "data" / "sentences.txt").read_text(encoding="utf-8").splitlines()]
    s = [l for l in lines if l and not l.startswith("#")]
    return s[: args.limit] if args.limit else s


def short(m):
    return re.sub(r"[^a-z0-9]+", "", m.lower())


def read_csv_any(p):
    for enc in ("utf-8-sig", "gb18030"):          # Excel 另存可能变成 GBK
        try:
            with open(p, newline="", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法读取 {p}")


def to01(dim, v):
    """把某维度的原始值统一到 [0,1]、越高越好。"""
    d = C.DIMENSIONS[dim]
    if d["kind"] == "noul":
        return 1 - v if dim == "condescending" else v
    x = v / (len(d["levels"]) - 1)
    return x if d["higher_is_better"] else 1 - x


def run_spec(judge, state, spec):
    if spec["kind"] == "score":
        return J.score(judge, state, spec["instruction"], spec["levels"], C.ORDERS, spec.get("guide"))
    return J.noul(judge, state, spec["statement"], spec["negation"], C.ORDERS, spec.get("guide"))


def base_spec(dim):
    return C.VARIANTS[dim]["base"]


# ---------------- check ----------------
def cmd_check(args):
    if not J.FAKE:
        try:
            print("Ollama 版本：", J._post("/api/version", {}, timeout=10).get("version"))
        except Exception:
            import urllib.request
            print("Ollama 版本：", urllib.request.urlopen(J.OLLAMA_URL + "/api/version").read().decode())
    st = C.STATE_TEMPLATE.format(src="你从来不听我说话。", out="我有时觉得自己的话没被听到，想和你聊聊。")
    for m in C.MODELS:
        try:
            t = time.time()
            s, c1 = run_spec(m, st, base_spec("fluency"))
            p, c2 = run_spec(m, st, base_spec("honesty"))
            ok = min(c1, c2) >= C.MIN_COVERAGE
            print(f"{m:14s} 流畅={s:.2f}/3 诚实={p:.2f} coverage={min(c1, c2):.2f} "
                  f"每次判定≈{(time.time() - t) / 6:.1f}s  {'OK' if ok else '格式不稳'}")
        except Exception as e:
            print(f"{m:14s} 失败：{e}")
        J.unload(m)


# ---------------- generate ----------------
def cmd_generate(args):
    d = rdir(args); out = d / "generations.jsonl"
    done = {g["id"] for g in read_jsonl(out)}
    sents = sentences(args)
    for m in C.MODELS:
        todo = [(i, s, k) for i, s in enumerate(sents) for k in range(C.SAMPLES_PER_MODEL)
                if f"s{i:03d}-{short(m)}-k{k}" not in done]
        def work(x, m=m):
            i, s, k = x
            txt = J.generate_text(m, s, C.REWRITE_SYSTEM, C.GEN_TEMPERATURE, seed=C.SEED + 1000 * k + i)
            append_jsonl(out, {"id": f"s{i:03d}-{short(m)}-k{k}", "sent": i, "src": s, "model": m, "k": k, "out": txt})
        print(f"[generate] {m}: {len(todo)} 条")
        pmap(work, todo, m, every=20)
        J.unload(m)
    print(f"→ {out}")


def load_gens(d, args):
    g = read_jsonl(d / "generations.jsonl")
    if args.limit:
        g = [x for x in g if x["sent"] < args.limit]
    if not g:
        sys.exit("还没有改写结果，请先运行 generate")
    return g


# ---------------- annotate ----------------
ANN_COLS = ["条目", "原句", "改写", "hate_residue(0-3)", "honesty(0/1)", "fluency(0-3)",
            "equal_connection(0-3)", "condescending(0/1)", "备注"]


def cmd_annotate(args):
    d = rdir(args); gens = load_gens(d, args)
    if args.set == "holdout":
        return annotate_holdout(d, gens, args)
    rng = random.Random(C.SEED)
    per = C.ANNOTATION_N // len(C.MODELS)
    chosen = []
    for m in C.MODELS:
        pool = [g for g in gens if g["model"] == m]
        chosen += rng.sample(pool, min(per, len(pool)))
    rng.shuffle(chosen)
    key = {f"A{n + 1:03d}": g["id"] for n, g in enumerate(chosen)}
    (d / "annotation_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    adir = d / "annotations"; adir.mkdir(exist_ok=True)
    byid = {g["id"]: g for g in gens}
    for who in ("annotator_A", "annotator_B"):
        items = list(key.items()); random.Random(who).shuffle(items)   # 两人顺序不同，避免顺序效应
        p = adir / f"{who}.csv"
        if p.exists() and not args.force:
            print(f"已存在 {p}（加 --force 覆盖）"); continue
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(ANN_COLS)
            for code, gid in items:
                w.writerow([code, byid[gid]["src"], byid[gid]["out"], "", "", "", "", "", ""])
        print(f"→ {p}")
    print(f"共 {len(key)} 条，已隐去模型名。请按 CODEBOOK.md 独立标注，互不商量。")


def annotate_holdout(d, gens, args):
    """留出集：与开发集不重叠，优先取开发集没出现过的句子。"""
    dev = set(json.loads((d / "annotation_key.json").read_text(encoding="utf-8")).values())
    dev_s = {g["sent"] for g in gens if g["id"] in dev}
    rng = random.Random(C.SEED + 1); chosen = []
    for m in C.MODELS:
        pool = [g for g in gens if g["model"] == m and g["id"] not in dev]
        fresh = [g for g in pool if g["sent"] not in dev_s]; rest = [g for g in pool if g["sent"] in dev_s]
        rng.shuffle(fresh); rng.shuffle(rest)
        chosen += (fresh + rest)[: C.ANNOTATION_N // len(C.MODELS)]
    rng.shuffle(chosen)
    key = {f"H{n + 1:03d}": g["id"] for n, g in enumerate(chosen)}
    kp = d / "holdout_key.json"
    if kp.exists() and not args.force:
        sys.exit(f"已存在 {kp}（加 --force 覆盖）")
    kp.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    hd = d / "holdout"; hd.mkdir(exist_ok=True)
    for who in ("annotator_A", "annotator_B"):
        items = list(key.items()); random.Random(who + "h").shuffle(items)
        with open(hd / f"{who}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(ANN_COLS)
            for code, gid in items:
                g = next(x for x in gens if x["id"] == gid)
                w.writerow([code, g["src"], g["out"], "", "", "", "", "", ""])
    print(f"留出集 {len(key)} 条 → {hd}。标注时不要查看判官对这些条目的打分。")


# ---------------- judge ----------------
def cmd_judge(args):
    d = rdir(args); gens = load_gens(d, args)
    tuned = json.loads((d / "tuned.json").read_text(encoding="utf-8")) if args.tuned else None
    if args.tuned and tuned is None:
        sys.exit("请先运行 tune")
    out = d / ("judgments_tuned.jsonl" if args.tuned else "judgments.jsonl")
    done = {(r["gen"], r["judge"], r["dim"]) for r in read_jsonl(out)}
    for judge in C.MODELS:
        todo = [(g, dim) for g in gens for dim in C.DIMENSIONS if (g["id"], judge, dim) not in done]
        if not todo:
            continue
        print(f"[judge] {judge}: {len(todo)} 项")
        def work(x, judge=judge):
            g, dim = x
            vname = tuned["choice"][dim][judge]["variant"] if tuned else "base"
            spec = C.VARIANTS[dim][vname]
            st = C.STATE_TEMPLATE.format(src=g["src"], out=g["out"])
            v, cov = run_spec(judge, st, spec)
            append_jsonl(out, {"gen": g["id"], "judge": judge, "dim": dim, "value": v, "coverage": cov,
                               "variant": vname, "kind": spec["kind"]})
        pmap(work, todo, judge)
        J.unload(judge)
    print(f"→ {out}")


def judgments_file(d, args):
    return d / ("judgments_tuned.jsonl" if getattr(args, "tuned", False) else "judgments.jsonl")


def load_judgments(d, args=None):
    J_ = defaultdict(dict)   # (gen, judge) -> dim -> (value, cov, kind)
    for r in read_jsonl(judgments_file(d, args)):
        J_[(r["gen"], r["judge"])][r["dim"]] = (r["value"], r["coverage"],
                                                r.get("kind", C.DIMENSIONS[r["dim"]]["kind"]))
    if not J_:
        sys.exit(f"还没有判定结果（{judgments_file(d, args).name}），请先运行 judge")
    return J_


# ---------------- validate ----------------
def load_human(d, which="dev"):
    kf, af = ("annotation_key.json", "annotations") if which == "dev" else ("holdout_key.json", "holdout")
    if not (d / kf).exists():
        sys.exit(f"找不到 {kf}")
    key = json.loads((d / kf).read_text(encoding="utf-8"))
    files = sorted((d / af).glob("*.csv"))
    ann = {}   # annotator -> gen -> dim -> value
    for f in files:
        rows = read_csv_any(f); vals = {}
        for r in rows:
            gid = key.get(r["条目"].strip())
            if not gid:
                continue
            item = {}
            for col in ANN_COLS[3:8]:
                dim = col.split("(")[0]; s = (r.get(col) or "").strip()
                if s != "":
                    item[dim] = float(s)
            if item:
                vals[gid] = item
        if vals:
            ann[f.stem] = vals
    return ann


def metric_for(dim, pts):
    """pts: [(judge_raw, human_mean)]。返回 (指标名, 值, 通过?)。"""
    if C.DIMENSIONS[dim]["kind"] == "score":
        if len(pts) < 10:
            return "spearman", float("nan"), False
        rho = S.spearman([p[0] for p in pts], [p[1] for p in pts])
        return "spearman", rho, bool(rho >= C.MIN_SPEARMAN)
    pts = [(a, h) for a, h in pts if h != 0.5]
    ys = [1 if h > 0.5 else 0 for _, h in pts]
    if len(pts) < 10 or not 0 < sum(ys) < len(ys):
        return "AUC", float("nan"), False
    a = S.auc([p[0] for p in pts], ys)
    return "AUC", a, bool(a >= C.MIN_AUC)


def human_means(ann, dim):
    names = list(ann)
    gids = sorted({g for a in ann.values() for g, it in a.items() if dim in it})
    return {g: float(np.mean([ann[a][g][dim] for a in names if g in ann[a] and dim in ann[a][g]])) for g in gids}


# ---------------- tune ----------------
def cmd_tune(args):
    """在开发集（annotation_key + annotations/）上为每个 判官×维度 试所有提示词变体，选最优。"""
    d = rdir(args); ann = load_human(d, "dev")
    if not ann:
        sys.exit("开发集 annotations/ 没有已填写的标注")
    byid = {g["id"]: g for g in load_gens(d, args)}
    dev_ids = sorted({g for a in ann.values() for g in a})
    out = d / "tune_judgments.jsonl"
    done = {(r["gen"], r["judge"], r["dim"], r["variant"]) for r in read_jsonl(out)}
    for judge in C.MODELS:
        todo = [(g, dim, vn) for dim, vs in C.VARIANTS.items() for vn in vs for g in dev_ids
                if (g, judge, dim, vn) not in done]
        if not todo:
            continue
        print(f"[tune] {judge}: {len(todo)} 项")
        def work(x, judge=judge):
            g, dim, vn = x
            st = C.STATE_TEMPLATE.format(src=byid[g]["src"], out=byid[g]["out"])
            v, cov = run_spec(judge, st, C.VARIANTS[dim][vn])
            append_jsonl(out, {"gen": g, "judge": judge, "dim": dim, "variant": vn, "value": v, "coverage": cov})
        pmap(work, todo, judge, every=100)
        J.unload(judge)
    R = defaultdict(dict)
    for x in read_jsonl(out):
        R[(x["judge"], x["dim"], x["variant"])][x["gen"]] = (x["value"], x["coverage"])
    rep = ["# 调优报告（开发集）", "", "在开发集上为每个 判官×维度 选指标最高的提示词变体；并列时取排在前面的（更简单的）。",
           "**开发集上的数值因为经过挑选会偏乐观，最终以留出集验证为准。**", "",
           "| 维度 | 判官 | " + " | ".join(["变体：指标值"]) + " | 选中 |", "|---|---|---|---|"]
    choice = defaultdict(dict)
    for dim, vs in C.VARIANTS.items():
        hm = human_means(ann, dim)
        for judge in C.MODELS:
            scores = []
            for vn in vs:
                pts = [(R[(judge, dim, vn)][g][0], hm[g]) for g in hm
                       if g in R[(judge, dim, vn)] and R[(judge, dim, vn)][g][1] >= C.MIN_COVERAGE]
                m, val, ok = metric_for(dim, pts)
                scores.append((vn, m, val))
            best = max(scores, key=lambda x: (-1 if np.isnan(x[2]) else x[2]))
            choice[dim][judge] = {"variant": best[0], "metric": best[1], "dev_value": best[2]}
            cells = ", ".join(f"{vn}={val:.2f}" for vn, m, val in scores)
            rep.append(f"| {dim} | {judge} | {cells} | **{best[0]}** ({best[2]:.2f}) |")
    (d / "tuned.json").write_text(json.dumps({"choice": choice}, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "tune_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print("\n下一步：python run.py judge --run", args.run, "--tuned")


def cmd_validate(args):
    d = rdir(args); Jd = load_judgments(d, args); ann = load_human(d, args.set)
    if not ann:
        sys.exit(f"{args.set} 集没有已填写的标注表")
    names = list(ann)
    src = judgments_file(d, args).name
    report = ["# 判官验证报告", "", f"验证集：{args.set}　判定文件：{src}　标注者：{', '.join(names)}", ""]
    res = {"annotators": names, "set": args.set, "judgments": src, "dims": {}}
    report += ["| 维度 | 标注者间 α | 判官 | n | 指标 | 值 | 校准前→后 Brier (LOO) | 准入 |",
               "|---|---|---|---|---|---|---|---|"]
    for dim, dd in C.DIMENSIONS.items():
        hm = human_means(ann, dim)
        alpha = float("nan")
        if len(names) >= 2:
            gids = sorted(hm)
            alpha = S.kripp_alpha_interval([[ann[a].get(g, {}).get(dim, np.nan) for g in gids] for a in names])
        res["dims"][dim] = {"alpha": alpha, "judges": {}}
        for judge in C.MODELS:
            pts = [(Jd[(g, judge)][dim][0], hm[g]) for g in hm
                   if dim in Jd.get((g, judge), {}) and Jd[(g, judge)][dim][1] >= C.MIN_COVERAGE]
            m, v, ok = metric_for(dim, pts)
            info = {"n": len(pts), "metric": m, "value": v, "pass": ok}
            brier = ""
            if dd["kind"] == "noul" and not np.isnan(v):
                bp = [(a, 1 if h > 0.5 else 0) for a, h in pts if h != 0.5]
                info["platt"] = S.platt_fit([p[0] for p in bp], [p[1] for p in bp])
                b0, b1 = S.loo_brier([p[0] for p in bp], [p[1] for p in bp])
                info.update(brier_raw=b0, brier_cal=b1); brier = f"{b0:.3f}→{b1:.3f}"
            res["dims"][dim]["judges"][judge] = info
            report.append(f"| {dim} | {alpha:.2f} | {judge} | {info['n']} | {m} | {v:.2f} | {brier} | {'✓' if ok else '✗'} |")
    report += ["", f"门槛（预注册）：Spearman ≥ {C.MIN_SPEARMAN}，AUC ≥ {C.MIN_AUC}。",
               "标注者间 α 是人类一致性的上限参照：若 α 本身很低（< 0.4），说明判据定义需要先修订，而不是判官的问题。"]
    if args.tuned and args.set == "dev":
        report.append("\n**警告**：调优后的判官在开发集上验证是循环的，请用 --set holdout。")
    (d / "validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


# ---------------- analyze ----------------
def panel(val, dim):
    return [j for j, i in val["dims"][dim]["judges"].items() if i["pass"]]


def working_value(dim, judge, raw, kind, val):
    """统一工作尺度：noul 维度 = 校准后 logit；score 维度 = 0–1 比例（score 变体除以满分，noul 变体直接用概率）。"""
    dd = C.DIMENSIONS[dim]
    if dd["kind"] == "noul":
        ab = val["dims"][dim]["judges"][judge].get("platt", (1.0, 0.0))
        return float(S.logit(S.platt_apply([raw], ab))[0])
    return raw / (len(dd["levels"]) - 1) if kind == "score" else raw


def final01(dim, w):
    """工作尺度 → 0–1、越高越好。"""
    dd = C.DIMENSIONS[dim]
    if dd["kind"] == "noul":
        p = float(S.sigmoid(w))
        return 1 - p if dim == "condescending" else p
    x = min(max(w, 0.0), 1.0)
    return x if dd["higher_is_better"] else 1 - x


def adjust(d, gens, Jd, val, sent_resample=None):
    """返回 per-gen 调整后分量 comp[gid][dim] ∈[0,1]、per-gen per-judge J，及各维度判官效应。"""
    byid = {g["id"]: g for g in gens}
    effects, comp, perj = {}, defaultdict(dict), defaultdict(dict)
    for dim in C.DIMENSIONS:
        js = panel(val, dim)
        if not js:
            continue
        rows = []
        for g in gens:
            for j in js:
                r = Jd.get((g["id"], j), {}).get(dim)
                if r and r[1] >= C.MIN_COVERAGE:
                    rows.append((g["id"], j, working_value(dim, j, r[0], r[2], val)))
        use_self = len(js) >= 2        # 单判官时自偏好不可识别：保留自评格，但在报告中标注“未校正”
        y = [r[2] for r in rows]
        e = S.fit_effects(y, [byid[r[0]]["model"] for r in rows], [r[1] for r in rows],
                          [byid[r[0]]["sent"] for r in rows],
                          [byid[r[0]]["model"] == r[1] for r in rows], C.MODELS, js, use_self)
        effects[dim] = e
        acc = defaultdict(list)
        for gid, j, w in rows:
            adj = w - e["judge_offsets"][j] - e["self_beta"] * (byid[gid]["model"] == j)
            acc[gid].append((j, adj))
        for gid, lst in acc.items():
            comp[gid][dim] = final01(dim, float(np.mean([a for _, a in lst])))
            for j, a in lst:
                perj[gid].setdefault(j, {})[dim] = final01(dim, a)
    return comp, perj, effects


def J_of(c):
    core = [k for k, v in C.DIMENSIONS.items() if v["role"] == "core"]
    if not all(k in c for k in core):
        return None
    j = np.prod([c[k] for k in core])
    return float(j)


def check_val(val, d, args):
    if val.get("judgments") != judgments_file(d, args).name:
        sys.exit(f"validation.json 基于 {val.get('judgments')}，与当前判定文件不一致；请先用相同参数运行 validate")


def cmd_analyze(args):
    d = rdir(args); gens = load_gens(d, args); Jd = load_judgments(d, args)
    vp = d / "validation.json"
    if not vp.exists():
        sys.exit("请先完成人工标注并运行 validate")
    val = json.loads(vp.read_text(encoding="utf-8")); check_val(val, d, args)
    rep = ["# 分析报告", "", f"判官准入依据：{val.get('set')} 集；判定文件：{val.get('judgments')}", ""]
    missing = [k for k, v in C.DIMENSIONS.items() if v["role"] == "core" and not panel(val, k)]
    for dim in C.DIMENSIONS:
        rep.append(f"- {dim}：准入判官 {panel(val, dim) or '无'}")
    if missing:
        rep += ["", f"**负结果**：核心维度 {missing} 没有任何判官通过人工验证。",
                "在这些维度上，本地判官不能替代人工判断；按预注册，不计算 J，也不产出偏好对。"]
    comp, perj, eff = adjust(d, gens, Jd, val)
    rep += ["", "## 判官宽严（中心化偏移；score 维度为 0–1 比例，noul 维度为 logit）与自偏好 β_self", "",
            "| 维度 | " + " | ".join(C.MODELS) + " | β_self | β_self 95% CI |", "|---" * (len(C.MODELS) + 3) + "|"]
    # bootstrap：按句子重抽样
    rng = np.random.default_rng(C.SEED)
    sents = sorted({g["sent"] for g in gens})
    by_sent = defaultdict(list)
    for g in gens:
        by_sent[g["sent"]].append(g)
    boot_self = defaultdict(list); boot_mean = defaultdict(list); boot_diff = defaultdict(list)
    for b in range(C.BOOTSTRAP_REPS if not args.quick else 200):
        pick = rng.choice(sents, len(sents), replace=True)
        bg, bJ = [], {}
        for n, s in enumerate(pick):
            for g in by_sent[s]:
                ng = dict(g, id=f"{g['id']}#b{n}", sent=n)
                bg.append(ng)
                for j in C.MODELS:
                    if (g["id"], j) in Jd:
                        bJ[(ng["id"], j)] = Jd[(g["id"], j)]
        bc, _, be = adjust(d, bg, bJ, val)
        for dim, e in be.items():
            boot_self[dim].append(e["self_beta"])
        if not missing:
            means = {}
            for m in C.MODELS:
                v = [J_of(bc[g["id"]]) for g in bg if g["model"] == m and J_of(bc.get(g["id"], {})) is not None]
                means[m] = np.mean(v) if v else np.nan
                boot_mean[m].append(means[m])
            for a in C.MODELS:
                for c in C.MODELS:
                    if a < c:
                        boot_diff[(a, c)].append(means[a] - means[c])
    ci = lambda xs: (np.nanpercentile(xs, 2.5), np.nanpercentile(xs, 97.5))
    for dim, e in eff.items():
        offs = " | ".join(f"{e['judge_offsets'].get(m, float('nan')):+.2f}" if m in e["judge_offsets"] else "—"
                          for m in C.MODELS)
        if len(panel(val, dim)) < 2:
            rep.append(f"| {dim} | {offs} | n/a | 单判官，不可识别 |"); continue
        lo, hi = ci(boot_self[dim]) if boot_self[dim] else (np.nan, np.nan)
        rep.append(f"| {dim} | {offs} | {e['self_beta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    single = [dim for dim in eff if len(panel(val, dim)) == 1]
    if single:
        rep.append(f"\n**注意**：{single} 只有 1 个判官通过验证，自偏好无法估计，该判官对自己改写的分数未经校正。")
    rep.append("\nβ_self > 0 且置信区间不含 0，说明判官给自己的改写打了更高的分（Panickssery et al. 2024）。已从选手分数中扣除。")

    # 逐条输出
    rows = []
    for g in gens:
        c = comp.get(g["id"], {})
        rows.append({"id": g["id"], "句": g["sent"] + 1, "模型": g["model"], "原句": g["src"], "改写": g["out"],
                     **{k: round(c[k], 3) if k in c else "" for k in C.DIMENSIONS},
                     "J": round(J_of(c), 3) if J_of(c) is not None else ""})
    with open(d / "generations_scored.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    if not missing:
        rep += ["", "## 主指标 J = 去恨语 × 诚实 × 流畅（TextDetox 式联合分）", "",
                "| 模型 | J 均值 | 95% CI | " + " | ".join(C.DIMENSIONS) + " |", "|---" * (3 + len(C.DIMENSIONS)) + "|"]
        for m in C.MODELS:
            gs = [g for g in gens if g["model"] == m]
            jm = np.mean([J_of(comp[g["id"]]) for g in gs if J_of(comp.get(g["id"], {})) is not None])
            dims = " | ".join(f"{np.mean([comp[g['id']][k] for g in gs if k in comp.get(g['id'], {})]):.3f}"
                              if any(k in comp.get(g['id'], {}) for g in gs) else "—" for k in C.DIMENSIONS)
            lo, hi = ci(boot_mean[m])
            rep.append(f"| {m} | {jm:.3f} | [{lo:.3f}, {hi:.3f}] | {dims} |")
        rep += ["", "各维度已换算到 0–1、越高越好（hate_residue 列 = 1 − 残留；condescending 列 = 1 − 居高临下）。", "",
                "## 两两差异（J）", "", "| 比较 | 差值 95% CI | 结论 |", "|---|---|---|"]
        for (a, c), xs in boot_diff.items():
            lo, hi = ci(xs)
            verdict = "差异可信" if lo > 0 or hi < 0 else "无法区分"
            rep.append(f"| {a} − {c} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
    rep += ["", "## 诚实条款",
            "- 判官只在通过人工验证的维度上使用；验证样本量有限，结论只适用于本批句子与本批模型。",
            "- 判官宽严与自偏好的校正是线性近似（多面 Rasch 思路），不是完整的 Rasch 模型。",
            "- 所有结果是假设生成性质，不能推广到“PoL 爱语改写对真实人群有效”。"]
    (d / "analysis_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    (d / "_analysis_cache.json").write_text(json.dumps({"missing": missing}), encoding="utf-8")


# ---------------- pairs ----------------
def cmd_pairs(args):
    d = rdir(args); gens = load_gens(d, args); Jd = load_judgments(d, args)
    vp = d / "validation.json"
    if not vp.exists():
        sys.exit("请先运行 validate")
    val = json.loads(vp.read_text(encoding="utf-8")); check_val(val, d, args)
    if args.tuned and val.get("set") != "holdout":
        sys.exit("调优后的判官必须在留出集上验证（validate --tuned --set holdout）后才能产出偏好对")
    core = [k for k, v in C.DIMENSIONS.items() if v["role"] == "core"]
    short_panel = {k: panel(val, k) for k in core if len(panel(val, k)) < C.MIN_JUDGES_FOR_PAIRS}
    if short_panel:
        sys.exit(f"拒绝产出偏好对：以下核心维度通过验证的判官少于 {C.MIN_JUDGES_FOR_PAIRS} 个：{short_panel}")
    comp, perj, _ = adjust(d, gens, Jd, val)
    full = [j for j in C.MODELS if all(j in panel(val, k) for k in core)]
    by_sent = defaultdict(list)
    for g in gens:
        jv = J_of(comp.get(g["id"], {}))
        if jv is None or comp[g["id"]]["honesty"] < 0.5:     # 校准后的诚实概率 < 0.5：粉饰，不能当 chosen
            continue
        pj = [J_of(perj[g["id"]][j]) for j in full if j in perj.get(g["id"], {}) and J_of(perj[g["id"]][j]) is not None]
        se = np.std(pj, ddof=1) / math.sqrt(len(pj)) if len(pj) >= 2 else np.nan
        by_sent[g["sent"]].append((jv, se, g))
    n = 0
    with open(d / "preference_pairs.jsonl", "w", encoding="utf-8") as f:
        for s, lst in by_sent.items():
            if len(lst) < 2:
                continue
            lst.sort(key=lambda x: x[0])
            (jr, ser, gr), (jc, sec, gc) = lst[0], lst[-1]
            need = 2 * math.sqrt((ser or 0) ** 2 + (sec or 0) ** 2)
            if np.isnan(need) or jc - jr <= need:
                continue
            f.write(json.dumps({"prompt": gc["src"], "chosen": gc["out"], "rejected": gr["out"],
                                "chosen_model": gc["model"], "rejected_model": gr["model"],
                                "J_chosen": round(jc, 3), "J_rejected": round(jr, 3),
                                "margin": round(jc - jr, 3), "required_margin": round(need, 3)},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"偏好对 {n} 条 → {d / 'preference_pairs.jsonl'}（只保留分差 > 2 倍判官噪声的样本）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "generate", "annotate", "tune", "judge", "validate", "analyze", "pairs"])
    ap.add_argument("--run", default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="analyze 时只做 200 次 bootstrap")
    ap.add_argument("--tuned", action="store_true", help="使用 tune 选出的提示词变体（judgments_tuned.jsonl）")
    ap.add_argument("--set", default="dev", choices=["dev", "holdout"], help="validate 使用的标注集")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
