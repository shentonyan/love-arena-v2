"""诊断：每个判官每个维度的均值/标准差，以及判官两两之间的 Spearman 相关。用法：python tools/diagnose.py [run名]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUN = "runs/" + (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1] in ("consensus", "random") else "flowtest")
import json, collections, numpy as np, stats_utils as S
V = collections.defaultdict(dict)
for l in open(RUN + "/judgments.jsonl", encoding="utf-8"):
    r = json.loads(l); V[(r["judge"], r["dim"])][r["gen"]] = r["value"]
judges = ["gemma4:26b", "qwen2.5:7b", "llama3.1:8b"]
dims = ["hate_residue", "honesty", "fluency", "equal_connection", "condescending"]
for d in dims:
    print("==", d)
    for j in judges:
        x = np.array(list(V[(j, d)].values()))
        print("  %-12s mean=%.2f sd=%.2f min=%.2f max=%.2f" % (j, x.mean(), x.std(), x.min(), x.max()))
    for a in range(3):
        for b in range(a + 1, 3):
            g = sorted(set(V[(judges[a], d)]) & set(V[(judges[b], d)]))
            rho = S.spearman([V[(judges[a], d)][k] for k in g], [V[(judges[b], d)][k] for k in g])
            print("  rho(%s, %s) = %.2f" % (judges[a], judges[b], rho))
