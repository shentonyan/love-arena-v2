"""列出 gemma 与 qwen 判为最不诚实的 8 条改写及三个判官的打分。用法：python tools/examples.py [run名]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUN = "runs/" + (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1] in ("consensus", "random") else "flowtest")
import json, collections
V = collections.defaultdict(dict)
for l in open(RUN + "/judgments.jsonl", encoding="utf-8"):
    r = json.loads(l); V[r["gen"]][(r["judge"], r["dim"])] = r["value"]
G = {json.loads(l)["id"]: json.loads(l) for l in open(RUN + "/generations.jsonl", encoding="utf-8")}
J = ["gemma4:26b", "qwen2.5:7b", "llama3.1:8b"]
def h(g): return (V[g][(J[0], "honesty")] + V[g][(J[1], "honesty")]) / 2
for g in sorted(V, key=h)[:8]:
    print("-" * 60)
    print("SRC:", G[g]["src"])
    print("OUT:", G[g]["out"], " [" + G[g]["model"] + "]")
    print("honesty      ", "  ".join("%s=%.2f" % (j.split(":")[0], V[g][(j, "honesty")]) for j in J))
    print("hate_residue ", "  ".join("%s=%.2f" % (j.split(":")[0], V[g][(j, "hate_residue")]) for j in J))
