"""仅用于测试工作流：把 runs/main 复制到 runs/flowtest，并用模拟值填写标注表（consensus=判官均值+噪声，random=纯随机）。结果没有研究意义。"""
import os, sys
import csv, json, random, shutil, sys
from pathlib import Path
mode = sys.argv[1] if len(sys.argv) > 1 else "consensus"  # consensus | random
src, dst = Path("runs/main"), Path("runs/flowtest")
if dst.exists():
    shutil.rmtree(dst)
(dst / "annotations").mkdir(parents=True)
for f in ["generations.jsonl", "judgments.jsonl", "annotation_key.json"]:
    shutil.copy(src / f, dst / f)
J = {}
for line in open(dst / "judgments.jsonl", encoding="utf-8"):
    r = json.loads(line)
    J.setdefault((r["gen"], r["dim"]), []).append(r["value"])
key = json.load(open(dst / "annotation_key.json", encoding="utf-8"))
dims = [("hate_residue", 3), ("honesty", 1), ("fluency", 3), ("equal_connection", 3), ("condescending", 1)]
def read(p):
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return list(csv.reader(open(p, newline="", encoding=enc)))
        except UnicodeDecodeError:
            pass
for who, seed in [("annotator_A", 1), ("annotator_B", 2)]:
    rng = random.Random(seed)
    rows = read(src / "annotations" / (who + ".csv"))
    for r in rows[1:]:
        gid = key[r[0].strip()]
        for col, (dim, top) in enumerate(dims, start=3):
            if mode == "random":
                v = rng.randint(0, top)
            else:
                vals = J[(gid, dim)]
                m = sum(vals) / len(vals)
                if top == 1:
                    v = int(m + rng.gauss(0, 0.15) > 0.5)
                else:
                    v = int(round(min(max(m + rng.gauss(0, 0.4), 0), top)))
            r[col] = str(v)
        r[8] = "SIMULATED-" + mode
    w = csv.writer(open(dst / "annotations" / (who + ".csv"), "w", newline="", encoding="utf-8-sig"))
    w.writerows(rows)
print("done ->", dst, "mode =", mode)
