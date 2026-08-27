"""Порівняння нашої реалізації GENIE3/GRNBoost2 з еталонними матрицями arboreto."""
import json, pathlib, sys, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from eval.panels import get as get_panel
from eval.metrics import auroc
from methods.trees import genie3 as our_genie3, grnboost2 as our_grnboost2

expr, idx, pos, neg, cfg = get_panel("confounder", 0, "small")
p = expr.shape[1]
meta = json.load(open("out/ref_trees_meta.json"))
rank = lambda x: np.argsort(np.argsort(x))
iu = np.triu_indices(p, 1)


def score(w, tag, sec):
    g = lambda t: np.array([w[i, j] for i, j in idx[t]])
    return {"tag": tag, "main": auroc(g(pos), g(neg)),
            "control": auroc(g("D"), g("N")), "sec": round(sec, 1)}


res = []
print(f"панель {expr.shape}, {p} генів\n")
for nm, fn in (("GENIE3", our_genie3), ("GRNBoost2", our_grnboost2)):
    wr = np.load(f"out/ref_{nm}.npy")
    t = time.time()
    wo = fn(expr, seed=0)
    to = time.time() - t
    r1, r2 = score(wr, f"{nm} еталон", meta[nm]), score(wo, f"{nm} наш", to)
    a, b = wr[iu], wo[iu]
    r1["spearman_with_other"] = float(np.corrcoef(rank(a), rank(b))[0, 1])
    top = lambda x, k=100: set(np.argsort(-x)[:k])
    r1["top100_overlap"] = len(top(a) & top(b)) / 100
    r1["top20_overlap"] = len(top(a, 20) & top(b, 20)) / 20
    res += [r1, r2]
    print(f"{nm}")
    print(f"  еталон  main {r1['main']:.3f}  контроль {r1['control']:.3f}  {r1['sec']:.0f} с")
    print(f"  наш     main {r2['main']:.3f}  контроль {r2['control']:.3f}  {r2['sec']:.0f} с")
    print(f"  збіг: spearman {r1['spearman_with_other']:.3f}  "
          f"топ-100 {r1['top100_overlap']:.2f}  топ-20 {r1['top20_overlap']:.2f}\n")

json.dump(res, open("out/validate_trees.json", "w"), ensure_ascii=False, indent=1)
print("out/validate_trees.json")
