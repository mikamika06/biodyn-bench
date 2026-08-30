import argparse, json, pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
import numpy as np
from sim.grid import SPEC, matched
from sim.grid2 import SPEC2, matched2
from eval.strength import r2_nodes, r2_central, strengths
from eval.metrics import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--panel", type=int, default=20)
ap.add_argument("--cells", type=int, default=2000)
a = ap.parse_args()
cfg = {"n_cells": a.cells, "n_struct": a.panel, "n_direct": a.panel, "n_ref": a.panel, "n_null": a.panel}


def stats(expr, pairs, sp):
    X = np.log1p(np.asarray(expr, dtype=float))
    r2 = r2_nodes(expr)
    D = pairs.get("D") or []
    v = X.var(0)
    vs = np.mean([v[j] > v[i] for i, j in D]) if D else float("nan")
    rs = np.mean([r2[j] > r2[i] for i, j in D]) if D else float("nan")
    pa, pb = pairs[sp["pos"]], pairs[sp["neg"]]
    ro = auroc(r2_central(expr, pa, r2), r2_central(expr, pb, r2))
    co = auroc(strengths(expr, pa), strengths(expr, pb))
    return vs, rs, ro, co


out = {}
print("%-16s %-4s %8s %8s %8s %8s" % ("структура", "gen", "varsort", "R2sort", "R2only", "corr"))
for name in SPEC2:
    for gen in ("v1", "v2"):
        if gen == "v1" and name not in SPEC:
            continue
        rows = []
        for seed in range(a.seeds):
            expr, pairs, k, tgt = (matched if gen == "v1" else matched2)(name, seed, cfg)
            rows.append(stats(expr, pairs, SPEC2[name]))
        m = np.nanmean(np.array(rows), axis=0)
        out["%s|%s" % (name, gen)] = m.tolist()
        print("%-16s %-4s %8.3f %8.3f %8.3f %8.3f" % (name, gen, *m))
pathlib.Path("out").mkdir(exist_ok=True)
pathlib.Path("out/gen2_check.json").write_text(json.dumps(out, indent=1))
