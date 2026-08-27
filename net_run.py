"""Стенд на ЗВʼЯЗНОМУ scale-free графі: ті самі питання всередині мережі."""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import numpy as np
from math import erf, sqrt
from sim.network import build
from sim.hill import build_hill
from eval.strength import matched_pairs
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix, glasso_matrix
from methods.marginal import corr_matrix, mi_matrix
from methods.lingam import asymmetry_matrix

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))

ap = argparse.ArgumentParser()
ap.add_argument("--genes", default="300")
ap.add_argument("--cells", default="300,600,1200,2400,4800")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--link", default="linear")
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--m", type=int, default=2)
ap.add_argument("--alpha", type=float, default=0.6)
ap.add_argument("--noise", default="gauss", choices=["gauss","uniform","laplace","exp"])
ap.add_argument("--kinetics", default="additive", choices=["additive","hill"])
ap.add_argument("--out", default="net.json")
a = ap.parse_args()

METHODS = {
    "кореляція":     lambda e, mi: np.abs(corr_matrix(e)),
    "взаємна інф.":  lambda e, mi: mi,
    "часткова кор.": lambda e, mi: np.abs(partial_corr_matrix(e)),
    "ARACNe":        lambda e, mi: aracne(e, mi=mi),
    "glasso":        lambda e, mi: glasso_matrix(e),
    "LiNGAM":        lambda e, mi: asymmetry_matrix(e),
}
TESTS = [("D", "CONF", "high"), ("D", "CHAIN", "high"),
         ("COLL", "N", "0.5"), ("D", "N", "high")]

dst = pathlib.Path("out") / a.out
report = json.loads(dst.read_text()) if dst.exists() else {}

for g in [int(x) for x in a.genes.split(",")]:
    for c in [int(x) for x in a.cells.split(",")]:
        key = f"{g}|{c}|{a.link}|{a.rho}|{a.m}|{a.alpha}|{a.kinetics}|{a.noise}"
        if key in report:
            print(f"  пропуск {key}", flush=True); continue
        t0 = time.time()
        acc = {}
        for s in range(a.seeds):
            expr, pairs, parents, W = (
                build_hill(g, c, s, m=a.m, rho=a.rho, alpha=a.alpha)
                if a.kinetics == "hill" else
                build(g, c, s, m=a.m, link=a.link, rho=a.rho, alpha=a.alpha,
                      noise_dist=a.noise))
            mi = mi_matrix(expr)
            mats = {m: fn(expr, mi) for m, fn in METHODS.items()}
            for pos, neg, corr in TESTS:
                if not pairs[pos] or not pairs[neg]:
                    continue
                pa, pb, d, sd, n = matched_pairs(expr, pairs[pos], pairs[neg],
                                                 seed=s, cap=200)
                if n < 10:
                    continue
                fl = Phi(d / (sd * sqrt(2))) if sd > 0 else 0.5
                t = f"{pos}/{neg}"
                acc.setdefault(t, {"floor": [], "n": [], "methods": {}})
                acc[t]["floor"].append(fl); acc[t]["n"].append(n)
                for m, w in mats.items():
                    v = auroc([w[i, j] for i, j in pa], [w[i, j] for i, j in pb])
                    acc[t]["methods"].setdefault(m, []).append(v)
        report[key] = {"genes": g, "cells": c, "kinetics": a.kinetics, "noise": a.noise, "p_over_n": round(g / c, 4),
                       "sec": round(time.time() - t0, 1),
                       "tests": {t: {"correct": dict((f"{p}/{n}", cc) for p, n, cc in TESTS)[t],
                                     "floor": float(np.mean(v["floor"])),
                                     "n_pairs": int(np.mean(v["n"])),
                                     "methods": {m: {"main": float(np.mean(x)),
                                                     "sd": float(np.std(x))}
                                                 for m, x in v["methods"].items()}}
                                 for t, v in acc.items()}}
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {g} генів / {c} клітин  p/n={g/c:.3f}  ({report[key]['sec']} с)", flush=True)

ms = list(METHODS)
for t in ("D/CONF", "D/CHAIN", "COLL/N", "D/N"):
    print(f"\n### {t}")
    print(f"{'p/n':>7}{'пар':>7}{'підлога':>9}" + "".join(f"{m[:12]:>14}" for m in ms))
    for k, r in report.items():
        if t not in r["tests"]:
            continue
        e = r["tests"][t]
        print(f"{r['p_over_n']:>7.3f}{e['n_pairs']:>7}{e['floor']:>9.3f}"
              + "".join(f"{e['methods'][m]['main']:>14.3f}" for m in ms))
print(f"\nout/{a.out}")
