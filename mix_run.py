"""Суміш типів клітин: найчастіший реальний конфаундер.

Порівнює три режими:
  наївний    метод не знає про типи
  оракул     тип відомий, його вплив вилучено регресією (верхня межа)
  один тип   лише одна популяція (контроль без суміші)
"""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import numpy as np
from math import erf, sqrt
from sim.network import build_mixture
from eval.strength import matched_pairs
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix
from methods.marginal import corr_matrix, mi_matrix

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))

ap = argparse.ArgumentParser()
ap.add_argument("--genes", type=int, default=200)
ap.add_argument("--cells", type=int, default=3000)
ap.add_argument("--types", default="1,2,3,5,8")
ap.add_argument("--share", type=float, default=0.6)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--shift", type=float, default=0.8)
ap.add_argument("--shift_frac", type=float, default=1.0)
ap.add_argument("--out", default="mixture.json")
a = ap.parse_args()

METHODS = {"кореляція": lambda e, mi: np.abs(corr_matrix(e)),
           "взаємна інф.": lambda e, mi: mi,
           "часткова кор.": lambda e, mi: np.abs(partial_corr_matrix(e)),
           "ARACNe": lambda e, mi: aracne(e, mi=mi)}
TESTS = [("CORE", "CONF"), ("CORE", "N"), ("D", "CONF")]


def deconfound(expr, labels):
    """Вилучає середнє по типу з кожного гена — оракульна поправка."""
    out = expr.copy()
    for t in np.unique(labels):
        sel = labels == t
        out[sel] -= out[sel].mean(axis=0)
    return out


dst = pathlib.Path("out") / a.out
report = json.loads(dst.read_text()) if dst.exists() else {}

for kt in [int(x) for x in a.types.split(",")]:
    for mode in ("наївний", "оракул"):
        key = f"{kt}|{mode}|{a.share}|{a.genes}|{a.cells}|{a.shift_frac}|{a.shift}"
        if key in report or (kt == 1 and mode == "оракул"):
            continue
        t0, acc = time.time(), {}
        for s in range(a.seeds):
            expr, types, labels, parents = build_mixture(
                a.genes, a.cells, s, k_types=kt, share=a.share,
                shift_frac=a.shift_frac, shift=a.shift)
            e = deconfound(expr, labels) if mode == "оракул" else expr
            mi = mi_matrix(e)
            mats = {m: fn(e, mi) for m, fn in METHODS.items()}
            for pos, neg in TESTS:
                if len(types.get(pos, [])) < 10 or len(types.get(neg, [])) < 10:
                    continue
                pa, pb, d, sd, n = matched_pairs(e, types[pos], types[neg],
                                                 seed=s, cap=150)
                if n < 10:
                    continue
                t = f"{pos}/{neg}"
                acc.setdefault(t, {"floor": [], "n": [], "methods": {}})
                acc[t]["floor"].append(Phi(d / (sd * sqrt(2))) if sd > 0 else 0.5)
                acc[t]["n"].append(n)
                for m, w in mats.items():
                    acc[t]["methods"].setdefault(m, []).append(
                        auroc([w[i, j] for i, j in pa], [w[i, j] for i, j in pb]))
        report[key] = {"k_types": kt, "mode": mode, "shift_frac": a.shift_frac, "shift": a.shift, "sec": round(time.time() - t0, 1),
                       "tests": {t: {"floor": float(np.mean(v["floor"])),
                                     "n_pairs": int(np.mean(v["n"])),
                                     "methods": {m: float(np.mean(x))
                                                 for m, x in v["methods"].items()}}
                                 for t, v in acc.items()}}
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {kt} типів / {mode}  ({report[key]['sec']} с)", flush=True)

ms = list(METHODS)
for t in ("CORE/CONF", "CORE/N", "D/CONF"):
    print(f"\n### {t}")
    print(f"{'типів':>7}{'режим':>10}{'пар':>6}{'підлога':>9}" + "".join(f"{m[:11]:>13}" for m in ms))
    for k, r in sorted(report.items(), key=lambda kv: (kv[1]["k_types"], kv[1]["mode"])):
        if t not in r["tests"]:
            continue
        e = r["tests"][t]
        print(f"{r['k_types']:>7}{r['mode']:>10}{e['n_pairs']:>6}{e['floor']:>9.3f}"
              + "".join(f"{e['methods'][m]:>13.3f}" for m in ms))
print(f"\nout/{a.out}")
