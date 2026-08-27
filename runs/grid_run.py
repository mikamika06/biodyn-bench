"""Прогін методів по сітці причинних структур."""
import argparse, json, pathlib, sys, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from sim.grid import STRUCTURES, SPEC, matched, mean_abs_corr
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix
from methods.marginal import corr_matrix, mi_matrix
from methods.lingam import asymmetry_matrix
from methods.trees import genie3, grnboost2

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--link", default="linear")
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--hide", action="store_true")
ap.add_argument("--only", default=None)
ap.add_argument("--trees", action="store_true")
ap.add_argument("--noise", default="gauss", choices=["gauss","uniform","laplace","exp"])
ap.add_argument("--stat", default="corr", choices=["corr","mi"])
ap.add_argument("--out", default=None)
a = ap.parse_args()

METHODS = {
    "кореляція":     lambda e, mi: np.abs(corr_matrix(e)),
    "взаємна інф.":  lambda e, mi: mi,
    "часткова кор.": lambda e, mi: np.abs(partial_corr_matrix(e)),
    "ARACNe":        lambda e, mi: aracne(e, mi=mi),
    "LiNGAM-асим.":  lambda e, mi: asymmetry_matrix(e),
}
if a.trees:
    METHODS["GENIE3"] = lambda e, mi: genie3(e, seed=0)
    METHODS["GRNBoost2"] = lambda e, mi: grnboost2(e, seed=0)

report = {}
names = a.only.split(",") if a.only else list(SPEC)
for name in names:
    sp = SPEC[name]
    acc, t0 = {}, time.time()
    for s in range(a.seeds):
        expr, pairs, k, tgt = matched(name, s, hide=a.hide, link=a.link, rho=a.rho, stat=a.stat, noise_dist=a.noise)
        mi = mi_matrix(expr)
        for m, fn in METHODS.items():
            w = fn(expr, mi)
            g = lambda t: np.array([w[i, j] for i, j in pairs[t]])
            acc.setdefault(m, []).append(
                (auroc(g(sp["pos"]), g(sp["neg"])), auroc(g("D"), g("N"))))
    report[name] = {"correct": sp["correct"], "asks": sp["asks"], "k": k,
                    "pos": sp["pos"], "neg": sp["neg"], "sec": round(time.time() - t0, 1),
                    "methods": {m: {"main": float(np.mean([x[0] for x in v])),
                                    "sd": float(np.std([x[0] for x in v])),
                                    "control": float(np.mean([x[1] for x in v]))}
                                for m, v in acc.items()}}
    print(f"  {name:12} ({report[name]['sec']} с)", flush=True)

print()
ms = list(METHODS)
print(f"{'структура':12}{'метрика':>10}{'треба':>7}" + "".join(f"{m[:12]:>14}" for m in ms))
print("-" * (29 + 14 * len(ms)))
for name, r in report.items():
    row = f"{name:12}{r['pos']+' vs '+r['neg']:>10}{r['correct']:>7}"
    for m in ms:
        row += f"{r['methods'][m]['main']:>14.3f}"
    print(row)
print("\nконтроль D проти N:")
for name, r in report.items():
    print(f"  {name:12}" + "  ".join(f"{m[:9]} {r['methods'][m]['control']:.2f}" for m in ms))

out = pathlib.Path("out"); out.mkdir(exist_ok=True)
fn = a.out or f"grid_{'trees_' if a.trees else ''}{a.noise}_{a.stat}_{a.link}{'_hidden' if a.hide else ''}{'_rho'+str(a.rho) if a.rho else ''}.json"
prev = json.loads((out / fn).read_text()) if (out / fn).exists() else {}
prev.update(report)
(out / fn).write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nout/{fn}")
