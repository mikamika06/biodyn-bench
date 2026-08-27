"""Ідентифіковність у шкалі AUROC: верхня межа на силу будь-якого методу,
що користується заданим набором статистик.

Замість безрозмірного відносного розбалансу рахуємо, яку AUROC дала б кожна
статистика при своєму залишковому розбалансі — тією самою формулою, що й
підлога стенда:  Φ(Δ / (σ·√2)).

Тоді:
  bound = max по незрівняних статистиках від |AUROC − 0.5|
  далі мінімум по тому, ЯКУ статистику зрівнювали
Це і є найкраще, чого може досягти метод на цьому наборі статистик.
"""
import argparse, json, pathlib, sys
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from math import erf, sqrt
from sim.grid import SPEC, matched
from sim.counts import pair_corr
from methods.marginal import pair_mi
from methods.conditional import partial_corr_matrix

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))


def per_pair(expr, pairs, stat):
    if stat == "corr":
        return np.array([abs(pair_corr(expr, i, j)) for i, j in pairs])
    if stat == "mi":
        return np.asarray(pair_mi(expr, pairs))
    if stat == "sq":
        x = expr - expr.mean(0)
        x = x / np.where(x.std(0) == 0, 1.0, x.std(0))
        q = x ** 2
        q = (q - q.mean(0)) / np.where(q.std(0) == 0, 1.0, q.std(0))
        return np.array([abs((q[:, i] * q[:, j]).mean()) for i, j in pairs])
    if stat == "partial":
        w = np.abs(partial_corr_matrix(expr))
        return np.array([w[i, j] for i, j in pairs])
    raise ValueError(stat)


ap = argparse.ArgumentParser()
ap.add_argument("--only", default="confounder,chain,and,feedback_dir")
ap.add_argument("--noises", default="gauss,laplace")
ap.add_argument("--stats", default="corr,mi,sq,partial")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--hide", action="store_true")
ap.add_argument("--out", default="ident_auroc.json")
a = ap.parse_args()

stats = a.stats.split(",")
dst = pathlib.Path("out") / a.out
rep = json.loads(dst.read_text()) if dst.exists() else {}

for st in a.only.split(","):
    sp = SPEC[st]
    for nd in a.noises.split(","):
        key = f"{st}|{nd}|{'hidden' if a.hide else 'visible'}"
        if key in rep:
            continue
        table = {}
        for s1 in stats:
            P = {s2: [] for s2 in stats}
            Q = {s2: [] for s2 in stats}
            for seed in range(a.seeds):
                e, p, k, t = matched(st, seed, stat=s1, noise_dist=nd, hide=a.hide)
                for s2 in stats:
                    P[s2].append(per_pair(e, p[sp["pos"]], s2))
                    Q[s2].append(per_pair(e, p[sp["neg"]], s2))
            row = {}
            for s2 in stats:
                x, y = np.concatenate(P[s2]), np.concatenate(Q[s2])
                d = x.mean() - y.mean()
                sd = 0.5 * (x.std() + y.std())
                row[s2] = float(Phi(d / (sd * sqrt(2)))) if sd > 0 else 0.5
            table[s1] = row
        # найкраще, чого може досягти метод: по кожному зрівнюванню беремо
        # найсильнішу незрівняну статистику, потім мінімум по зрівнюваннях
        bound = min(max(abs(table[s1][s2] - 0.5) for s2 in stats if s2 != s1)
                    for s1 in stats)
        rep[key] = {"table": table, "bound": float(bound),
                    "bound_auroc": float(0.5 + bound)}
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {st:14}{nd:9}{'прих' if a.hide else 'вид':>6}  "
              f"межа AUROC {0.5 + bound:.3f}", flush=True)

print(f"\n{'стовпець':14}{'шум':9}{'корінь':>7}{'межа AUROC':>12}")
for k, r in rep.items():
    st, nd, vis = k.split("|")
    print(f"{st:14}{nd:9}{vis[:4]:>7}{r['bound_auroc']:>12.3f}")
print("\nмежа = найкраща AUROC, доступна методу на цьому наборі статистик")
print(f"out/{a.out}")
