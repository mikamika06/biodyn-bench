"""Підлога стенду: залишкова похибка зрівнювання й AUROC, яку вона дає."""
import json, pathlib, sys
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from math import erf, sqrt
from eval.columns import COLUMNS, set_link
from sim.counts import pair_corr

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

print(f"{'стовпець':24}{'Δ середніх':>12}{'sd':>9}{'Δ/sd':>8}{'підлога':>10}")
print("-" * 63)
out = {}
for label, builder, _ in COLUMNS:
    P, Q = [], []
    for s in range(N):
        expr, idx, pos, neg = builder(s)
        P += [abs(pair_corr(expr, i, j)) for i, j in idx[pos]]
        Q += [abs(pair_corr(expr, i, j)) for i, j in idx[neg]]
    P, Q = np.array(P), np.array(Q)
    d = P.mean() - Q.mean()
    sd = 0.5 * (P.std() + Q.std())
    floor = Phi(d / (sd * sqrt(2)))
    out[label] = {"delta": float(d), "sd": float(sd), "floor": float(floor)}
    print(f"{label:24}{d:>+12.5f}{sd:>9.5f}{d/sd:>8.3f}{floor:>10.4f}")

pathlib.Path("out").mkdir(exist_ok=True)
json.dump(out, open("out/floor.json", "w"), ensure_ascii=False, indent=1)
print("\nпідлога = AUROC, якої досягне метод, що бачить ЛИШЕ силу звʼязку")
print("out/floor.json")
