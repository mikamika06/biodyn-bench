import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
import numpy as np
from sim.grid import SPEC, matched
from sim.counts import pair_corr

name = sys.argv[1] if len(sys.argv) > 1 else "collider"
cfg = {"n_cells": 600, "n_struct": 2, "n_direct": 2, "n_ref": 2, "n_null": 2}
expr, pairs, k, tgt = matched(name, 0, cfg)
np.set_printoptions(precision=2, suppress=True, linewidth=140)
print("структура:", name, "|", SPEC[name]["asks"])
print("матриця клітини × гени:", expr.shape)
print("перші 4 клітини, усі гени:")
print(expr[:4])
print()
print("відповідь (індекси генів):")
for t, ps in pairs.items():
    if ps:
        print("  %s  %s" % (t, ps))
print()
print("кореляція між генами кожної пари (log-нормалізовані значення):")
for t, ps in pairs.items():
    for i, j in ps:
        print("  %s (%2d,%2d)  corr = %+.3f" % (t, i, j, pair_corr(expr, i, j)))
print()
print("D = пряме ребро x→y, S = пара під випробуванням у цій структурі, N = незалежні гени")
print("код структури: src/sim/grid.py, функція s_%s" % name)
