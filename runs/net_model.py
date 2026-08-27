"""Дві істини у ЗВʼЯЗНОМУ графі — найсильніша версія задачі 104.

Досі модель тренувалась лише на сітці ізольованих модулів, де обумовлюватись
треба на трьох чистих генах. Тут те саме питання всередині мережі.
"""
import argparse, json, pathlib, sys, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np, torch
from math import erf, sqrt
from sim.network import build
from sim.hill import build_hill
from eval.strength import matched_pairs
from eval.metrics import auroc
from model.data import standardize
from model.interp import (attention_network, gradient_network,
                          intervention_effect, probe_effect)
from model.train import train, device, linear_ceiling

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))

ap = argparse.ArgumentParser()
ap.add_argument("--genes", type=int, default=150)
ap.add_argument("--cells", type=int, default=4000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--d", type=int, default=192)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--cap", type=int, default=40)
ap.add_argument("--icells", type=int, default=512)
ap.add_argument("--kinetics", default="additive")
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--out", default="net_model.json")
a = ap.parse_args()

TESTS = [("D", "CONF", "high"), ("D", "CHAIN", "high"), ("COLL", "N", "0.5")]
dev = device()
dst = pathlib.Path("out") / a.out
report = json.loads(dst.read_text()) if dst.exists() else {}
spearman = lambda x, y: float(np.corrcoef(np.argsort(np.argsort(x)),
                                          np.argsort(np.argsort(y)))[0, 1])

for seed in range(a.seeds):
    key = f"{a.genes}|{a.cells}|{seed}|{a.kinetics}|{a.rho}"
    if key in report:
        print(f"  пропуск {key}", flush=True); continue
    t0 = time.time()
    expr, types, parents, W = (
        build_hill(a.genes, a.cells, seed, rho=a.rho) if a.kinetics == "hill"
        else build(a.genes, a.cells, seed, rho=a.rho))
    ceil = linear_ceiling(expr)
    m, hist, _ = train(expr, steps=a.steps, d=a.d, n_layers=a.layers,
                       seed=seed, verbose=False, dev=dev)
    zs, _, _ = standardize(expr)

    sel, floors = {}, {}
    for pos, neg, corr in TESTS:
        if len(types[pos]) < 10 or len(types[neg]) < 10:
            continue
        pa, pb, d, sd, n = matched_pairs(expr, types[pos], types[neg],
                                         seed=seed, cap=a.cap)
        if n < 8:
            continue
        sel[f"{pos}/{neg}"] = (pa, pb, corr)
        floors[f"{pos}/{neg}"] = Phi(d / (sd * sqrt(2))) if sd > 0 else 0.5

    allp = sorted({p for pa, pb, _ in sel.values() for p in pa + pb})
    eff = intervention_effect(m, zs, dev, allp, n_cells=a.icells, seed=seed)
    pr = probe_effect(m, zs, dev, allp, n_cells=min(1024, 2 * a.icells), seed=seed)
    nets = {"увага": attention_network(m, zs, dev, n_cells=a.icells, seed=seed),
            "градієнт": gradient_network(m, zs, dev, n_cells=a.icells, seed=seed)}
    ev = np.array([eff[p] for p in allp])

    row = {"genes": a.genes, "cells": a.cells, "seed": seed, "kinetics": a.kinetics,
           "rho": a.rho, "ceiling": ceil, "val_mse": min(h[2] for h in hist),
           "sec": round(time.time() - t0, 1), "tests": {}}
    for t, (pa, pb, corr) in sel.items():
        e = {"correct": corr, "floor": floors[t], "n_pairs": len(pa),
             "model_truth": auroc([eff[p] for p in pa], [eff[p] for p in pb]),
             "effects": {"pos": float(np.mean([eff[p] for p in pa])),
                         "neg": float(np.mean([eff[p] for p in pb]))},
             "methods": {}}
        for nm, w in nets.items():
            e["methods"][nm] = {
                "planted": auroc([w[i, j] for i, j in pa], [w[i, j] for i, j in pb]),
                "faith": spearman(np.array([w[i, j] for i, j in allp]), ev)}
        e["methods"]["проба"] = {
            "planted": auroc([pr[p] for p in pa], [pr[p] for p in pb]),
            "faith": spearman(np.array([pr[p] for p in allp]), ev)}
        row["tests"][t] = e
    report[key] = row
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  зерно {seed}  MSE {row['val_mse']:.3f}/{ceil:.3f}  "
          + "  ".join(f"{t} істина {e['model_truth']:.3f}"
                      for t, e in row["tests"].items())
          + f"  ({row['sec']:.0f} с)", flush=True)

print(f"\n{'тест':10}{'підлога':>9}{'істина':>9}{'увага':>9}{'градієнт':>10}{'проба':>8}"
      f"{'в.увага':>9}{'в.град':>9}")
agg = {}
for r in report.values():
    for t, e in r["tests"].items():
        agg.setdefault(t, []).append(e)
for t, es in agg.items():
    mm = lambda f: float(np.mean([f(e) for e in es]))
    print(f"{t:10}{mm(lambda e: e['floor']):>9.3f}{mm(lambda e: e['model_truth']):>9.3f}"
          f"{mm(lambda e: e['methods']['увага']['planted']):>9.3f}"
          f"{mm(lambda e: e['methods']['градієнт']['planted']):>10.3f}"
          f"{mm(lambda e: e['methods']['проба']['planted']):>8.3f}"
          f"{mm(lambda e: e['methods']['увага']['faith']):>9.3f}"
          f"{mm(lambda e: e['methods']['градієнт']['faith']):>9.3f}")
print(f"\nout/{a.out}")
