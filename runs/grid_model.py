"""Дві істини на сітці структур.

Для кожної структури: навчити модель, виміряти чим вона КОРИСТУЄТЬСЯ (втручання
у вхід), і зіставити з тим, що методи інтерпретованості показують.

Три числа на клітинку:
  planted    AUROC методу проти ЗАКЛАДЕНОЇ структури даних
  model      AUROC ВТРУЧАННЯ проти закладеної — чи причинна сама модель
  faith      ранговий збіг методу з втручанням — чи вірний метод моделі
"""
import argparse, json, pathlib, sys, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np, torch
from sim.grid import SPEC, matched
from eval.metrics import auroc
from model.data import standardize
from model.interp import (attention_network, gradient_network,
                          intervention_effect, probe_effect)
from model.train import train, device, linear_ceiling

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--cells", type=int, default=512)
ap.add_argument("--only", default=None, help="кома-список структур")
ap.add_argument("--link", default="linear")
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--panel", type=int, default=25)
ap.add_argument("--d", type=int, default=192)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--out", default="grid_model.json")
a = ap.parse_args()

SMALL = {"n_cells": 5000, "n_struct": a.panel, "n_direct": a.panel,
         "n_ref": a.panel, "n_null": a.panel}
dev = device()
names = a.only.split(",") if a.only else list(SPEC)
dst = pathlib.Path("out") / a.out
report = json.loads(dst.read_text()) if dst.exists() else {}


def spearman(x, y):
    r = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(r(x), r(y))[0, 1])


for seed in range(a.seeds):
    for name in names:
        sp = SPEC[name]
        key = f"{name}|{seed}|{a.link}|{a.rho}|{a.panel}|{a.d}x{a.layers}"
        if key in report:
            print(f"  пропуск {key}", flush=True); continue
        t0 = time.time()
        expr, pairs, k, tgt = matched(name, seed, SMALL, link=a.link, rho=a.rho)
        ceil = linear_ceiling(expr)
        m, hist, _ = train(expr, steps=a.steps, d=a.d, n_layers=a.layers,
                           seed=seed, verbose=False, dev=dev)
        zs, _, _ = standardize(expr)

        types = [t for t in ("D", "S", "R", "N") if pairs.get(t)]
        plist = [(t, i, j) for t in types for i, j in pairs[t]]
        pl = [(i, j) for _, i, j in plist]

        eff = intervention_effect(m, zs, dev, pl, n_cells=a.cells, seed=seed)
        pr = probe_effect(m, zs, dev, pl, n_cells=min(1024, 2 * a.cells), seed=seed)
        nets = {"увага": attention_network(m, zs, dev, n_cells=a.cells, seed=seed),
                "градієнт": gradient_network(m, zs, dev, n_cells=a.cells, seed=seed)}

        by = lambda d: {t: np.array([d[(i, j)] for tt, i, j in plist if tt == t])
                        for t in types}
        eff_t, pr_t = by(eff), by(pr)
        ev = np.array([eff[(i, j)] for i, j in pl])

        row = {"k": k, "ceiling": ceil, "val_mse": min(h[2] for h in hist),
               "correct": sp["correct"], "pos": sp["pos"], "neg": sp["neg"],
               "model_truth": auroc(eff_t[sp["pos"]], eff_t[sp["neg"]]),
               "model_control": auroc(eff_t["D"], eff_t["N"]) if "N" in eff_t else None,
               "effects": {t: float(v.mean()) for t, v in eff_t.items()},
               "methods": {}}
        for nm, w in nets.items():
            g = lambda t: np.array([w[i, j] for i, j in pairs[t]])
            wv = np.array([w[i, j] for i, j in pl])
            row["methods"][nm] = {"planted": auroc(g(sp["pos"]), g(sp["neg"])),
                                  "control": auroc(g("D"), g("N")),
                                  "faith": spearman(wv, ev)}
        row["methods"]["проба"] = {
            "planted": auroc(pr_t[sp["pos"]], pr_t[sp["neg"]]),
            "control": auroc(pr_t["D"], pr_t["N"]),
            "faith": spearman(np.array([pr[(i, j)] for i, j in pl]), ev)}
        report[key] = row
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {name:12} зерно {seed}  істина {row['model_truth']:.3f}  "
              f"MSE {row['val_mse']:.3f}/{ceil:.3f}  ({time.time()-t0:.0f} с)", flush=True)

# бутстреп-інтервали по зернах
import collections
agg = collections.defaultdict(list)
for k, r in report.items():
    agg[k.split("|")[0]].append(r)


def ci(vals, n_boot=2000, seed=0):
    v = np.array(vals, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


print(f"\n{'структура':13}{'n':>3}{'треба':>7}{'істина [95% ДІ]':>26}"
      f"{'вірність уваги':>22}{'вірність градієнта':>24}")
summary = {}
for st, rs in agg.items():
    row = {}
    for nm, fn in (("model_truth", lambda r: r["model_truth"]),
                   ("faith_att", lambda r: r["methods"]["увага"]["faith"]),
                   ("faith_grad", lambda r: r["methods"]["градієнт"]["faith"]),
                   ("val_mse", lambda r: r["val_mse"]),
                   ("ceiling", lambda r: r["ceiling"])):
        row[nm] = ci([fn(r) for r in rs])
    summary[st] = {"n": len(rs), "correct": rs[0]["correct"], **row}
    f = lambda k: f"{row[k][0]:.3f} [{row[k][1]:.3f},{row[k][2]:.3f}]"
    print(f"{st:13}{len(rs):>3}{rs[0]['correct']:>7}{f('model_truth'):>26}"
          f"{f('faith_att'):>22}{f('faith_grad'):>24}")
(pathlib.Path("out") / a.out.replace(".json", "_ci.json")).write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nout/{a.out}  записів: {len(report)}")
