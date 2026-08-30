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
from sim.grid2 import SPEC2, matched2
from eval.metrics import auroc
from model.data import standardize
from model.interp import (attention_network, gradient_network,
                          intervention_effect, probe_effect, probe_control,
                          cascade_randomize, probe_layers)
from eval.strength import matched_pairs_r2, match_subsets, strengths, r2_nodes, r2_central
from model.train import train, device, linear_ceiling, validate
from model.transformer import GeneTransformer
from model.data import CellDataset, split

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
ap.add_argument("--out", default=None)
ap.add_argument("--control", action="store_true")
ap.add_argument("--control-task", action="store_true")
ap.add_argument("--cascade", action="store_true")
ap.add_argument("--probe-layers", action="store_true")
ap.add_argument("--match-r2", action="store_true")
ap.add_argument("--gen", default="v1", choices=["v1", "v2"])
a = ap.parse_args()
if a.out is None:
    a.out = "grid_model_control.json" if a.control else "grid_model.json"

SMALL = {"n_cells": 5000, "n_struct": a.panel, "n_direct": a.panel,
         "n_ref": a.panel, "n_null": a.panel}
dev = device()
if a.gen == "v2":
    SPEC = SPEC2
    matched = matched2
names = a.only.split(",") if a.only else list(SPEC)
dst = pathlib.Path("out") / a.out
report = json.loads(dst.read_text()) if dst.exists() else {}


def spearman(x, y):
    r = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(r(x), r(y))[0, 1])


for seed in range(a.seeds):
    for name in names:
        sp = SPEC[name]
        key = f"{name}|{seed}|{a.link}|{a.rho}|{a.panel}|{a.d}x{a.layers}" + ("|v2" if a.gen == "v2" else "") + ("|random" if a.control else "")
        if key in report:
            print(f"  пропуск {key}", flush=True); continue
        t0 = time.time()
        expr, pairs, k, tgt = matched(name, seed, SMALL, link=a.link, rho=a.rho)
        ceil = linear_ceiling(expr)
        if a.control:
            torch.manual_seed(seed)
            m = GeneTransformer(expr.shape[1], a.d, a.layers).to(dev)
            zs, mu, sd = standardize(expr)
            _, te = split(expr, seed=seed)
            dlv = torch.utils.data.DataLoader(CellDataset((te - mu) / sd, 0.15, seed + 1), batch_size=128)
            hist = [(0, float("nan"), validate(m, dlv, dev))]
        else:
            m, hist, _ = train(expr, steps=a.steps, d=a.d, n_layers=a.layers,
                               seed=seed, verbose=False, dev=dev)
            zs, _, _ = standardize(expr)

        if SPEC[name].get("neg") == "M" and pairs.get("M"):
            sa, sb = strengths(expr, pairs["S"]), strengths(expr, pairs["M"])
            ia, ib = match_subsets(sa, sb, np.random.default_rng(seed), bins=6)
            if len(ia) >= 6:
                pairs["S"] = [pairs["S"][i] for i in ia]
                pairs["M"] = [pairs["M"][i] for i in ib]
            r2n = r2_nodes(expr)
            baseline = {"n_matched": int(len(pairs["S"])),
                        "corr_only": auroc(strengths(expr, pairs["S"]), strengths(expr, pairs["M"])),
                        "r2_only": auroc(r2_central(expr, pairs["S"], r2n), r2_central(expr, pairs["M"], r2n))}
        else:
            baseline = None
        types = [t for t in ("D", "S", "R", "N", "M") if pairs.get(t)]
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
               "baseline_m": baseline,
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
        if a.match_r2:
            pa, pb, st = matched_pairs_r2(expr, pairs[sp["pos"]], pairs[sp["neg"]], seed=seed)
            row["r2_matched"] = {"n": st.get("n", 0), "corr_gap": st.get("corr_gap"), "r2_gap": st.get("r2_gap")}
            if st.get("n", 0) >= 8:
                ea = np.array([eff[p] for p in pa]); eb = np.array([eff[p] for p in pb])
                row["r2_matched"]["model_truth"] = auroc(ea, eb)
                for nm, w in nets.items():
                    row["r2_matched"][nm] = auroc(np.array([w[i, j] for i, j in pa]), np.array([w[i, j] for i, j in pb]))
                row["r2_matched"]["проба"] = auroc(np.array([pr[p] for p in pa]), np.array([pr[p] for p in pb]))
        if a.cascade:
            stages = cascade_randomize(m, zs, dev, n_cells=a.cells, seed=seed)
            base_att = np.array([stages[0]["att"][i, j] for i, j in pl])
            base_grad = np.array([stages[0]["grad"][i, j] for i, j in pl])
            row["cascade"] = []
            for stg in stages:
                av = np.array([stg["att"][i, j] for i, j in pl]); gv = np.array([stg["grad"][i, j] for i, j in pl])
                ga = lambda w, t: np.array([w[i, j] for i, j in pairs[t]])
                row["cascade"].append({"stage": stg["stage"], "label": stg["label"],
                                       "att_corr": spearman(av, base_att), "grad_corr": spearman(gv, base_grad),
                                       "att_planted": auroc(ga(stg["att"], sp["pos"]), ga(stg["att"], sp["neg"])),
                                       "grad_planted": auroc(ga(stg["grad"], sp["pos"]), ga(stg["grad"], sp["neg"]))})
        if a.probe_layers:
            plr = probe_layers(m, zs, dev, pl, n_cells=min(1024, 2 * a.cells), seed=seed)
            n_l = len(next(iter(plr.values()))["true"])
            row["probe_layers"] = []
            for li in range(n_l):
                sel = {p: v["true"][li] - v["control"][li] for p, v in plr.items()}
                tru = {p: v["true"][li] for p, v in plr.items()}
                row["probe_layers"].append({"layer": li,
                                            "mean_true": float(np.mean([v["true"][li] for v in plr.values()])),
                                            "mean_control": float(np.mean([v["control"][li] for v in plr.values()])),
                                            "true_planted": auroc(by(tru)[sp["pos"]], by(tru)[sp["neg"]]),
                                            "selectivity_planted": auroc(by(sel)[sp["pos"]], by(sel)[sp["neg"]])})
        if a.control_task:
            pc = probe_control(m, zs, dev, pl, n_cells=min(1024, 2 * a.cells), seed=seed)
            sel = {(i, j): v["selectivity"] for (i, j), v in pc.items()}
            ctrl = {(i, j): v["control"] for (i, j), v in pc.items()}
            sel_t = by(sel)
            row["methods"]["проба"]["control_task"] = {
                "selectivity_planted": auroc(sel_t[sp["pos"]], sel_t[sp["neg"]]),
                "mean_true": float(np.mean([v["true"] for v in pc.values()])),
                "mean_control": float(np.mean([v["control"] for v in pc.values()])),
                "mean_selectivity": float(np.mean(list(sel.values()))),
                "control_planted": auroc(by(ctrl)[sp["pos"]], by(ctrl)[sp["neg"]])}
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
