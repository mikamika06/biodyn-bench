"""Калькулятор режиму: чи витримають ці дані причинні твердження.

Бере СПРАВЖНІЙ набір лічильників, рахує три виміряні характеристики й
переводить їх у очікуване розрізнення за нашими виміряними кривими.

Характеристики (усі рахуються з сирих лічильників, без припущень про модель):
  CV_g   коефіцієнт варіації СЕРЕДНІХ рівнів генів — головна вісь
  p/n    гени поділити на клітини — друга вісь
  λ      середня кількість молекул на ген у клітині
"""
import argparse, json, pathlib, sys
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--counts", default="data/pbmc3k_counts.npy")
ap.add_argument("--genes", default="data/pbmc3k_genes.txt")
ap.add_argument("--out", default="regime.json")
a = ap.parse_args()

C = np.load(a.counts)
if C.shape[0] > C.shape[1]:
    pass
names = [l.strip() for l in open(a.genes)] if pathlib.Path(a.genes).exists() else None
print(f"матриця {C.shape}")

# ── криві, виміряні стендом ──────────────────────────────────────────────────
# виміряно depth_run.py, стовпець chain, 4 зерна; CV = 1/sqrt(shape)
HET = [(5.00, 0.478), (3.54, 0.459), (2.58, 0.492), (1.83, 0.534),
       (1.29, 0.543), (1.00, 0.565), (0.71, 0.714), (0.45, 0.799),
       (0.22, 0.894), (0.10, 0.922)]
HET_FLOOR = [(5.00, 0.453), (3.54, 0.431), (2.58, 0.435), (1.83, 0.493),
             (1.29, 0.478), (1.00, 0.490), (0.71, 0.560), (0.45, 0.566),
             (0.22, 0.578), (0.10, 0.530)]
PN = [(1.000, 0.575), (0.500, 0.700), (0.250, 0.786), (0.125, 0.824),
      (0.062, 0.851)]


def interp(curve, x, invert_x=False):
    xs = np.array([c[0] for c in curve]); ys = np.array([c[1] for c in curve])
    o = np.argsort(xs)
    return float(np.interp(np.clip(x, xs.min(), xs.max()), xs[o], ys[o]))


TF_HINT = ("STAT", "IRF", "NFKB", "JUN", "FOS", "MYC", "GATA", "TCF", "LEF",
           "RUNX", "SPI1", "PAX", "EGR", "KLF", "ATF", "CEBP", "REL", "ELF",
           "ETS", "ID2", "ID3", "BCL11", "IKZF", "ZEB", "SOX", "TBX", "EOMES")


def conditioning_gain(sub, max_genes=400, seed=0):
    """Скільки обумовлення додає ПОНАД парну кореляцію — на справжніх даних,
    без жодної істини.

    Якщо часткова кореляція це просто перемасштабована звичайна (ранговий
    збіг близько 1), то обумовлення не приносить нової інформації, і методи,
    що на нього спираються, працюватимуть як маргінальні.
    """
    import numpy as np
    from methods.conditional import partial_corr_matrix
    from methods.marginal import corr_matrix
    rng = np.random.default_rng(seed)
    g = sub.shape[1]
    idx = rng.choice(g, min(g, max_genes), replace=False) if g > max_genes else np.arange(g)
    x = np.log1p(sub[:, idx] / np.maximum(sub[:, idx].sum(1, keepdims=True), 1) * 1e4)
    keep = x.std(0) > 0
    x = x[:, keep]
    if x.shape[1] < 10:
        return float("nan")
    c = np.abs(corr_matrix(x))
    p_ = np.abs(partial_corr_matrix(x))
    iu = np.triu_indices(x.shape[1], 1)
    a, b = c[iu], p_[iu]
    rk = lambda v: np.argsort(np.argsort(v))
    return float(1.0 - abs(np.corrcoef(rk(a), rk(b))[0, 1]))


def panel(mask, label):
    sub = C[:, mask]
    if sub.shape[1] < 5:
        return None
    gm = sub.mean(axis=0)
    gm = gm[gm > 0]
    cv = float(gm.std() / gm.mean()) if len(gm) and gm.mean() > 0 else float("nan")
    n_cells, n_genes = sub.shape
    r = {"label": label, "genes": int(n_genes), "cells": int(n_cells),
         "cv_gene_means": cv, "p_over_n": n_genes / n_cells,
         "zero_frac": float((sub == 0).mean()),
         "lambda_mean": float(sub.mean()),
         "umi_median": float(np.median(sub.sum(axis=1))),
         "expected_by_het": interp(HET, cv),
         "floor_at_cv": interp(HET_FLOOR, cv),
         "expected_by_pn": interp(PN, n_genes / n_cells),
         "conditioning_gain": conditioning_gain(sub)}
    r["expected"] = min(r["expected_by_het"], r["expected_by_pn"])
    r["headroom"] = r["expected_by_het"] - r["floor_at_cv"]
    return r


det = (C > 0).sum(axis=0)
gm = C.mean(axis=0)
var = C.var(axis=0)
disp = np.divide(var, np.maximum(gm, 1e-9))
order_expr = np.argsort(-gm)
order_var = np.argsort(-disp)

panels = []
for n in (155, 500, 2000):
    m = np.zeros(C.shape[1], bool); m[order_expr[:n]] = True
    panels.append(panel(m, f"{n} найекспресованіших"))
    m = np.zeros(C.shape[1], bool); m[order_var[:n]] = True
    panels.append(panel(m, f"{n} найваріабельніших"))
m = det >= 3
panels.append(panel(m, "детектовані у >=3 клітинах"))
panels.append(panel(np.ones(C.shape[1], bool), "усі гени"))
if names:
    tf = np.array([any(h in g.upper() for h in TF_HINT) for g in names])
    panels.append(panel(tf, f"транскрипційні фактори ({tf.sum()})"))

panels = [p for p in panels if p]
print(f"\n{'панель':32}{'генів':>7}{'p/n':>7}{'CV_g':>7}{'нулі':>7}{'λ':>7}"
      f"{'за CV':>8}{'підлога':>9}{'запас':>8}{'за p/n':>8}{'обум.+':>8}")
print("-" * 104)
for r in sorted(panels, key=lambda r: -r["headroom"]):
    print(f"{r['label']:32}{r['genes']:>7}{r['p_over_n']:>7.3f}{r['cv_gene_means']:>7.2f}"
          f"{r['zero_frac']:>7.2f}{r['lambda_mean']:>7.2f}"
          f"{r['expected_by_het']:>8.3f}{r['floor_at_cv']:>9.3f}"
          f"{r['headroom']:>+8.3f}{r['expected_by_pn']:>8.3f}"
          f"{r['conditioning_gain']:>8.3f}")

print("\n«запас» = очікуване AUROC МІНУС підлога при тому самому CV.")
print("Це і є те, що метод може заробити структурою, а не величиною.")
print("Запас нижче +0.05 означає: дані не підтримують причинних тверджень")
print("незалежно від методу. Криві виміряні depth_run.py, CV 0.10-5.00.")
print("\n«обум.+» = 1 мінус ранговий збіг звичайної та часткової кореляції.")
print("Близько 0: обумовлення НЕ змінює ранжування, тобто нічого не додає —")
print("методи, що на нього спираються, працюватимуть як маргінальні.")
print("Високе значення НЕ гарантує користі: різниця може бути й шумом при")
print("поганій обумовленості. Це необхідна умова, не достатня.")
pathlib.Path("out").mkdir(exist_ok=True)
json.dump(panels, open(f"out/{a.out}", "w"), ensure_ascii=False, indent=1)
print(f"\nout/{a.out}")
