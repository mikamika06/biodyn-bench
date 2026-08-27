"""Чи передбачає міра ідентифіковності, що витягнуть методи й модель.

Зіставляє залишок із:
  - силою кожного класичного методу (|AUROC - підлога|)
  - розділенням, якого досягла НАВЧЕНА модель (втручання у вхід)
"""
import json, pathlib, sys
import numpy as np

O = pathlib.Path("out")
rk = lambda v: np.argsort(np.argsort(v))


def load(n):
    f = O / n
    return json.loads(f.read_text()) if f.exists() else None


ident = {}
for fn in ("identify3.json", "identify4.json"):
    d = load(fn)
    if not d:
        continue
    for k, r in d.items():
        if "per_stat" not in r:
            continue
        p = k.split("|")
        st, nd = p[0], p[1]
        vis = p[3] if len(p) > 3 else "visible"
        stats = p[2] if len(p) > 2 else "corr,mi,sq"
        ident[(st, nd, vis, "partial" in stats)] = r["residual"]

rows = []
for (st, nd, vis, has_partial), res in ident.items():
    if vis != "visible":
        continue
    g, f = load(f"grid_{nd}_corr_linear.json"), load(f"floor_grid_{nd}_corr_linear.json")
    if not g or not f or st not in g or st not in f:
        continue
    fl = f[st]["floor"]
    pw = {m: abs(v["main"] - fl) for m, v in g[st]["methods"].items()}
    rows.append({"st": st, "nd": nd, "partial": has_partial, "res": res, "pw": pw})

if not rows:
    raise SystemExit("немає даних")
methods = list(rows[0]["pw"])
for tag, sel in (("три парні статистики", False), ("з обумовленою статистикою", True)):
    sub = [r for r in sel and [x for x in rows if x["partial"]] or
           [x for x in rows if not x["partial"]]]
    if len(sub) < 4:
        continue
    a = np.array([r["res"] for r in sub])
    print(f"\n### {tag}   n={len(sub)}")
    print(f"{'метод':17}{'spearman':>10}{'pearson':>10}")
    for m in methods:
        b = np.array([r["pw"].get(m, np.nan) for r in sub])
        ok = ~np.isnan(b)
        if ok.sum() < 4:
            continue
        print(f"  {m:15}{np.corrcoef(rk(a[ok]), rk(b[ok]))[0, 1]:>+10.3f}"
              f"{np.corrcoef(a[ok], b[ok])[0, 1]:>+10.3f}")

# модель
md = load("grid_model_s12.json")
if md:
    agg = {}
    for k, r in md.items():
        agg.setdefault(k.split("|")[0], []).append(r["model_truth"])
    print(f"\n### модель проти залишку (шум gauss, три парні статистики)")
    print(f"{'структура':14}{'залишок':>10}{'істина моделі':>15}{'n':>4}")
    pairs = []
    for st, v in agg.items():
        res = ident.get((st, "gauss", "visible", False))
        if res is None:
            continue
        mt = float(np.mean(v))
        pairs.append((res, mt))
        print(f"{st:14}{res:>10.4f}{mt:>15.3f}{len(v):>4}")
    if len(pairs) >= 4:
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        print(f"\nspearman {np.corrcoef(rk(a), rk(b))[0, 1]:+.3f}   n={len(pairs)}")
