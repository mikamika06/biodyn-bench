"""Зведений звіт із усіх наявних JSON. Перезапускати після кожного етапу."""
import json, pathlib, sys
import numpy as np

O = pathlib.Path("out")
L = []
p = L.append


def load(n):
    f = O / n
    return json.loads(f.read_text()) if f.exists() else None


p("# Стенд: зведені результати\n")
p(f"Згенеровано `report_all.py`. Джерело — JSON у `out/`.\n")

# ── 1. класичні методи проти підлоги ────────────────────────────────────────
p("\n## 1. Класичні методи, сітка структур\n")
p("Кожне число — проти підлоги ТОГО САМОГО режиму. `*` = перевершує підлогу.\n")
for stat in ("corr", "mi"):
    for reg in ("linear", "linear_hidden", "linear_rho0.5", "tanh", "relu"):
        g, fl = load(f"grid_{stat}_{reg}.json"), load(f"floor_grid_{stat}_{reg}.json")
        if not g or not fl:
            continue
        ms = list(g[list(g)[0]]["methods"])
        p(f"\n### зрівняно за {stat}, режим {reg}\n")
        p("```")
        p(f"{'структура':14}{'треба':>7}{'підлога':>9}" + "".join(f"{m[:11]:>13}" for m in ms))
        for st, r in g.items():
            f = fl[st]["floor"]
            row = f"{st:14}{r['correct']:>7}{f:>9.3f}"
            for m in ms:
                v = r["methods"][m]["main"]
                good = abs(v - 0.5) < 0.06 if r["correct"] == "0.5" else v > f + 0.03
                row += f"{v:>12.3f}{'*' if good else ' '}"
            p(row)
        p("```")

# ── 2. подвійний критерій ───────────────────────────────────────────────────
dc = load("double_criterion.json")
if dc:
    p("\n## 2. Подвійний критерій\n```")
    ms = list(next(iter(dc.values()))["score"])
    p(f"{'структура':14}" + "".join(f"{m[:11]:>13}" for m in ms))
    tot = {m: 0 for m in ms}
    den = 0
    for st, v in dc.items():
        p(f"{st:14}" + "".join(f"{v['score'][m]:>13}" for m in ms))
        for m in ms:
            a, b = v["score"][m].split("/")
            tot[m] += int(a)
        den += int(b)
    p(f"{'РАЗОМ':14}" + "".join(f"{str(tot[m])+'/'+str(den):>13}" for m in ms))
    p("```")

# ── 3. звʼязний граф ────────────────────────────────────────────────────────
for f, tag in (("net.json", "адитивна кінетика"), ("net_hill.json", "кінетика Хілла"),
               ("net_rho.json", "адитивна, 50% пригнічення"),
               ("net_hill_rho.json", "Хілла, 50% пригнічення")):
    d = load(f)
    if not d:
        continue
    p(f"\n## 3. Звʼязний граф — {tag}\n")
    ms = list(next(iter(d.values()))["tests"][list(next(iter(d.values()))["tests"])[0]]["methods"])
    for t in ("D/CONF", "D/CHAIN", "COLL/N", "D/N"):
        rows = [(r["p_over_n"], r) for r in d.values() if t in r["tests"]]
        if not rows:
            continue
        p(f"\n### {t}   (треба {rows[0][1]['tests'][t]['correct']})\n```")
        p(f"{'генів':>7}{'клітин':>8}{'p/n':>7}{'пар':>6}{'підлога':>9}"
          + "".join(f"{m[:11]:>13}" for m in ms))
        for _, r in sorted(rows, key=lambda x: -x[0]):
            e = r["tests"][t]
            p(f"{r['genes']:>7}{r['cells']:>8}{r['p_over_n']:>7.3f}{e['n_pairs']:>6}"
              f"{e['floor']:>9.3f}" + "".join(f"{e['methods'][m]['main']:>13.3f}" for m in ms))
        p("```")

# ── 4. суміш типів ──────────────────────────────────────────────────────────
for f in ("mixture.json", "mix_2x2.json"):
    d = load(f)
    if not d:
        continue
    p(f"\n## 4. Суміш типів клітин — {f}\n")
    for t in ("D/CONF", "CORE/N"):
        rows = [r for r in d.values() if t in r["tests"]]
        if not rows:
            continue
        ms = list(rows[0]["tests"][t]["methods"])
        p(f"\n### {t}\n```")
        p(f"{'зсув':>6}{'типів':>7}{'режим':>10}{'підлога':>9}" + "".join(f"{m[:11]:>13}" for m in ms))
        for r in sorted(rows, key=lambda r: (-r.get("shift", 0.8), r["k_types"], r["mode"])):
            e = r["tests"][t]
            p(f"{r.get('shift', 0.8):>6.1f}{r['k_types']:>7}{r['mode']:>10}{e['floor']:>9.3f}"
              + "".join(f"{e['methods'][m]:>13.3f}" for m in ms))
        p("```")

# ── 5. глибина і різнорідність ──────────────────────────────────────────────
for f, tag in (("depth.json", "глибина зчитування"),
               ("depth_het.json", "різнорідність рівнів генів"),
               ("depth_iso.json", "різнорідність проти нулів")):
    d = load(f)
    if not d:
        continue
    p(f"\n## 5. {tag}\n```")
    p(f"{'структура':13}{'тег':>8}{'lib':>6}{'UMI':>8}{'λ сер.':>8}{'нулі':>7}"
      f"{'підлога':>9}{'частк.':>9}{'ARACNe':>9}")
    for k, r in sorted(d.items(), key=lambda kv: (kv[1]["structure"], -kv[1]["lib_loc"])):
        s = r["stats"]
        p(f"{r['structure']:13}{r.get('tag','')[:7]:>8}{r['lib_loc']:>6}"
          f"{s['umi_median']:>8.0f}{s['lambda_mean']:>8.2f}{s['zero_frac']:>7.2f}"
          f"{r['floor']:>9.3f}{r['methods']['часткова кор.']['main']:>9.3f}"
          f"{r['methods']['ARACNe']['main']:>9.3f}")
    p("```")

# ── 6. модель ───────────────────────────────────────────────────────────────
for f, tag in (("grid_model_s12.json", "сітка з еталонними парами, 6 зерен"),
               ("grid_model_p25.json", "сітка, панель 25"),
               ("grid_model.json", "сітка, панель 10 (стара, без пар R)"),
               ("net_model.json", "звʼязний граф")):
    d = load(f)
    if not d:
        continue
    p(f"\n## 6. Модель — {tag}\n```")
    if "net_model" in f:
        agg = {}
        for r in d.values():
            for t, e in r["tests"].items():
                agg.setdefault(t, []).append(e)
        p(f"{'тест':10}{'n':>4}{'підлога':>9}{'істина':>9}{'увага':>9}{'градієнт':>10}"
          f"{'проба':>8}{'в.увага':>9}{'в.град':>9}")
        for t, es in agg.items():
            mm = lambda fn: float(np.mean([fn(e) for e in es]))
            p(f"{t:10}{len(es):>4}{mm(lambda e: e['floor']):>9.3f}"
              f"{mm(lambda e: e['model_truth']):>9.3f}"
              f"{mm(lambda e: e['methods']['увага']['planted']):>9.3f}"
              f"{mm(lambda e: e['methods']['градієнт']['planted']):>10.3f}"
              f"{mm(lambda e: e['methods']['проба']['planted']):>8.3f}"
              f"{mm(lambda e: e['methods']['увага']['faith']):>9.3f}"
              f"{mm(lambda e: e['methods']['градієнт']['faith']):>9.3f}")
    else:
        agg = {}
        for k, r in d.items():
            agg.setdefault(k.split("|")[0], []).append(r)
        p(f"{'структура':13}{'n':>4}{'треба':>7}{'MSE':>8}{'стеля':>8}{'запас':>7}"
          f"{'істина':>9}{'увага':>8}{'градієнт':>10}{'проба':>8}{'в.увага':>9}{'в.град':>9}")
        for st, rs in agg.items():
            mm = lambda fn: float(np.mean([fn(r) for r in rs]))
            sd = lambda fn: float(np.std([fn(r) for r in rs]))
            marg = mm(lambda r: r["ceiling"] - r["val_mse"])
            p(f"{st:13}{len(rs):>4}{rs[0]['correct']:>7}"
              f"{mm(lambda r: r['val_mse']):>8.3f}{mm(lambda r: r['ceiling']):>8.3f}"
              f"{marg:>+7.3f}{mm(lambda r: r['model_truth']):>9.3f}"
              f"{mm(lambda r: r['methods']['увага']['planted']):>8.3f}"
              f"{mm(lambda r: r['methods']['градієнт']['planted']):>10.3f}"
              f"{mm(lambda r: r['methods']['проба']['planted']):>8.3f}"
              f"{mm(lambda r: r['methods']['увага']['faith']):>9.3f}"
              f"{mm(lambda r: r['methods']['градієнт']['faith']):>9.3f}")
    p("```")
    p("\n«запас» = стеля мінус MSE моделі. Якщо він менший за 0.02, модель")
    p("робить те саме, що лінійна регресія, і її внутрішній устрій")
    p("інтерпретувати не можна — там немає нелінійної структури.")

# ── 7. абляція ──────────────────────────────────────────────────────────────
d = load("ablation.json")
if d:
    p("\n## 7. Вичерпна абляція (задача 106)\n```")
    p(f"{'структура':13}{'база MSE':>10}{'база AUROC':>12}{'одиниць':>9}"
      f"{'відб/ΔMSE':>11}{'відб/причин':>13}{'ΔMSE/причин':>13}{'recall@1':>10}")
    for k, r in d.items():
        s = r.get("selection", {})
        p(f"{r['structure']:13}{r['base_mse']:>10.3f}{r['base_auroc']:>12.3f}"
          f"{r['n_units']:>9}{s.get('spearman_sel_vs_mse', float('nan')):>11.3f}"
          f"{s.get('spearman_sel_vs_causal', float('nan')):>13.3f}"
          f"{s.get('spearman_mse_vs_causal', float('nan')):>13.3f}"
          f"{s.get('recall_at_k', {}).get('1', float('nan')):>10.2f}")
    p("```")

# ── 8. дерева ───────────────────────────────────────────────────────────────
d = load("validate_trees.json")
if d:
    p("\n## 8. Звірка дерев з arboreto\n```")
    for r in d:
        p(f"{r['tag']:20} main {r['main']:.3f}  контроль {r['control']:.3f}  "
          f"{r.get('sec', 0):.0f} с" +
          (f"  spearman {r['spearman_with_other']:.3f}  "
           f"топ-100 {r['top100_overlap']:.2f}" if "spearman_with_other" in r else ""))
    p("```")

out = O / "RESULTS.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"out/RESULTS.md   {len(L)} рядків")
