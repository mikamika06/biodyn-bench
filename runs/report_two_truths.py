import json, pathlib, sys, os, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
import numpy as np

import sys as _sys
SRC = pathlib.Path(_sys.argv[1] if len(_sys.argv) > 1 else "out/grid_model_two_truths.json")
DST = pathlib.Path(_sys.argv[2] if len(_sys.argv) > 2 else "out/RESULTS-two-truths.md")


def ci(vals, n_boot=2000, seed=0):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return None
    if len(v) < 2:
        return (float(v.mean()), float("nan"), float("nan"), len(v))
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return (float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(v))


def f(c):
    if c is None:
        return "—"
    m, lo, hi, n = c
    if np.isnan(lo):
        return "%.3f (n=%d)" % (m, n)
    return "%.3f [%.3f,%.3f]" % (m, lo, hi)


def main():
    if not SRC.exists():
        print("немає", SRC)
        return 1
    rep = json.loads(SRC.read_text())
    trained = collections.defaultdict(list)
    random_ = collections.defaultdict(list)
    for k, r in rep.items():
        st = k.split("|")[0]
        (random_ if k.endswith("|random") else trained)[st].append(r)
    structs = sorted(set(trained) | set(random_), key=lambda s: s)
    L = []
    A = L.append
    A("# Дві істини на одному стенді: навчена модель проти випадкової\n")
    A("Джерело: `out/grid_model_two_truths.json`. Кожне число — середнє по зернах з 95% бутстреп-інтервалом по зернах.\n")
    A("## 1. Істина моделі: чи модель причинна, і чи це не артефакт зчитувача\n")
    A("```")
    A("%-13s %-6s %-26s %-26s %-26s" % ("структура", "треба", "істина, навчена", "істина, випадкова", "val_mse навч / стеля"))
    for st in structs:
        tr, rd = trained.get(st, []), random_.get(st, [])
        need = tr[0]["correct"] if tr else (rd[0]["correct"] if rd else "")
        mse = ci([r["val_mse"] for r in tr]); ceil = ci([r["ceiling"] for r in tr])
        A("%-13s %-6s %-26s %-26s %s / %s" % (st, need, f(ci([r["model_truth"] for r in tr])), f(ci([r["model_truth"] for r in rd])),
                                             f(mse), f(ceil)))
    A("```\n")
    A("## 1b. Бейзлайни на роль-зрівняних парах (де neg = M)\n")
    A("```")
    A("%-13s %-10s %-12s %-12s" % ("структура", "n пар", "corr-only", "R²-only"))
    for st in structs:
        rows = [r["baseline_m"] for r in trained.get(st, []) if r.get("baseline_m")]
        if not rows:
            continue
        A("%-13s %-10.0f %-12.3f %-12.3f" % (st, np.mean([x["n_matched"] for x in rows]),
                                             np.mean([x["corr_only"] for x in rows]), np.mean([x["r2_only"] for x in rows])))
    A("```\n")
    A("## 2. Методи проти закладеної істини: навчена / випадкова\n")
    for nm in ("увага", "градієнт", "проба"):
        A("### %s\n" % nm)
        A("```")
        A("%-13s %-26s %-26s %-26s" % ("структура", "planted навчена", "planted випадкова", "faith навчена"))
        for st in structs:
            tr, rd = trained.get(st, []), random_.get(st, [])
            g = lambda rows, key: ci([r["methods"][nm][key] for r in rows if nm in r.get("methods", {})])
            A("%-13s %-26s %-26s %-26s" % (st, f(g(tr, "planted")), f(g(rd, "planted")), f(g(tr, "faith"))))
        A("```\n")
    A("## 3. Control task (Hewitt & Liang): проба на справжнє джерело проти випадкового гена\n")
    A("```")
    A("%-13s %-26s %-26s %-12s %-12s" % ("структура", "selectivity planted", "control-only planted", "R² true", "R² control"))
    for st in structs:
        tr = trained.get(st, [])
        rows = [r["methods"]["проба"]["control_task"] for r in tr if "control_task" in r.get("methods", {}).get("проба", {})]
        if not rows:
            continue
        A("%-13s %-26s %-26s %-12s %-12s" % (st, f(ci([x["selectivity_planted"] for x in rows])), f(ci([x["control_planted"] for x in rows])),
                                            "%.3f" % np.mean([x["mean_true"] for x in rows]), "%.3f" % np.mean([x["mean_control"] for x in rows])))
    A("```\n")
    A("## 4. Каскадна рандомізація ваг (Adebayo): що лишається від карти, коли ваги випадкові\n")
    stages = None
    for st in structs:
        for r in trained.get(st, []):
            if r.get("cascade"):
                stages = [c["label"] for c in r["cascade"]]
                break
        if stages:
            break
    if stages:
        A("```")
        A("%-13s " % "структура" + " ".join("%-22s" % s for s in stages))
        for st in structs:
            rows = [r["cascade"] for r in trained.get(st, []) if r.get("cascade")]
            if not rows:
                continue
            cells = []
            for si in range(len(stages)):
                a = ci([row[si]["att_corr"] for row in rows]); g = ci([row[si]["grad_corr"] for row in rows])
                cells.append("ув %.2f гр %.2f" % (a[0], g[0]) if a and g else "—")
            A("%-13s " % st + " ".join("%-22s" % c for c in cells))
        A("```")
        A("Число = Spearman карти після рандомізації з картою оригіналу по всіх парах; 1.0 = карта не змінилась.\n")
        A("```")
        A("%-13s " % "planted після" + " ".join("%-22s" % s for s in stages))
        for st in structs:
            rows = [r["cascade"] for r in trained.get(st, []) if r.get("cascade")]
            if not rows:
                continue
            cells = []
            for si in range(len(stages)):
                a = ci([row[si]["att_planted"] for row in rows]); g = ci([row[si]["grad_planted"] for row in rows])
                cells.append("ув %.2f гр %.2f" % (a[0], g[0]) if a and g else "—")
            A("%-13s " % st + " ".join("%-22s" % c for c in cells))
        A("```\n")
    A("## 5. Проба по шарах residual stream: де читається ребро\n")
    A("```")
    A("%-13s %-8s %-10s %-10s %-22s %-22s" % ("структура", "шар", "R² true", "R² ctrl", "true planted", "selectivity planted"))
    for st in structs:
        rows = [r["probe_layers"] for r in trained.get(st, []) if r.get("probe_layers")]
        if not rows:
            continue
        for li in range(len(rows[0])):
            A("%-13s %-8d %-10.3f %-10.3f %-22s %-22s" % (st, li, np.mean([row[li]["mean_true"] for row in rows]), np.mean([row[li]["mean_control"] for row in rows]),
                                                       f(ci([row[li]["true_planted"] for row in rows])), f(ci([row[li]["selectivity_planted"] for row in rows]))))
    A("```\n")
    A("## 6. Зрівнювання за R² центрального вузла\n")
    A("```")
    A("%-13s %-6s %-26s %-26s %-26s" % ("структура", "n", "істина моделі", "увага planted", "градієнт planted"))
    for st in structs:
        rows = [r["r2_matched"] for r in trained.get(st, []) if r.get("r2_matched") and r["r2_matched"].get("n", 0) >= 8]
        if not rows:
            A("%-13s %-6s %s" % (st, "0", "недостатньо спільних пар за R²"))
            continue
        A("%-13s %-6d %-26s %-26s %-26s" % (st, int(np.mean([x["n"] for x in rows])), f(ci([x.get("model_truth") for x in rows])),
                                             f(ci([x.get("увага") for x in rows])), f(ci([x.get("градієнт") for x in rows]))))
    A("```\n")
    DST.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("->", DST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
