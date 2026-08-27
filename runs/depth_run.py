"""Розгортка по ГЛИБИНІ ЗЧИТУВАННЯ і λ, а не по частці нулів.

Частка нулів — похідна величина й залежить від нашої ручки роздуття, існування
якої в полі заперечується (Svensson 2020; Sarkar & Stephens 2021). λ (очікувана
кількість молекул на ген у клітині) і глибина бібліотеки — вимірювані в будь-
якому наборі, тому твердження стає механізм-агностичним.
"""
import argparse, json, pathlib, sys, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from math import erf, sqrt
from sim.grid import SPEC, matched, build, tune, tune_ref
from sim.counts import pair_corr
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix
from methods.marginal import corr_matrix, mi_matrix

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))

ap = argparse.ArgumentParser()
ap.add_argument("--libs", default="9.0,8.0,7.0,6.14,5.0,4.0,3.0")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--only", default="confounder,chain,collider,two_hidden")
ap.add_argument("--link", default="linear")
ap.add_argument("--kw", default="{}", help="перекриття PARAMS у JSON")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="depth.json")
a = ap.parse_args()

METHODS = {
    "кореляція":     lambda e, mi: np.abs(corr_matrix(e)),
    "взаємна інф.":  lambda e, mi: mi,
    "часткова кор.": lambda e, mi: np.abs(partial_corr_matrix(e)),
    "ARACNe":        lambda e, mi: aracne(e, mi=mi),
}

dst = pathlib.Path("out") / a.out
report = json.loads(dst.read_text()) if dst.exists() else {}


def counts_stats(structure, seed, kw, k, kr):
    """Вимірні характеристики зчитування: λ на ген, UMI на клітину, детекція."""
    from sim.counts import make_counts
    from sim.grid import Panel, STRUCTURES, DEFAULT
    rng = np.random.default_rng(seed)
    p = Panel(DEFAULT["n_cells"], rng, a.link, 0.0)
    fn = STRUCTURES[structure]
    for _ in range(DEFAULT["n_struct"]):
        fn(p, k, False)
    for _ in range(DEFAULT["n_direct"]):
        x = p.new(); y = k * p.sign() * p.f(x) + p.eps()
        p.pairs["D"].append((p.add(x), p.add(y)))
    z, _, _ = p.finish()
    c = make_counts(z, rng, kw)
    return {"umi_median": float(np.median(c.sum(1))),
            "lambda_median": float(np.median(c[c > 0])) if (c > 0).any() else 0.0,
            "lambda_mean": float(c.mean()),
            "detected_median": float(np.median((c > 0).sum(1))),
            "zero_frac": float((c == 0).mean())}


for lib in [float(x) for x in a.libs.split(",")]:
    kw = {"lib_loc": lib, **json.loads(a.kw)}
    for name in a.only.split(","):
        sp = SPEC[name]
        key = f"{name}|{lib}|{a.link}|{a.tag}"
        if key in report:
            continue
        t0, acc = time.time(), {}
        P, Q = [], []
        for s in range(a.seeds):
            expr, pairs, k, tgt = matched(name, s, link=a.link, counts_kw=kw)
            mi = mi_matrix(expr)
            P += [abs(pair_corr(expr, i, j)) for i, j in pairs[sp["pos"]]]
            Q += [abs(pair_corr(expr, i, j)) for i, j in pairs[sp["neg"]]]
            for m, fn in METHODS.items():
                w = fn(expr, mi)
                g = lambda t: np.array([w[i, j] for i, j in pairs[t]])
                acc.setdefault(m, []).append(
                    (auroc(g(sp["pos"]), g(sp["neg"])), auroc(g("D"), g("N"))))
        P, Q = np.array(P), np.array(Q)
        d, sd = P.mean() - Q.mean(), 0.5 * (P.std() + Q.std())
        kr = tune_ref(name, link=a.link, counts_kw=kw, k=k)[0]
        report[key] = {"lib_loc": lib, "structure": name, "tag": a.tag, "correct": sp["correct"],
                       "floor": float(Phi(d / (sd * sqrt(2)))) if sd > 0 else 0.5,
                       "stats": counts_stats(name, 0, kw, k, kr),
                       "sec": round(time.time() - t0, 1),
                       "methods": {m: {"main": float(np.mean([x[0] for x in v])),
                                       "control": float(np.mean([x[1] for x in v]))}
                                   for m, v in acc.items()}}
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  lib={lib} {name:12} ({report[key]['sec']} с)", flush=True)

ms = list(METHODS)
for name in a.only.split(","):
    print(f"\n### {name}   (потрібно {SPEC[name]['correct']})")
    print(f"{'UMI/кл':>8}{'λ сер.':>8}{'детект':>8}{'нулі':>7}{'підлога':>9}"
          + "".join(f"{m[:11]:>13}" for m in ms))
    for k, r in sorted(report.items(), key=lambda kv: -kv[1]["lib_loc"]):
        if r["structure"] != name:
            continue
        st = r["stats"]
        print(f"{st['umi_median']:>8.0f}{st['lambda_mean']:>8.2f}"
              f"{st['detected_median']:>8.0f}{st['zero_frac']:>7.2f}{r['floor']:>9.3f}"
              + "".join(f"{r['methods'][m]['main']:>13.3f}" for m in ms))
print(f"\nout/{a.out}")
