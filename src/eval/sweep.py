import json
import pathlib
import time
import numpy as np
import sim.chain as chain
import sim.collider as collider
import sim.confounder as confounder
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix
from methods.marginal import corr_matrix, mi_matrix
from methods.trees import genie3, grnboost2

OUT = pathlib.Path(__file__).resolve().parent.parent.parent / "out"

COLUMNS = [
    ("1 спільна причина",      lambda s, kw: confounder.matched(s, counts_kw=kw)[:2] + ("D", "C"),  "якнайвище"),
    ("2 прихований регулятор", lambda s, kw: confounder.matched(s, hide_root=True, counts_kw=kw)[:2] + ("D", "C"), "якнайвище"),
    ("3 колайдер",             lambda s, kw: collider.build(s, counts_kw=kw) + ("K", "N"),          "рівно 0.5"),
    ("4a ланцюг, B видно",     lambda s, kw: chain.matched(s, False, counts_kw=kw)[:2] + ("D", "T"), "якнайвище"),
    ("4b ланцюг, B приховано", lambda s, kw: chain.matched(s, True, counts_kw=kw)[:2] + ("D", "T"),  "якнайвище"),
]


def score_panel(expr, idx, pos, neg, seed, with_trees=False):
    mi = mi_matrix(expr)
    ws = {"кореляція": np.abs(corr_matrix(expr)),
          "взаємна інф.": mi,
          "часткова кор.": np.abs(partial_corr_matrix(expr)),
          "ARACNe": aracne(expr, mi=mi)}
    if with_trees:
        ws["GENIE3"] = genie3(expr, seed=1000 * seed)
        ws["GRNBoost2"] = grnboost2(expr, seed=1000 * seed)
    out = {}
    for nm, w in ws.items():
        g = lambda t: np.array([w[i, j] for i, j in idx[t]])
        out[nm] = {"main": auroc(g(pos), g(neg)),
                   "control": auroc(g("D"), g("N")) if "D" in idx else None}
    return out


def sweep(levels, n_seeds=20, with_trees=False, tag="sparsity"):
    report = {"levels": [], "n_seeds": n_seeds, "with_trees": with_trees}
    for lib_loc in levels:
        kw = None if lib_loc is None else {"lib_loc": lib_loc}
        entry = {"lib_loc": lib_loc, "columns": {}}
        t0 = time.time()
        for label, builder, correct in COLUMNS:
            acc, spars, umis = {}, [], []
            for s in range(n_seeds):
                expr, idx, pos, neg = builder(s, kw)
                spars.append(float((expr == 0).mean()))
                umis.append(float(np.median(np.expm1(expr).sum(axis=1))))
                for nm, v in score_panel(expr, idx, pos, neg, s, with_trees).items():
                    acc.setdefault(nm, []).append(v)
            entry["columns"][label] = {
                "correct": correct,
                "sparsity": float(np.mean(spars)),
                "methods": {nm: {"main_mean": float(np.mean([x["main"] for x in vs])),
                                 "main_sd": float(np.std([x["main"] for x in vs])),
                                 "control_mean": float(np.mean([x["control"] for x in vs]))
                                 if vs[0]["control"] is not None else None,
                                 "main": [x["main"] for x in vs]}
                            for nm, vs in acc.items()}}
            print(f"  lib_loc={lib_loc}  {label:24} нулі={entry['columns'][label]['sparsity']:.3f}"
                  f"  ({time.time()-t0:.0f} с)", flush=True)
        report["levels"].append(entry)
        OUT.mkdir(exist_ok=True)
        (OUT / f"sweep_{tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def table(report):
    lines = []
    methods = list(next(iter(report["levels"][0]["columns"].values()))["methods"])
    for label in report["levels"][0]["columns"]:
        correct = report["levels"][0]["columns"][label]["correct"]
        lines.append("")
        lines.append(f"── {label}   (правильно: {correct}) ──")
        lines.append(f"{'нулі':>7}" + "".join(f"{m[:12]:>14}" for m in methods))
        lines.append("-" * (7 + 14 * len(methods)))
        for lv in report["levels"]:
            c = lv["columns"][label]
            row = f"{c['sparsity']:>7.3f}"
            for m in methods:
                row += f"{c['methods'][m]['main_mean']:>14.3f}"
            lines.append(row)
    return "\n".join(lines)
