import json
import pathlib
import time
import numpy as np
from eval.columns import COLUMNS
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix
from methods.marginal import corr_matrix, mi_matrix
from methods.trees import genie3, grnboost2

CHEAP_SEEDS = 20
TREE_SEEDS = 10
OUT = pathlib.Path(__file__).resolve().parent.parent / "out"

CHEAP = {
    "кореляція":     lambda e, s: np.abs(corr_matrix(e)),
    "взаємна інф.":  lambda e, s: mi_matrix(e),
    "часткова кор.": lambda e, s: np.abs(partial_corr_matrix(e)),
    "ARACNe":        lambda e, s: aracne(e),
}
TREE = {
    "GENIE3":    lambda e, s: genie3(e, seed=1000 * s),
    "GRNBoost2": lambda e, s: grnboost2(e, seed=1000 * s),
}
ALL = {**CHEAP, **TREE}


def evaluate(w, idx, pos, neg):
    g = lambda t: np.array([w[i, j] for i, j in idx[t]])
    out = {"main": auroc(g(pos), g(neg)),
           "m_pos": float(g(pos).mean()), "m_neg": float(g(neg).mean())}
    if "D" in idx and "N" in idx:
        out["control"] = auroc(g("D"), g("N"))
    return out


def run(only=None, verbose=True):
    report = {}
    for label, builder, correct in COLUMNS:
        report[label] = {"correct": correct, "methods": {}}
        panels = {}
        for name, fn in ALL.items():
            if only and name not in only:
                continue
            n = TREE_SEEDS if name in TREE else CHEAP_SEEDS
            acc, t0 = [], time.time()
            for s in range(n):
                if s not in panels:
                    panels[s] = builder(s)
                expr, idx, pos, neg = panels[s]
                acc.append(evaluate(fn(expr, s), idx, pos, neg))
            report[label]["methods"][name] = {
                "n_seeds": n, "sec": round(time.time() - t0, 1),
                **{k: [a[k] for a in acc] for k in acc[0]}}
            if verbose:
                v = report[label]["methods"][name]["main"]
                print(f"  {label:24}{name:16}{np.mean(v):.3f} ± {np.std(v):.3f}"
                      f"   ({report[label]['methods'][name]['sec']} с)", flush=True)
        panels.clear()
    return report


def table(report):
    names = [n for n in ALL if n in next(iter(report.values()))["methods"]]
    lines = ["", "=" * 96,
             f"{'метод':16}" + "".join(f"{c[0][:13]:>16}" for c in COLUMNS),
             f"{'правильно':16}" + "".join(f"{c[2]:>16}" for c in COLUMNS),
             "-" * 96]
    for nm in names:
        row = f"{nm:16}"
        for label, _, _ in COLUMNS:
            v = report[label]["methods"][nm]["main"]
            row += f"{np.mean(v):>10.3f}±{np.std(v):.2f}"
        lines.append(row)
    return "\n".join(lines)


def save(report, name="bench_classical.json"):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return OUT / name
