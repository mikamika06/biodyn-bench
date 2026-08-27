"""Дерева на нелінійності. Лише стовпці, де в лінійному режимі вони давали 1.000."""
import json, pathlib, sys, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from eval.columns import COLUMNS, set_link
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix
from methods.marginal import corr_matrix, mi_matrix
from methods.trees import genie3, grnboost2

WANT = ["1 спільна причина", "4a ланцюг, B видно"]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
dst = OUT / "trees_nonlinear.json"
report = json.loads(dst.read_text()) if dst.exists() else {}

for link in ("tanh", "relu"):
    for match in ("corr", "mi"):
        set_link(link, match)
        for label, builder, _ in COLUMNS:
            if label not in WANT:
                continue
            key = f"{link}|{match}|{label}"
            if key in report:
                print(f"пропуск {key}", flush=True); continue
            t0 = time.time(); acc = {}
            for s in range(SEEDS):
                expr, idx, pos, neg = builder(s)
                mi = mi_matrix(expr)
                ws = {"часткова кор.": np.abs(partial_corr_matrix(expr)),
                      "ARACNe": aracne(expr, mi=mi),
                      "GENIE3": genie3(expr, seed=1000 * s),
                      "GRNBoost2": grnboost2(expr, seed=1000 * s)}
                for nm, w in ws.items():
                    g = lambda t: np.array([w[i, j] for i, j in idx[t]])
                    acc.setdefault(nm, []).append(
                        (auroc(g(pos), g(neg)), auroc(g("D"), g("N")) if "D" in idx else None))
            report[key] = {"n_seeds": SEEDS, "sec": round(time.time() - t0, 1),
                           "methods": {nm: {"main": float(np.mean([x[0] for x in v])),
                                            "sd": float(np.std([x[0] for x in v])),
                                            "control": float(np.mean([x[1] for x in v]))
                                            if v[0][1] is not None else None}
                                       for nm, v in acc.items()}}
            dst.write_text(json.dumps(report, ensure_ascii=False, indent=1))
            r = report[key]["methods"]
            print(f"{key:46} " + "  ".join(f"{nm} {r[nm]['main']:.3f}" for nm in r)
                  + f"   ({report[key]['sec']} с)", flush=True)
print("\nout/trees_nonlinear.json")
