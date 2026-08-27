import json, pathlib, sys, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from eval.columns import COLUMNS
from eval.metrics import auroc
from methods.aracne import aracne
from methods.conditional import partial_corr_matrix
from methods.marginal import corr_matrix, mi_matrix
import sim.chain as chain, sim.collider as collider, sim.confounder as confounder

CAL = json.load(open("data/real_calibration.json"))
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def build(name, seed, kw):
    if name == "1 спільна причина":
        return confounder.matched(seed, counts_kw=kw)[:2] + ("D", "C")
    if name == "2 прихований регулятор":
        return confounder.matched(seed, hide_root=True, counts_kw=kw)[:2] + ("D", "C")
    if name == "3 колайдер":
        return collider.build(seed, counts_kw=kw) + ("K", "N")
    if name == "4a ланцюг, B видно":
        return chain.matched(seed, False, counts_kw=kw)[:2] + ("D", "T")
    return chain.matched(seed, True, counts_kw=kw)[:2] + ("D", "T")


report = {}
for name, cal in CAL.items():
    kw = {"mean_shape": cal["mean_shape"], "mean_rate": cal["mean_rate"],
          "lib_loc": cal["lib_loc"]}
    report[name] = {"real_sparsity": cal["sparsity"], "n_genes_real": cal["n"],
                    "counts_kw": kw, "columns": {}}
    t0 = time.time()
    for label, _, correct in COLUMNS:
        acc, spars = {}, []
        for s in range(N_SEEDS):
            expr, idx, pos, neg = build(label, s, kw)
            spars.append(float((expr == 0).mean()))
            mi = mi_matrix(expr)
            ws = {"кореляція": np.abs(corr_matrix(expr)), "взаємна інф.": mi,
                  "часткова кор.": np.abs(partial_corr_matrix(expr)),
                  "ARACNe": aracne(expr, mi=mi)}
            for nm, w in ws.items():
                g = lambda t: np.array([w[i, j] for i, j in idx[t]])
                acc.setdefault(nm, []).append(
                    (auroc(g(pos), g(neg)), auroc(g("D"), g("N")) if "D" in idx else None))
        report[name]["columns"][label] = {
            "correct": correct, "sim_sparsity": float(np.mean(spars)),
            "methods": {nm: {"main": float(np.mean([x[0] for x in v])),
                             "sd": float(np.std([x[0] for x in v])),
                             "control": float(np.mean([x[1] for x in v]))
                             if v[0][1] is not None else None}
                        for nm, v in acc.items()}}
    print(f"{name:28} реальні нулі {cal['sparsity']:.3f} | "
          f"симуляція {report[name]['columns'][COLUMNS[0][0]]['sim_sparsity']:.3f} "
          f"({time.time()-t0:.0f} с)", flush=True)

pathlib.Path("out").mkdir(exist_ok=True)
json.dump(report, open("out/real_calibrated.json", "w"), ensure_ascii=False, indent=1)
print("\nout/real_calibrated.json")
