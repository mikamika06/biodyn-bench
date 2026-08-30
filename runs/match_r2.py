import argparse, json, pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
import numpy as np
from sim.grid import SPEC, matched
from eval.strength import strengths, r2_nodes, r2_central, match_subsets, matched_pairs_r2

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--panel", type=int, default=20)
ap.add_argument("--cells", type=int, default=2000)
ap.add_argument("--out", default="match_r2.json")
a = ap.parse_args()
cfg = {"n_cells": a.cells, "n_struct": a.panel, "n_direct": a.panel, "n_ref": a.panel, "n_null": a.panel}
report = {}
print("%-13s %6s %9s %9s %9s %9s" % ("структура", "n", "corr Δ", "R² Δ raw", "R² Δ corr", "R² Δ both"))
for name in SPEC:
    sp = SPEC[name]
    rows = []
    for seed in range(a.seeds):
        expr, pairs, k, tgt = matched(name, seed, cfg)
        pa, pb = pairs[sp["pos"]], pairs[sp["neg"]]
        r2 = r2_nodes(expr)
        sa, sb = strengths(expr, pa), strengths(expr, pb)
        ra, rb = r2_central(expr, pa, r2), r2_central(expr, pb, r2)
        rng = np.random.default_rng(seed)
        ia, ib = match_subsets(sa, sb, rng)
        raw_gap = float(ra.mean() - rb.mean())
        corr_gap = float(sa[ia].mean() - sb[ib].mean()) if len(ia) else float("nan")
        r2_after_corr = float(ra[ia].mean() - rb[ib].mean()) if len(ia) else float("nan")
        _, _, st = matched_pairs_r2(expr, pa, pb, seed=seed)
        rows.append({"seed": seed, "n_corr": int(len(ia)), "n_both": st.get("n", 0), "corr_gap": corr_gap,
                     "r2_gap_raw": raw_gap, "r2_gap_after_corr": r2_after_corr, "r2_gap_after_both": st.get("r2_gap")})
    m = lambda key: float(np.nanmean([r[key] for r in rows if r[key] is not None]))
    report[name] = {"pos": sp["pos"], "neg": sp["neg"], "rows": rows,
                    "mean": {k: m(k) for k in ("corr_gap", "r2_gap_raw", "r2_gap_after_corr", "r2_gap_after_both")}}
    print("%-13s %6d %9.3f %9.3f %9.3f %9.3f" % (name, int(np.mean([r["n_both"] for r in rows])), m("corr_gap"), m("r2_gap_raw"), m("r2_gap_after_corr"), m("r2_gap_after_both")))
pathlib.Path("out").mkdir(exist_ok=True)
pathlib.Path("out", a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("out/" + a.out)
