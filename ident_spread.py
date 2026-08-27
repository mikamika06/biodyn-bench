"""Розкид залишку по зернах — у чисел міри поки немає похибки."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import numpy as np
from sim.grid import SPEC, matched, STATS

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="confounder,feedback_dir,and")
ap.add_argument("--noises", default="gauss,laplace")
ap.add_argument("--stats", default="corr,mi,sq")
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--out", default="ident_spread.json")
a = ap.parse_args()

stats = a.stats.split(",")
rep = {}
print(f"{'стовпець':13}{'шум':8}{'середнє':>10}{'sd':>9}{'мін':>9}{'макс':>9}")
for st in a.only.split(","):
    sp = SPEC[st]
    for nd in a.noises.split(","):
        per = []
        for seed in range(a.seeds):
            row = {}
            for s1 in stats:
                e, p, k, t = matched(st, seed, stat=s1, noise_dist=nd)
                row[s1] = {s2: abs(STATS[s2](e, p[sp["pos"]]) - STATS[s2](e, p[sp["neg"]])) /
                               max(abs(STATS[s2](e, p[sp["pos"]])), 1e-9) for s2 in stats}
            per.append(min(max(row[s1][s2] for s2 in stats if s2 != s1) for s1 in stats))
        v = np.array(per)
        rep[f"{st}|{nd}"] = {"per_seed": v.tolist(), "mean": float(v.mean()),
                             "sd": float(v.std()), "min": float(v.min()),
                             "max": float(v.max())}
        print(f"{st:13}{nd:8}{v.mean():>10.4f}{v.std():>9.4f}{v.min():>9.4f}{v.max():>9.4f}",
              flush=True)
pathlib.Path("out").mkdir(exist_ok=True)
json.dump(rep, open(f"out/{a.out}", "w"), ensure_ascii=False, indent=1)
print(f"\nout/{a.out}")
