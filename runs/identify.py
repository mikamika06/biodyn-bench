"""Неможливість зрівняти дві статистики ОДНОЧАСНО як міра ідентифіковності.

Ідея. Якщо дві структури можна зрівняти і за кореляцією, і за взаємною
інформацією одночасно — вони статистично нерозрізненні, і жоден метод їх не
відділить. Якщо не можна — інформація існує, і питання лише в тому, чи вміє
метод її взяти.

Тобто РОЗБІЖНІСТЬ між двома зрівнюваннями є прямою мірою ідентифіковності,
яку можна порахувати НЕ маючи жодного методу.
"""
import argparse, json, pathlib, sys
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np
from sim.grid import SPEC, matched, STATS

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="feedback_dir,confounder,chain,collider")
ap.add_argument("--noises", default="gauss,uniform,laplace,exp")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--stats", default="corr,mi,sq,partial")
ap.add_argument("--hide", action="store_true")
ap.add_argument("--out", default="identify.json")
a = ap.parse_args()

dst = pathlib.Path("out") / a.out
rep = json.loads(dst.read_text()) if dst.exists() else {}

for st in a.only.split(","):
    sp = SPEC[st]
    for nd in a.noises.split(","):
        key = f"{st}|{nd}|{a.stats}|{'hidden' if a.hide else 'visible'}"
        if key in rep:
            continue
        stats = a.stats.split(",")
        row = {}
        for stat in stats:
            v = []
            for seed in range(a.seeds):
                e, p, k, t = matched(st, seed, stat=stat, noise_dist=nd, hide=a.hide)
                v.append([STATS[s2](e, p[sp[t2]]) for s2 in stats for t2 in ("pos", "neg")])
            m = np.array(v).mean(0)
            row[stat] = {}
            for i, s2 in enumerate(stats):
                row[stat][s2] = {"pos": float(m[2 * i]), "neg": float(m[2 * i + 1]),
                                 "rel": float(abs(m[2 * i] - m[2 * i + 1]) /
                                              max(abs(m[2 * i]), 1e-9))}
        # залишок: найкраще, чого вдалось досягти — по зрівнюванню беремо
        # НАЙГІРШУ незрівняну статистику, потім мінімум по зрівнюваннях
        row["residual"] = float(min(max(row[s1][s2]["rel"] for s2 in stats if s2 != s1)
                                    for s1 in stats))
        row["hide"] = a.hide
        row["per_stat"] = {s1: {s2: row[s1][s2]["rel"] for s2 in stats} for s1 in stats}
        rep[key] = row
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {st:14}{nd:9} залишок {row['residual']:.4f}", flush=True)

stats = a.stats.split(",")
print(f"\n{'структура':12}{'шум':8}{'корінь':>9}" +
      "".join(f"{'зрівн ' + s:>11}" for s in stats) + f"{'залишок':>10}")
for k, r in rep.items():
    parts = k.split("|")
    st, nd = parts[0], parts[1]
    vis = parts[3] if len(parts) > 3 else "visible"
    if "per_stat" not in r:
        continue
    worst = [max(r["per_stat"][s1][s2] for s2 in stats if s2 != s1) for s1 in stats]
    print(f"{st:12}{nd:8}{vis:>9}" + "".join(f"{w:>11.4f}" for w in worst) +
          f"{r['residual']:>10.4f}")
print("\nзалишок ~ 0  структури нерозрізненні, жоден метод їх не відділить")
print("залишок > 0  інформація існує, питання лише в методі")
print(f"\nout/{a.out}")
