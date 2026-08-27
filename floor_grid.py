"""Підлога стенду для сітки структур grid.py."""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import numpy as np
from math import erf, sqrt
from sim.grid import SPEC, matched
from sim.counts import pair_corr

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--link", default="linear")
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--hide", action="store_true")
ap.add_argument("--only", default=None)
ap.add_argument("--noise", default="gauss", choices=["gauss","uniform","laplace","exp"])
ap.add_argument("--stat", default="corr", choices=["corr","mi"])
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows, t0 = {}, time.time()
names = a.only.split(",") if a.only else list(SPEC)
for name in names:
    sp = SPEC[name]
    P, Q = [], []
    for s in range(a.seeds):
        expr, pairs, k, tgt = matched(name, s, hide=a.hide, link=a.link, rho=a.rho, stat=a.stat, noise_dist=a.noise)
        P += [abs(pair_corr(expr, i, j)) for i, j in pairs[sp["pos"]]]
        Q += [abs(pair_corr(expr, i, j)) for i, j in pairs[sp["neg"]]]
    P, Q = np.array(P), np.array(Q)
    d = P.mean() - Q.mean()
    sd = 0.5 * (P.std() + Q.std())
    rows[name] = {"pos": sp["pos"], "neg": sp["neg"], "correct": sp["correct"],
                  "match": sp["match"], "m_pos": float(P.mean()), "m_neg": float(Q.mean()),
                  "delta": float(d), "sd": float(sd),
                  "floor": float(Phi(d / (sd * sqrt(2)))) if sd > 0 else 0.5,
                  "n_pairs": len(P)}

print(f"{'структура':13}{'пари':>9}{'m_pos':>9}{'m_neg':>9}{'Δ':>11}{'sd':>9}{'підлога':>10}")
print("-" * 70)
for n, r in rows.items():
    print(f"{n:13}{r['pos']+'/'+r['neg']:>9}{r['m_pos']:>9.4f}{r['m_neg']:>9.4f}"
          f"{r['delta']:>+11.5f}{r['sd']:>9.5f}{r['floor']:>10.4f}")
print(f"\n{time.time()-t0:.1f} с")

out = pathlib.Path("out"); out.mkdir(exist_ok=True)
fn = a.out or f"floor_grid_{a.noise}_{a.stat}_{a.link}{'_hidden' if a.hide else ''}{'_rho'+str(a.rho) if a.rho else ''}.json"
prev = json.loads((out / fn).read_text()) if (out / fn).exists() else {}
prev.update(rows)
(out / fn).write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"out/{fn}")
