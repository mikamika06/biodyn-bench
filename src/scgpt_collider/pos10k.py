import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.engine import Engine
from scgpt_collider.pairs import trrust_sets
from scgpt_collider.stats import evaluate, strata

OUT = Path("/Users/macbook/biodyn-bench/out/scgpt_collider")


def edge_frame(eng, tgt, tgt_in, tfs, min_co):
    gi = {g: i for i, g in enumerate(eng.genes)}
    mean_expr = eng.xl.mean(0)
    targets = sorted({c for t in tfs for c in tgt_in[t]})
    tf_idx = np.array([gi[t] for t in tfs])
    tg_idx = np.array([gi[c] for c in targets])
    P = eng.present.astype(np.float32)
    co = P[:, tf_idx].T @ P[:, tg_idx]
    rows = []
    for i, t in enumerate(tfs):
        for j, c in enumerate(targets):
            if c != t and co[i, j] >= min_co:
                rows.append({"a": c, "b": t, "ia": gi[c], "ib": gi[t], "edge": int(c in tgt[t]), "n_co": int(co[i, j]), "deg": len(tgt[t]), "expr": np.log(mean_expr[gi[t]] + 1e-3) + np.log(mean_expr[gi[c]] + 1e-3)})
    return pd.DataFrame(rows)


def run_pairs(eng, E, n_cells, rng, mask_frac=0.25):
    rows, meta = [], []
    for r in E.itertuples():
        a, b = r.ia, r.ib
        cells = np.where(eng.present[:, a] & eng.present[:, b])[0]
        cells = rng.choice(cells, min(n_cells, len(cells)), replace=False)
        for c in cells:
            idx = np.where(eng.present[c])[0]
            others = idx[(idx != a) & (idx != b)]
            extra = set(rng.choice(others, min(int(mask_frac * len(idx)), len(others)), replace=False).tolist())
            pool = np.where(eng.present[:, b] & (eng.xb[:, b] != eng.xb[c, b]))[0]
            if len(pool) == 0:
                continue
            d = int(rng.choice(pool))
            rows.append((c, a, b, extra | {a}, {}))
            rows.append((c, a, b, extra | {a}, {b: eng.xb[d, b]}))
            meta.append({"pair": r.Index, "cell": int(c), "dv": float(abs(eng.xb[d, b] - eng.xb[c, b]))})
    print("rows", len(rows), flush=True)
    p, arp, apr = eng.run(rows)
    m = pd.DataFrame(meta)
    m["eff"] = np.abs(p[1::2] - p[0::2])
    m["att"] = arp[0::2].mean(1)
    g = m.groupby("pair").agg(
        eff=("eff", "mean"), eff_med=("eff", "median"),
        frac05=("eff", lambda x: float((x >= 0.5).mean())),
        frac1=("eff", lambda x: float((x >= 1.0).mean())),
        q90=("eff", lambda x: float(np.quantile(x, 0.9))),
        att=("att", "mean"), n_cells=("eff", "size")).reset_index().set_index("pair")
    return m, g


def main(n_edges=400, n_cells=50, min_co=100, seed=0):
    rng = np.random.default_rng(seed)
    eng = Engine(want_attn=True, fname="pbmc10k_prepped.h5ad")
    tr, tfs, tgt, tgt_in = trrust_sets(eng.genes)
    ed = edge_frame(eng, tgt, tgt_in, tfs, min_co)
    print("candidate pairs", len(ed), "edges", int(ed.edge.sum()), flush=True)
    ed["key"] = strata(np.log(ed.deg.values), ed.expr.values, np.log(ed.n_co.values), q=4)
    E1 = ed[ed.edge == 1].sample(min(n_edges, int(ed.edge.sum())), random_state=1)
    picks = []
    for k, cnt in E1.key.value_counts().items():
        cand = ed[(ed.edge == 0) & (ed.key == k)]
        picks.append(cand.sample(min(cnt, len(cand)), random_state=2))
    E = pd.concat([E1] + picks)
    print("selected", len(E), "edges", int(E.edge.sum()), flush=True)
    m, g = run_pairs(eng, E, n_cells, rng)
    E2 = E.join(g, how="inner")
    E2.to_csv(OUT / "pos10k_pairs.csv", index=False)
    m.to_csv(OUT / "pos10k_cells.csv", index=False)
    res = {"n": int(len(E2)), "n_edges": int(E2.edge.sum()), "n_cells_per_pair": n_cells, "min_co": min_co}
    for c in ("eff", "eff_med", "frac05", "frac1", "q90", "att"):
        res[c] = evaluate(E2, c, None, rng, n_rewire=0, n_perm=1000, label="edge")
    json.dump(res, open(OUT / "pos10k.json", "w"), indent=1)
    for c in ("eff", "eff_med", "frac05", "frac1", "q90", "att"):
        v = res[c]
        print(c, round(v["auroc"], 4), [round(x, 4) for x in v["auroc_ci95"]], "mwu", "%.2g" % v["mwu_p"], "strat", round(v["strat_null_mean"], 3), v["strat_p_ge"], "medA", "%.4g" % v["median_A"], "medB", "%.4g" % v["median_B"])


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 400, int(a[1]) if len(a) > 1 else 50, int(a[2]) if len(a) > 2 else 100)
