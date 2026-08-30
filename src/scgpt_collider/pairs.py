import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.prep import load_trrust


def trrust_sets(genes):
    tr = load_trrust()
    gset = set(genes)
    tgt = tr.groupby("tf").target.apply(set).to_dict()
    tfs = [g for g in genes if g in tgt]
    tgt_in = {t: tgt[t] & gset for t in tfs}
    return tr, tfs, tgt, tgt_in


def pair_table(genes, present, min_co=1):
    tr, tfs, tgt, tgt_in = trrust_sets(genes)
    gi = {g: i for i, g in enumerate(genes)}
    P = present.astype(np.float32)
    tf_idx = np.array([gi[t] for t in tfs])
    co = P[:, tf_idx].T @ P[:, tf_idx]
    rows = []
    for a, b in itertools.combinations(range(len(tfs)), 2):
        x, y = tfs[a], tfs[b]
        n_co = int(co[a, b])
        if n_co < min_co:
            continue
        sh = tgt[x] & tgt[y]
        sh_in = tgt_in[x] & tgt_in[y]
        rows.append({
            "a": x, "b": y, "ia": gi[x], "ib": gi[y], "n_co": n_co,
            "n_shared": len(sh), "n_shared_in": len(sh_in), "shared": int(len(sh) > 0), "shared_in": int(len(sh_in) > 0),
            "direct": int((y in tgt[x]) or (x in tgt[y])),
            "deg": len(tgt[x]) + len(tgt[y]), "deg_in": len(tgt_in[x]) + len(tgt_in[y]),
            "n_a": int(P[:, gi[x]].sum()), "n_b": int(P[:, gi[y]].sum()),
        })
    return pd.DataFrame(rows), tfs, tgt, tgt_in


if __name__ == "__main__":
    import scanpy as sc
    a = sc.read_h5ad(Path("/Users/macbook/biodyn-bench/data/scgpt_collider/pbmc3k_prepped.h5ad"))
    genes = list(a.var_names)
    present = np.asarray(a.layers["X_binned"]) > 0
    df, tfs, tgt, tgt_in = pair_table(genes, present, min_co=0)
    print("tfs", len(tfs), "pairs", len(df), "A", df.shared.sum(), "B", (1 - df.shared).sum(), "A_in", df.shared_in.sum(), "direct", df.direct.sum())
    for k in (1, 5, 10, 20, 50):
        m = df.n_co >= k
        print("n_co>=", k, "pairs", int(m.sum()), "A", int(df.shared[m].sum()), "B", int((1 - df.shared[m]).sum()), "A_in", int(df.shared_in[m].sum()))
    gi = {g: i for i, g in enumerate(genes)}
    P = present.astype(np.float32)
    tri = []
    for r in df[(df.shared_in == 1) & (df.n_co >= 5)].itertuples():
        cs = [gi[c] for c in (tgt_in[r.a] & tgt_in[r.b])]
        both = P[:, r.ia] * P[:, r.ib]
        anyc = (P[:, cs].sum(1) > 0).astype(np.float32)
        tri.append(int((both * anyc).sum()))
    tri = np.array(tri)
    print("A_in pairs n_co>=5:", len(tri), "with >=5 cells where A,B,someC present:", int((tri >= 5).sum()), ">=10:", int((tri >= 10).sum()))
    print("n_co quantiles A", np.percentile(df.n_co[df.shared == 1], [50, 75, 90, 99]), "B", np.percentile(df.n_co[df.shared == 0], [50, 75, 90, 99]))
