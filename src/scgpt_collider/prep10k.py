import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.prep import OUT, ROOT, VOCAB, bin_row, load_trrust


def main(n_hvg=1200, seed=0, max_tokens=1199):
    vocab = json.load(open(VOCAB))
    tr = load_trrust()
    a = sc.read_10x_h5(ROOT / "data/pbmc10k_v3.h5")
    a.var_names_make_unique()
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    a = a[:, a.var_names.isin(set(vocab))].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts")
    tfs = set(tr.tf)
    keep = a.var["highly_variable"].to_numpy() | a.var_names.isin(tfs)
    a = a[:, keep].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    a.layers["X_log1p"] = X.astype(np.float32)
    np.random.seed(seed)
    xb = np.stack([bin_row(r) for r in X]).astype(np.float32)
    trunc = 0
    for c in range(xb.shape[0]):
        nz = np.where(xb[c] > 0)[0]
        if len(nz) > max_tokens:
            drop = nz[np.argsort(X[c, nz])[:len(nz) - max_tokens]]
            xb[c, drop] = 0.0
            trunc += 1
    a.layers["X_binned"] = xb
    a.var["is_tf"] = a.var_names.isin(tfs)
    a.var["vocab_id"] = [vocab[g] for g in a.var_names]
    rng = np.random.default_rng(seed)
    a.obs["order"] = rng.permutation(a.n_obs)
    a = a[np.argsort(a.obs["order"].to_numpy())].copy()
    a.write_h5ad(OUT / "pbmc10k_prepped.h5ad")
    genes = list(a.var_names)
    tf_in = [g for g in genes if g in tfs]
    tgt_of = tr.groupby("tf").target.apply(set).to_dict()
    gset = set(genes)
    nz = (xb > 0).sum(1)
    info = {
        "n_cells": int(a.n_obs), "n_genes": len(genes), "n_hvg": int(a.var["highly_variable"].sum()),
        "n_tf": len(tf_in), "n_tf_with_target_in_geneset": sum(1 for t in tf_in if tgt_of[t] & gset),
        "n_cells_truncated": trunc, "median_tokens": float(np.median(nz)), "max_tokens": int(nz.max()),
        "median_genes_per_cell_full": float(np.median(a.obs["n_genes"].to_numpy())),
        "tf_frac_cells_nonzero_median": float(np.median((xb[:, a.var["is_tf"].to_numpy()] > 0).mean(0))),
    }
    json.dump(info, open(OUT / "prep10k_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
